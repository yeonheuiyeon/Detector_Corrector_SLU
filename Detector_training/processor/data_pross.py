import os
import copy
import json
import logging
import re
import torch
from torch.utils.data import TensorDataset
from processor.aligner import Levenshtein,ExpandedAlignment

logger = logging.getLogger(__name__)

class InputExample(object):
    """
    A single training/test example for simple sequence classification.
    """

    def __init__(self, guid,realori, original,asr):
        self.guid = guid
        self.realori = realori
        self.original = original
        self.asr = asr

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask, token_type_ids,mask_ids,real_ids):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.mask_ids = mask_ids
        self.real_ids=real_ids


    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def seq_cls_convert_examples_to_features(args, examples, tokenizer, max_length):
    processor = MultiProcessor(args)
    logger.info("Using label list {}".format('hello'))
    all_inputs=[]
    all_attions=[]
    all_token_type=[]
    all_labels=[]
    for words, labels in zip([example.original for example in examples], [example.asr for example in examples]):
        tokens = []
        label_ids = []
        for word, label in zip(words, labels):

            word_tokens = tokenizer.tokenize(word)
            # print(len(word_tokens))
            if not word_tokens:
                word_tokens = [tokenizer.unk_token]  # For handling the bad-encoded word
            # tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            tokens.extend(word_tokens)
            if label == '[MASK]':
                label_ids.extend([tokenizer.mask_token] * (len(word_tokens)))
            else:
                label_ids.extend(word_tokens)

        special_tokens_count = 2
        if len(tokens) > 128 - special_tokens_count:
            tokens = tokens[:(128 - special_tokens_count)]
            label_ids = label_ids[:(128 - special_tokens_count)]
        tokens += [tokenizer.sep_token]
        label_ids += [tokenizer.sep_token]
        tokens = [tokenizer.cls_token] + tokens
        label_ids = [tokenizer.cls_token] + label_ids
        token_type_ids = [0] * len(tokens)
        inputs = tokenizer.convert_tokens_to_ids(tokens)
        label_ids=tokenizer.convert_tokens_to_ids(label_ids)
        attention_mask = [1] * len(inputs)
        padding_length = 128 - len(inputs)
        inputs += [tokenizer.pad_token_id] * padding_length
        attention_mask += [0] * padding_length
        token_type_ids += [0] * padding_length
        label_ids += [0] * padding_length
        # 여기서 부터 보기 배치 시발
        assert len(inputs) == 128
        assert len(attention_mask) == 128
        assert len(token_type_ids) == 128
        assert len(label_ids) == 128
        all_inputs.append(inputs)
        all_labels.append(label_ids)
        all_attions.append(attention_mask)
        all_token_type.append(token_type_ids)
    batch_encoding_realoriginal = tokenizer.batch_encode_plus(
        [example.realori for example in examples],
        max_length=max_length,
        padding="max_length",
        add_special_tokens=True,
        truncation=True,
    )



    features = []
    for i in range(len(examples)):
        #inputs = {k: batch_encoding_semioriginal[k][i] for k in batch_encoding_semioriginal}
        inputs_for_real = {k: batch_encoding_realoriginal[k][i] for k in batch_encoding_realoriginal}
        #inputs_asr = {k: batch_encoding_asr[k][i] for k in batch_encoding_asr}

        feature = InputFeatures(input_ids=all_inputs[i], attention_mask=all_attions[i], token_type_ids=all_token_type[i], mask_ids=all_labels[i],real_ids=inputs_for_real['input_ids'])

        features.append(feature)

    for i, example in enumerate(examples[-10:]):
        logger.info("*** Example ***")
        logger.info("guid: {}".format(example.guid))
        logger.info("input_ids: {}".format(" ".join([str(x) for x in features[i].input_ids])))
        logger.info("attention_mask: {}".format(" ".join([str(x) for x in features[i].attention_mask])))
        logger.info("token_type_ids: {}".format(" ".join([str(x) for x in features[i].token_type_ids])))
        logger.info("masked_ids: {}".format(features[i].mask_ids))
        logger.info("real_input_ids: {}".format(features[i].real_ids))

    return features

class MultiProcessor(object):
    """Processor for the NSMC data set """

    def __init__(self, args):
        self.args = args


    @classmethod
    def _read_file(cls, input_file):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                lines.append(line.strip())
            return lines

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            line = line.strip().split("\t")
            guid = "%s-%s" % (set_type, i)
            #print(line)
            try:
                original = line[0].strip()
                #original=re.sub(r'[^\w\s]', '', original)
                asr = line[1].strip()
            except:
                continue
            #asr = re.sub(r'[^\w\s]', '', asr)
            ref = [x for x in original.strip().split(' ') if x]
            hyp = [x for x in asr.strip().split(' ') if x]
            lev = Levenshtein.align(ref,hyp)
            lev.editops()
            find = lev.expandAlign()
            widths = [max(len(find.s1[i]), len(find.s2[i])) for i in range(len(find.s1))]
            s1_args = zip(widths, find.s1)
            s2_args = zip(widths, find.s2)
            align_args = zip(widths, find.align)
            listofs1_args = [x for x in s1_args]
            listofs2_args = [x for x in s2_args]
            listofalign = [x for x in align_args]
            newinputs = []
            inputs_id = []
            for idx, hyps in enumerate(listofalign):
                if not hyps[1] == 'D':
                    inputs_id.append(listofs2_args[idx][1])
                    if hyps[1] == 'I' or hyps[1] == 'S':
                        newinputs.append('[MASK]')
                    else:
                        newinputs.append(listofs2_args[idx][1])

            noise=inputs_id
            asr_masked=newinputs
            if i % 10000 == 0:
                logger.info(' '.join(noise))
                logger.info(' '.join(asr_masked))
            examples.append(InputExample(guid=guid, realori=original,original=noise, asr=asr_masked))
        return examples

    def get_examples(self, mode):
        """
        Args:
            mode: train, dev, test
        """
        file_to_read = None
        if mode == "train":
            file_to_read = self.args.train_file
        elif mode == "dev":
            file_to_read = self.args.dev_file
        elif mode == "test":
            file_to_read = self.args.test_file

        logger.info("LOOKING AT {}".format(os.path.join(self.args.data_dir,  file_to_read)))
        return self._create_examples(
            self._read_file(os.path.join(self.args.data_dir,  file_to_read)), mode
        )



def seq_cls_load_and_cache_examples(args, tokenizer, mode):
    processor = MultiProcessor(args)
    output_mode ='classification'
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}".format(
            list(filter(None, args.model_name_or_path.split("/"))).pop(), str(args.max_seq_len), mode
        ),
    )
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if mode == "train":
            examples = processor.get_examples("train")
        elif mode == "dev":
            examples = processor.get_examples("dev")
        elif mode == "test":
            examples = processor.get_examples("test")
        else:
            raise ValueError("For mode, only train, dev, test is avaiable")
        features = seq_cls_convert_examples_to_features(
            args, examples, tokenizer, max_length=args.max_seq_len)
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    #print(all_input_ids.shape)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_token_masked_id = torch.tensor([f.mask_ids for f in features], dtype=torch.long)
    all_token_real_id = torch.tensor([f.real_ids for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_token_masked_id,all_token_real_id)
    return dataset

