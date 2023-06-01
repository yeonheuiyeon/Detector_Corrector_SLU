from transformers import ElectraForPreTraining, AutoTokenizer,ElectraConfig,AutoConfig
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import math
import time
from functools import reduce
from collections import namedtuple
from processor import Levenshtein
import torch
from torch import nn
import torch.nn.functional as F
import re
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

class dis_Features(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask, token_type_ids,label_ids):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label_ids = label_ids

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

def dis_label_maker(generator_prediction, d_label,tokenizer):
    max_seq_length=128
    features=[]

    ref = [x for x in d_label.strip().split(' ') if x]
    hyp = [x for x in generator_prediction.strip().split(' ') if x]
    lev = Levenshtein.align(ref,hyp)  # , reserve_list=PowerAligner.reserve_list, exclusive_sets=PowerAligner.exclusive_sets)
    lev.editops()
    find = lev.expandAlign()
    widths = [max(len(find.s1[i]), len(find.s2[i])) for i in range(len(find.s1))]
    s1_args = zip(widths, find.s1)
    s2_args = zip(widths, find.s2)
    align_args = zip(widths, find.align)
    listofs1_args = [x for x in s1_args]
    listofs2_args = [x for x in s2_args]
    listofalign = [x for x in align_args]
    inputs_id = []
    inputs_label = []
    flag=1
    for idx, hyps in enumerate(listofalign):
        if hyps[1] == 'D':
            try:
                listofalign[idx + 1][1] = 'S'
            except:
                try:
                    inputs_label[idx-1] = 'S'
                except:
                    flag=0
        else:
            inputs_id.append(listofs2_args[idx][1])
            inputs_label.append(hyps[1])
    if flag==0:
        try:
            inputs_label[0]='S'
        except:
            print(generator_prediction, d_label)
    tokens = []
    label_ids = []
    for word, label in zip(inputs_id, inputs_label):
        labels_map = {'S': 1, 'D': 1, 'C': 0, 'I': 1}
        word_tokens = tokenizer.tokenize(word)
        #print(len(word_tokens))
        if not word_tokens:
            word_tokens = [tokenizer.unk_token]  # For handling the bad-encoded word
        #tokens.extend(word_tokens)
        # Use the real label id for the first token of the word, and padding ids for the remaining tokens
        tokens.extend(word_tokens)
        label_ids.extend([labels_map[label]] * (len(word_tokens)))

    label_ids += [0]
    label_ids = [0] + label_ids




    dis_label_ids = torch.tensor([label_ids], dtype=torch.long)
    #dataset = TensorDataset(dis_input_ids, dis_attention_mask, dis_token_type_ids, dis_label_ids)
    return dis_label_ids

def main(refs, asrs):
    all_labels=[]
    all_predict=[]
    all_predict_n=[]
    discriminator=ElectraForPreTraining.from_pretrained(f"./models/epoch_dis_5")
    discriminator.to('cuda')
    tokenizer = AutoTokenizer.from_pretrained("google/electra-base-discriminator")
    total_start_time = time.time()
    for idx,content in enumerate(refs):
        dis_label_ids=dis_label_maker(asrs[idx],content,tokenizer)
        f_inputs=tokenizer.encode(asrs[idx], return_tensors="pt")
        discriminator_outputs = discriminator(f_inputs.to('cuda'))
        predictions = torch.round((torch.sign(discriminator_outputs[0]) + 1) / 2)
        if idx<3:
            print(content)
            print(asrs[idx])
            print(len(asrs[idx]))
            print(f_inputs)
            print(predictions.squeeze().tolist())
            print(len(predictions.squeeze().tolist()))
            print(dis_label_ids.squeeze().tolist())
            print(len(dis_label_ids.squeeze().tolist()))
        all_predict.extend(predictions.squeeze().tolist())
        all_predict_n.append([str(c) for c in predictions.squeeze().tolist()])
        all_labels.extend(dis_label_ids.squeeze().tolist())
    total_end_time = time.time()
    print((total_end_time - total_start_time))
    print(accuracy_score(all_labels,all_predict))
    print(f1_score(all_labels, all_predict))

    return all_predict_n



if __name__ == '__main__':
        for i in ['train', 'test']:
            f = open(f"./data/{i}_er_whisper.txt", 'r')
            lines = f.readlines()
            refs = []
            asrs = []
            labels = []
            for line in lines:
                line = line.strip().split("\t")
                try:
                    original = line[0].strip()
                    asr = line[1].strip()
                    label = line[2].strip()
                    refs.append(original.lower())
                    asrs.append(asr.lower())
                    labels.append(label)
                except:
                    print(line)
            f.close()
            assert len(refs) == len(asrs)
            predictions = main(refs, asrs)
            with open(f"/data/result_{i}_er_whisper.txt", "w") as f:
                for idx, content in enumerate(refs):
                    f.write(content + '\t' + asrs[idx] + '\t' + ','.join(predictions[idx]).strip() + '\t' + labels[idx] + '\n')

        # lib_test
        #for i in ['whisper']:
        #    f = open(f"./data/test_set_asr_{i}.txt", 'r')
        #    lines = f.readlines()
        #    refs = []
        #    asrs = []
        #    # labels=[]
        #    for line in lines:
        #        line = line.strip().split("\t")
        #        try:
        #            original = line[0].strip()
        #            #original = re.sub(r'[^\w\s]', '', original)
        #            asr = line[1].strip()
        #            #asr = re.sub(r'[^\w\s]', '', asr)
        #            # label=line[2].strip()
        #            refs.append(original.lower())
        #            asrs.append(asr.lower())
        #            # labels.append(label)
        #        except:
        #            print(line)
        #    f.close()
        #    predictions = main(refs, asrs)
        #    with open(f"./data/result_test_set_asr_{i}.txt", "w") as f:
        #        for idx, content in enumerate(refs):
        #            f.write(content + '\t' + asrs[idx] + '\t' + ','.join(predictions[idx]).strip() + '\n')


        # intent_classification
        #for i in ['test','train']:
        #    f = open(f"./data/slurp_{i}_whisper.txt", 'r')
        #    lines = f.readlines()
        #    refs = []
        #    asrs = []
        #    labels = []
        #    idxs = []
        #    for line in lines:
        #        line = line.strip().split("\t")
        #        try:
        #            original = line[1].strip()
        #            #original = re.sub(r'[^\w\s]', '', original)
        #            asr = line[2].strip()
        #            #asr = re.sub(r'[^\w\s]', '', asr)
        #            label = line[3].strip()
        #            refs.append(original.lower())
        #            asrs.append(asr.lower())
        #            labels.append(label)
        #            idxs.append(line[0].strip())
        #        except:
        #            print(line)
        #    f.close()
        #    assert len(refs) == len(asrs)
        #    predictions = main(refs, asrs)
            #with open(f"./data/result_slurp_{i}_whisper.txt", "w") as f:
            #    for idx, content in enumerate(refs):
            #        f.write(idxs[idx] + '\t' + content + '\t' + asrs[idx] + '\t' + ','.join(
            #            predictions[idx]).strip() + '\t' + labels[idx] + '\n')
