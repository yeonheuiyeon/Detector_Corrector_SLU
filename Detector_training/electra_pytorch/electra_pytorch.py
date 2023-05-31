import math
from functools import reduce
from collections import namedtuple
from processor import Levenshtein
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
# constants

Results = namedtuple('Results', [
    'loss',
    'mlm_loss',
    'd_loss',
])

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

def log(t, eps=1e-9):
    return torch.log(t + eps)

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1.):
    return ((t / temperature) + gumbel_noise(t)).argmax(dim=-1)

def prob_mask_like(t, prob):
    return torch.zeros_like(t).float().uniform_(0, 1) < prob

def mask_with_tokens(t, token_ids):
    init_no_mask = torch.full_like(t, False, dtype=torch.bool)
    mask = reduce(lambda acc, el: acc | (t == el), token_ids, init_no_mask)
    return mask

def get_mask_subset_with_prob(mask, prob):
    batch, seq_len, device = *mask.shape, mask.device
    max_masked = math.ceil(prob * seq_len)

    num_tokens = mask.sum(dim=-1, keepdim=True)
    mask_excess = (mask.cumsum(dim=-1) > (num_tokens * prob).ceil())
    mask_excess = mask_excess[:, :max_masked]

    rand = torch.rand((batch, seq_len), device=device).masked_fill(~mask, -1e9)
    _, sampled_indices = rand.topk(max_masked, dim=-1)
    sampled_indices = (sampled_indices + 1).masked_fill_(mask_excess, 0)

    new_mask = torch.zeros((batch, seq_len + 1), device=device)
    new_mask.scatter_(-1, sampled_indices, 1)
    return new_mask[:, 1:].bool()



def dis_label_maker(generator_prediction, d_label,tokenizer):
    max_seq_length=128
    features=[]

    for batches, content in enumerate(generator_prediction):

        #디스크리미네이터라벨 만드는 정답값
        ref=tokenizer.decode(d_label[batches],skip_special_tokens=True)
        #제너레이터로부터 생성된 결과 값
        hyp=tokenizer.decode(content,skip_special_tokens=True)
        ref = [x for x in ref.strip().split(' ') if x]
        hyp = [x for x in hyp.strip().split(' ') if x]

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
        #flag=1
        #for idx, hyps in enumerate(listofalign):
        #    if hyps[1] == 'D':
        #        try:
        #            listofalign[idx + 1][1] = 'S'
        #        except:
        #            try:
        #                inputs_label[idx-1] = 'S'
        #            except:
        #                flag=0
        #    else:
        #        inputs_id.append(listofs2_args[idx][1])
        #        inputs_label.append(hyps[1])
        #if flag==0:
        #    inputs_label[0]='S'
        flag = 0
        for idx, hyps in enumerate(listofalign):
            try:
                if hyps[1] == 'D' and listofalign[idx + 1][1] != 'D':
                    if len(inputs_label) > 0:
                        inputs_label[-1] = 'S'
                    else:
                        flag = 1
                elif hyps[1] == 'D' and listofalign[idx + 1][1] == 'D':
                    continue
                else:
                    inputs_id.append(listofs2_args[idx][1])
                    inputs_label.append(hyps[1])
            except:
                try:
                    inputs_label[-1] = 'S'
                except:
                    print(ref, hyp, inputs_label, inputs_id)
                    break
        if flag == 1:
            inputs_label[0] = 'S'

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

        special_tokens_count = 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[:(max_seq_length - special_tokens_count)]
            label_ids = label_ids[:(max_seq_length - special_tokens_count)]
        tokens += [tokenizer.sep_token]
        label_ids += [0]
        tokens = [tokenizer.cls_token] + tokens
        label_ids = [0] + label_ids

        token_type_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        attention_mask = [1] * len(input_ids)

        padding_length = max_seq_length - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * padding_length
        attention_mask += [0] * padding_length
        token_type_ids += [0] * padding_length
        label_ids += [0] * padding_length


        assert len(input_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        assert len(token_type_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        features.append(
            dis_Features(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          label_ids=label_ids)
        )

    dis_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    dis_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    dis_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    dis_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    #dataset = TensorDataset(dis_input_ids, dis_attention_mask, dis_token_type_ids, dis_label_ids)
    return dis_input_ids, dis_attention_mask, dis_token_type_ids, dis_label_ids

# hidden layer extractor class, for magically adding adapter to language model to be pretrained

class HiddenLayerExtractor(nn.Module):
    def __init__(self, net, layer = -2):
        super().__init__()
        self.net = net
        self.layer = layer

        self.hidden = None
        self.hook_registered = False

    def _find_layer(self):
        if type(self.layer) == str:
            modules = dict([*self.net.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*self.net.children()]
            return children[self.layer]
        return None

    def _hook(self, _, __, output):
        self.hidden = output

    def _register_hook(self):
        layer = self._find_layer()
        assert layer is not None, f'hidden layer ({self.layer}) not found'
        handle = layer.register_forward_hook(self._hook)
        self.hook_registered = True

    def forward(self, x):
        if self.layer == -1:
            return self.net(x)

        if not self.hook_registered:
            self._register_hook()

        _ = self.net(x)
        hidden = self.hidden
        self.hidden = None
        assert hidden is not None, f'hidden layer {self.layer} never emitted an output'
        return hidden

# main electra class

class Electra(nn.Module):
    def __init__(
        self,
        generator,
        discriminator,
        *,
        num_tokens = None,
        discr_dim = -1,
        discr_layer = -1,
        mask_prob = 0,
        replace_prob = 0,
        random_token_prob = 0,
        mask_token_id = 2,
        pad_token_id = 0,
        mask_ignore_token_ids = [],
        disc_weight = 50.,
        gen_weight = 1.,
        temperature = 1.,
        tokenizer=AutoTokenizer.from_pretrained("google/electra-small-generator")):
        super().__init__()

        self.generator = generator
        self.discriminator = discriminator

        if discr_dim > 0:
            self.discriminator = nn.Sequential(
                HiddenLayerExtractor(discriminator, layer = discr_layer),
                nn.Linear(discr_dim, 1)
            )

        # mlm related probabilities
        self.mask_prob = mask_prob
        self.replace_prob = replace_prob

        self.num_tokens = num_tokens
        self.random_token_prob = random_token_prob

        # token ids
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.mask_ignore_token_ids = set([*mask_ignore_token_ids, pad_token_id])

        # sampling temperature
        self.temperature = temperature

        # loss weights
        self.disc_weight = disc_weight
        self.gen_weight = gen_weight
        self.tokenizer=tokenizer


    def forward(self, input_ids=None, attention_mask=None,token_type_ids=None,mask_ids=None,real_ids=None,**kwargs):
        labels = torch.where(mask_ids.view(-1) == self.tokenizer.mask_token_id, input_ids.view(-1), -100)
        #print('라벨_오리지널_모양 :', labels.shape)
        #print('라벨_값 :', labels)
        # get generator output and get mlm loss
        logits = self.generator(input_ids=mask_ids, attention_mask=attention_mask,token_type_ids=token_type_ids,**kwargs).logits
        #print('mlm_logit_모양 :', logits.shape)
        #print('mlm_logit값 :', logits.view(-1, 30522))
        mlm_loss = F.cross_entropy(
            logits.view(-1, 30522),
            labels,
            ignore_index = -100
        )
        #print(mlm_loss)
        #print(logits.shape)
        #print(real_ids.shape)
        #여기서도 batch encoding 적용하기

        reshape_logits=[]
        for idxs,logit_indi in enumerate(logits):
            indi_logit=logit_indi.argmax(axis=-1)
            #if idxs <3:
            #    print('로짓값모양 :',indi_logit.shape)
            #    print('로짓값 :',indi_logit)
            #    print('ids 모양:',input_ids[idxs].shape)
            #    print('ids 값 :',input_ids[idxs])
            indi_logit=torch.where((input_ids[idxs].view(-1) == 103)|(input_ids[idxs].view(-1) == 101)|(input_ids[idxs].view(-1) == 0)|(input_ids[idxs].view(-1) == 102), 0,indi_logit.view(-1))
           # if idxs <3:
           #     print('변환 로짓:',indi_logit)
            reshape_logits.append(indi_logit)

        dis_input_ids, dis_attention_mask, dis_token_type_ids, dis_label_ids=dis_label_maker(reshape_logits, real_ids, self.tokenizer)
        disc_logits = self.discriminator(input_ids=dis_input_ids.to('cuda'), attention_mask=dis_attention_mask.to('cuda'),token_type_ids=dis_token_type_ids.to('cuda'), labels=dis_label_ids.to('cuda'), **kwargs)
        d_loss=disc_logits.loss
        #print(d_loss)
        return Results(self.gen_weight * mlm_loss + self.disc_weight * d_loss, mlm_loss, d_loss)
