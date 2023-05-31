import argparse
import json
import logging
import os
import glob
import re
from seqeval import metrics as seqeval_metrics
from sklearn import metrics as sklearn_metrics
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from fastprogress.fastprogress import master_bar, progress_bar
from attrdict import AttrDict
from transformers import AutoConfig, AutoTokenizer,BertConfig,ElectraForMaskedLM,ElectraForPreTraining,AutoModelForMaskedLM
from transformers.trainer_utils import is_main_process
import logging
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup
)
from processor import seq_cls_load_and_cache_examples as load_and_cache_examples
from processor import MultiProcessor
from electra_pytorch import Electra
import random

def init_logger():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )


logger = logging.getLogger(__name__)

def simple_accuracy(labels, preds):
    return (labels == preds).mean()
def acc_score(labels, preds):
    return {
        "acc": simple_accuracy(labels, preds),
    }
def f1_pre_rec(labels, preds, is_ner=True):
    if is_ner:
        return {
            "precision": seqeval_metrics.precision_score(labels, preds, suffix=True),
            "recall": seqeval_metrics.recall_score(labels, preds, suffix=True),
            "f1": seqeval_metrics.f1_score(labels, preds, suffix=True),
        }
    else:
        return {
            "precision": sklearn_metrics.precision_score(labels, preds, average="macro"),
            "recall": sklearn_metrics.recall_score(labels, preds, average="macro"),
            "f1": sklearn_metrics.f1_score(labels, preds, average="macro"),
        }

def train(args,
          model,
          train_dataset,
          dev_dataset=None,
          test_dataset=None):
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(t_total * args.warmup_proportion), num_training_steps=t_total)

    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Total train batch size = %d", args.train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Logging steps = %d", args.logging_steps)
    logger.info("  Save steps = %d", args.save_steps)

    global_step = 0
    tr_loss = 0.0

    model.zero_grad()
    mb = master_bar(range(int(args.num_train_epochs)))
    for epoch in mb:
        epoch_iterator = progress_bar(train_dataloader, parent=mb)
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "mask_ids":batch[3],
                "real_ids": batch[4],
            }
            if args.model_type not in ["distilkobert", "xlm-roberta"]:
                inputs["token_type_ids"] = batch[2]  # Distilkobert, XLM-Roberta don't use segment_ids
            outputs = model(**inputs)
            #loss=round(outputs.loss.item(), 2)
            loss, loss_mlm,loss_d = outputs.loss, outputs.mlm_loss, outputs.d_loss
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0 or (
                    len(train_dataloader) <= args.gradient_accumulation_steps
                    and (step + 1) == len(train_dataloader)
            ):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1
            if args.max_steps > 0 and global_step > args.max_steps:
                break

        #if args.evaluate_test_during_training:
        #    evaluate(args, model, test_dataset, "test", epoch + 1)
        #    evaluate(args, model, dev_dataset, "dev", epoch + 1)

        output_dir = os.path.join(args.output_dir, "epoch_dis_{}".format(epoch+1))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model.discriminator.save_pretrained(output_dir)
        torch.save(args, os.path.join(output_dir, "training_args.bin"))
        logger.info("Saving model checkpoint to {}".format(output_dir))
        if args.save_optimizer:
            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
            logger.info("Saving optimizer and scheduler states to {}".format(output_dir))
        mb.write("Epoch {} done".format(epoch + 1))
        #torch.save(model.discriminator, f'{output_dir}/pretrained-model_{epoch+1}.pt')
        if args.max_steps > 0 and global_step > args.max_steps:
            break

    return global_step, tr_loss / global_step



def evaluate(args, model, eval_dataset, mode, epoch=None):
    results = {}
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    if epoch != None:
        logger.info("***** Running evaluation on {} dataset ({} epoch) *****".format(mode, epoch))
    else:
        logger.info("***** Running evaluation on {} dataset *****".format(mode))
    logger.info("  Num examples = {}".format(len(eval_dataset)))
    logger.info("  Eval Batch size = {}".format(args.eval_batch_size))
    eval_loss = 0.0
    nb_eval_steps = 0
    preds_class = None
    preds_spans=None
    out_label_ids = None
    out_label_span = None

    for batch in progress_bar(eval_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "class_label": batch[3],
                "span_label": batch[5],
                "p_mask": batch[4],
            }
            if args.model_type not in ["distilkobert", "xlm-roberta"]:
                inputs["token_type_ids"] = batch[2]  # Distilkobert, XLM-Roberta don't use segment_ids
            outputs = model(**inputs)
            tmp_eval_loss, class_logits,span_logits = outputs.loss,outputs.class_logits,outputs.span_logits

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds_class is None:
            preds_class = class_logits.detach().cpu().numpy()
            out_label_ids = inputs["class_label"].detach().cpu().numpy()
            preds_spans=span_logits.detach().cpu().numpy()
            out_label_span = inputs["span_label"].detach().cpu().numpy()
        else:
            preds_class = np.append(preds_class, class_logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["class_label"].detach().cpu().numpy(), axis=0)
            preds_spans = np.append(preds_spans, span_logits.detach().cpu().numpy(), axis=0)
            out_label_span = np.append(out_label_span, inputs["span_label"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds_class, axis=1)

    result = acc_score(out_label_ids, preds)
    results.update(result)

    output_dir = os.path.join(args.output_dir, mode)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_eval_file = os.path.join(output_dir, "{}-{}.txt".format(mode, epoch) if epoch else "{}.txt".format(mode))
    with open(output_eval_file, "w") as f_w:
        logger.info("***** Eval results on {} dataset *****".format(mode))
        for key in sorted(results.keys()):
            logger.info("  {} = {}".format(key, str(results[key])))
            f_w.write("  {} = {}\n".format(key, str(results[key])))

    return results

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

def main(cli_args):
    # Read from config file and make args
    with open(os.path.join(cli_args.config_dir, cli_args.config_file)) as f:
        args = AttrDict(json.load(f))
    logger.info("Training/evaluation parameters {}".format(args))
    args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    args.output_dir = os.path.join(args.ckpt_dir, args.output_dir)

    init_logger()
    set_seed(args)
    def tie_weights(generator, discriminator):
        generator.electra.embeddings.word_embeddings = discriminator.electra.embeddings.word_embeddings
        generator.electra.embeddings.position_embeddings = discriminator.electra.embeddings.position_embeddings
        generator.electra.embeddings.token_type_embeddings = discriminator.electra.embeddings.token_type_embeddings


    processor = MultiProcessor(args)
    generator = AutoModelForMaskedLM.from_pretrained("google/electra-small-generator")
    discriminator = ElectraForPreTraining.from_pretrained("google/electra-base-discriminator")
    #tie_weights(generator, discriminator)
    tokenizer = AutoTokenizer.from_pretrained("google/electra-small-generator")
    model = Electra(
        generator,
        discriminator,
        num_tokens=30522,
        mask_token_id=103,
        pad_token_id=0,
        mask_prob=0,
        mask_ignore_token_ids=[tokenizer.vocab['[CLS]'], tokenizer.vocab['[SEP]']],
        random_token_prob=0.0).to(args.device)


    model.to(args.device)

    # Load dataset
    train_dataset = load_and_cache_examples(args=args, tokenizer=tokenizer, mode="train") if args.train_file else None
    dev_dataset = load_and_cache_examples(args, tokenizer=tokenizer, mode="dev") if args.dev_file else None
    test_dataset = load_and_cache_examples(args, tokenizer=tokenizer, mode="test") if args.test_file else None

    if dev_dataset == None:
        args.evaluate_test_during_training = False  # If there is no dev dataset, only use testset

    if args.do_train:
        global_step, tr_loss = train(args, model, train_dataset, dev_dataset, test_dataset)
        logger.info(" global_step = {}, average loss = {}".format(global_step, tr_loss))




if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()

    cli_parser.add_argument("--config_dir", type=str, default="config")
    cli_parser.add_argument("--config_file", type=str, required=True)

    cli_args = cli_parser.parse_args()

    main(cli_args)