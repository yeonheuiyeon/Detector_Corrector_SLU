
import argparse
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import os

# Importing the T5 modules from huggingface/transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer

from rich.table import Column, Table
from rich import box
from rich.console import Console
import random
import pandas as pd


def display_df(df):
  """display dataframe in ASCII format"""

  console=Console()
  table = Table(Column("source_text", justify="center" ), Column("target_text", justify="center"), title="Sample Data",pad_edge=False, box=box.ASCII)

  for i, row in enumerate(df.values.tolist()):
    table.add_row(row[0], row[1])

  console.print(table)


class YourDataSetClass(Dataset):
  """
  Creating a custom dataset for reading the dataset and
  loading it into the dataloader to pass it to the neural network for finetuning the model

  """

  def __init__(self, dataframe, tokenizer, source_len, target_len, source_text, target_text):
    self.tokenizer = tokenizer
    self.data = dataframe
    self.source_len = source_len
    self.target_len = target_len
    self.target_text = self.data[target_text]
    self.source_text = self.data[source_text]

  def __len__(self):
    return len(self.target_text)

  def __getitem__(self, index):
    source_text = str(self.source_text[index])
    target_text = str(self.target_text[index])

    #cleaning data so as to ensure data is in string type
    source_text = ' '.join(source_text.split())
    target_text = ' '.join(target_text.split())

    source = self.tokenizer.batch_encode_plus([source_text], max_length= self.source_len, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors='pt')
    target = self.tokenizer.batch_encode_plus([target_text], max_length= self.target_len, pad_to_max_length=True, truncation=True, padding="max_length", return_tensors='pt')

    source_ids = source['input_ids'].squeeze()
    source_mask = source['attention_mask'].squeeze()
    target_ids = target['input_ids'].squeeze()
    target_mask = target['attention_mask'].squeeze()

    return {
        'source_ids': source_ids.to(dtype=torch.long),
        'source_mask': source_mask.to(dtype=torch.long),
        'target_ids': target_ids.to(dtype=torch.long),
        'target_ids_y': target_ids.to(dtype=torch.long)
    }


def train(epoch, tokenizer, model, device, loader, optimizer):

  """
  Function to be called for training with the parameters passed from main function

  """

  model.train()
  for _,data in enumerate(loader, 0):
    y = data['target_ids'].to(device, dtype = torch.long)
    y_ids = y[:, :-1].contiguous()
    #print(' '.join([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in y_ids[0, :]]))
    lm_labels = y[:, 1:].clone().detach()
    lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
    ids = data['source_ids'].to(device, dtype = torch.long)
    #print(' '.join([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in ids[0, :]]))
    mask = data['source_mask'].to(device, dtype = torch.long)

    outputs = model(input_ids = ids, attention_mask = mask, decoder_input_ids=y_ids, labels=lm_labels)
    loss = outputs[0]

    if _%100==0:
      training_logger.add_row(str(epoch), str(_), str(loss))
      console.print(training_logger)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def validate(tokenizer, model, device, loader, trg_max_len):

  """
  Function to evaluate model for predictions

  """
  model.eval()
  predictions = []
  actuals = []
  with torch.no_grad():
      uu=0
      for _, data in enumerate(loader, 0):
          y = data['target_ids'].to(device, dtype = torch.long)
          ids = data['source_ids'].to(device, dtype = torch.long)
          mask = data['source_mask'].to(device, dtype = torch.long)
          total_start_time = time.time()
          generated_ids = model.generate(
              input_ids = ids,
              attention_mask = mask,
              max_length=trg_max_len,
              num_beams=2,
              length_penalty=1.0,
              early_stopping=True,
              temperature=0.8
              )
          total_end_time = time.time()
          pred=total_end_time - total_start_time
          uu+=pred
          preds = [tokenizer.decode(g, skip_special_tokens=False, clean_up_tokenization_spaces=True) for g in generated_ids]
          target = [tokenizer.decode(t, skip_special_tokens=False, clean_up_tokenization_spaces=True)for t in y]
          if _%10==0:
              console.print(f'Completed {_}')

          predictions.extend(preds)
          actuals.extend(target)
  print(uu)
  return predictions, actuals


def T5Trainer(dataframe, test_dataframe, source_text, target_text, model_params, output_dir="./outputs/" ):

    """
    T5 trainer

    """

    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(model_params["SEED"]) # pytorch random seed
    np.random.seed(model_params["SEED"]) # numpy random seed
    torch.backends.cudnn.deterministic = True

    # logging
    console.log(f"""[Model]: Loading {model_params["MODEL"]}...\n""")

    # tokenzier for encoding the text
    tokenizer = AutoTokenizer.from_pretrained(model_params["MODEL"])

    # Defining the model. We are using t5-base model and added a Language model layer on top for generation of Summary.
    # Further this model is sent to device (GPU/TPU) for using the hardware.
    model = T5ForConditionalGeneration.from_pretrained(model_params["MODEL"])
    model = model.to(device)

    # logging
    console.log(f"[Data]: Reading data...\n")

    # Importing the raw dataset
    dataframe = dataframe[[source_text,target_text]]
    test_dataframe = test_dataframe[[source_text, target_text]]
    display_df(dataframe.head(2))
    display_df(test_dataframe.head(2))

    train_dataset=dataframe
    val_dataset=test_dataframe

    console.print(f"TRAIN Dataset: {train_dataset.shape}")
    console.print(f"TEST Dataset: {val_dataset.shape}\n")

    # Creating the Training and Validation dataset for further creation of Dataloader
    training_set = YourDataSetClass(train_dataset, tokenizer, model_params["MAX_SOURCE_TEXT_LENGTH"], model_params["MAX_TARGET_TEXT_LENGTH"], source_text, target_text)
    val_set = YourDataSetClass(val_dataset, tokenizer, model_params["MAX_SOURCE_TEXT_LENGTH"], model_params["MAX_TARGET_TEXT_LENGTH"], source_text, target_text)


    # Defining the parameters for creation of dataloaders
    train_params = {
      'batch_size': model_params["TRAIN_BATCH_SIZE"],
      'shuffle': True,
      'num_workers': 0
      }


    val_params = {
      'batch_size': model_params["VALID_BATCH_SIZE"],
      'shuffle': False,
      'num_workers': 0
      }


    # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
    training_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)


    # Defining the optimizer that will be used to tune the weights of the network in the training session.
    optimizer = torch.optim.Adam(params =  model.parameters(), lr=model_params["LEARNING_RATE"])


    # Training loop
    console.log(f'[Initiating Fine Tuning]...\n')

    # for epoch in range(model_params["TRAIN_EPOCHS"]):
    #     train(epoch, tokenizer, model, device, training_loader, optimizer)
    #
    #     # evaluating test dataset
    #     predictions, actuals = validate(tokenizer, model, device, val_loader, model_params["MAX_TARGET_TEXT_LENGTH"])
    #     final_df = pd.DataFrame({'Generated Text': predictions, 'Actual Text': actuals})
    #     final_df.to_csv(os.path.join(output_dir, f'predictions_{epoch}.csv'))
    #
    #     console.log(f"[Saving Model]...\n")
    #     # Saving the model after training
    #     path = os.path.join(output_dir, f"model_files_{epoch}")
    #     model.save_pretrained(path)
    #     tokenizer.save_pretrained(path)

    if model_params["TRAIN_EPOCHS"] ==0:
        # evaluating test dataset
        predictions, actuals = validate(tokenizer, model, device, val_loader, model_params["MAX_TARGET_TEXT_LENGTH"])
        final_df = pd.DataFrame({'Generated Text': predictions, 'Actual Text': actuals})
        final_df.to_csv(os.path.join(output_dir, f'predictions.csv'))


    # evaluating test dataset
    console.log(f"[Initiating Validation]...\n")

    console.save_text(os.path.join(output_dir,'logs.txt'))

    console.log(f"[Validation Completed.]\n")
    console.print(f"""[Model] Model saved @ {os.path.join(output_dir, "model_files")}\n""")
    console.print(f"""[Validation] Generation on Validation data saved @ {os.path.join(output_dir,'predictions.csv')}\n""")
    console.print(f"""[Logs] Logs saved @ {os.path.join(output_dir,'logs.txt')}\n""")


if __name__=='__main__':
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument("--model", type=str, required=True)
    cli_parser.add_argument("--batch", type=int, required=True)
    cli_parser.add_argument("--data_dir", type=str, required=True)
    cli_parser.add_argument("--max_src_len", type=int, required=True)
    cli_parser.add_argument("--max_trg_len", type=int, required=True)

    cli_args=cli_parser.parse_args()

    # define a rich console logger
    console = Console(record=True)

    #load dataset
    df = pd.read_csv(f"{cli_args.data_dir}", sep=',')
    df=df[['src_text', 'target_text']]
    print(df.head(10))

    #load test dataset
    df2 = pd.read_csv(f"{cli_args.data_dir}", sep=',')

    #training logger
    training_logger = Table(Column("Epoch", justify="center"),
                            Column("Steps", justify="center"),
                            Column("Loss", justify="center"),
                            title="Training Status", pad_edge=False, box=box.ASCII)

    #GPU or CPU
    from torch import cuda
    device = 'cuda' if cuda.is_available() else 'cpu'

    #model_params
    model_params = {
        "MODEL": cli_args.model,  # model_type: t5-base/t5-large
        "TRAIN_BATCH_SIZE": cli_args.batch,  # training batch size
        "VALID_BATCH_SIZE": cli_args.batch,  # validation batch size
        "TRAIN_EPOCHS": 0,  # number of training epochs
        "VAL_EPOCHS": 1,  # number of validation epochs
        "LEARNING_RATE": 1e-4,  # learning rate
        "MAX_SOURCE_TEXT_LENGTH": cli_args.max_src_len,  # max length of source text
        "MAX_TARGET_TEXT_LENGTH": cli_args.max_trg_len,  # max length of target text
        "SEED": 42  # set seed for reproducibility
    }

    output_dir=f'{cli_args.model}_{cli_args.data_dir[:-4]}_test_inference'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    T5Trainer(dataframe=df, test_dataframe=df2, source_text="src_text", target_text="target_text", model_params=model_params, output_dir=output_dir)