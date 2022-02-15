from datasets import load_dataset

# model_name = 't5-small'

from transformers import AutoTokenizer, AutoModelWithLMHead , TrainingArguments, Trainer, T5Tokenizer
from torch import nn
import torch
from pathlib import Path
import pandas as pd

from datasets import Dataset, DatasetDict

# datasets.Dataset.from_pandas(tmp)
# tokenized_datasets = datasets.map(tokenize_fn, remove_columns=datasets["train"].column_names)
# fn_kwargs = {'input_cols':input_cols}
# tokenized_datasets.set_format('torch')


class T5_trainer():

    # TODO: add functions: repetition metrics,
    def __init__(self, args):
        self.args = args
        self.model_name = args.T5.model_name
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.model = PlotGenerationModel(self.model_name)
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.training_args = TrainingArguments(**args.T5.train_args)
        self.tranier =  Trainer(
        model=self.model, args=self.training_args,
        train_dataset = self.tokenized_datasets['train'],
        eval_dataset = self.tokenized_datasets['validation'],
        # compute_metrics=metric_fn,
        data_collator=self.collate_fn
    )


    def organize_dataset(self, input_cols):
        self.df = pd.read_csv(self.args.data_paths.full_dataset)
        train_ds = self.df[self.df['row_class']=='train'][input_cols]
        test_ds = self.df[self.df['row_class']=='test'][input_cols]
        val_ds = self.df[self.df['row_class']=='validation'][input_cols]
        train_ds = Dataset.from_pandas(train_ds)
        test_ds = Dataset.from_pandas(test_ds)
        val_ds = Dataset.from_pandas(val_ds)
        ds = DatasetDict({'train': train_ds, 'test': test_ds, 'validation': val_ds})
        self.tokenized_datasets = ds.map(self.tokenize_fn, fn_kwargs={'input_cols': input_cols},
                                          remove_columns=ds["train"].column_names)
        self.tokenized_datasets.set_format('torch')

    def tokenize_fn(self, examples, input_cols):
        tokenized_examples = self.tokenizer(
            *[examples[col] for col in input_cols], truncation=True, padding="max_length",
        )
        plot = examples['Plot']
        tok_plot = self.tokenizer(
            plot, truncation=True, padding="max_length"
        )['input_ids']
        tokenized_examples['labels'] = tok_plot
        return tokenized_examples

    def collate_fn(self, data):
        out = {}
        num_labels = []
        for sen in data:
            labels = sen['labels']
            num_labels.append(len(labels))
        num_labels = max(num_labels)
        for sen in data:
            labels = torch.stack(sen['labels'])
            add_labels = num_labels - len(labels)
            add_labels = torch.zeros((add_labels, labels.shape[1]), device=labels.device, dtype=labels.dtype)
            labels = torch.cat([labels, add_labels])
            sen['labels'] = labels
        for k in data[0]:
            out[k] = torch.stack([f[k] for f in data])
        return out



class PlotGenerationModel(nn.Module):

    def __init__(self, model_name):
        super(PlotGenerationModel, self).__init__()
        self.model = AutoModelWithLMHead.from_pretrained(model_name)
        self.model: AutoModelWithLMHead
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask, labels):
        if self.model.train(): ##TODO check
            labels = labels.squeeze(1)
            ans = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            return ans
        else:
            gen_pred = self.model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                           return_dict_in_generate=True)
            gen_pred['loss'] = torch.zeros(0).to(self.model.device)
            return gen_pred



