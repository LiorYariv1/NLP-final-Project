from datasets import load_dataset

# model_name = 't5-small'

from transformers import T5ForConditionalGeneration, AutoModelWithLMHead , TrainingArguments, Trainer, T5Tokenizer
from torch import nn
import torch
from pathlib import Path
import pandas as pd
from collections import Counter
from numpy import mean
from datasets import Dataset, DatasetDict
import numpy as np

# datasets.Dataset.from_pandas(tmp)
# tokenized_datasets = datasets.map(tokenize_fn, remove_columns=datasets["train"].column_names)
# fn_kwargs = {'input_cols':input_cols}
# tokenized_datasets.set_format('torch')


class T5_trainer():

    # TODO: add functions: repetition metrics,
    def __init__(self, args, kw_type='kw_Rake_1'):
        """
        this class is used for the T5 training process, including tokenization, training and evaluation
        :param args: input arguments
        """
        self.args = args
        self.model_name = args.T5.model_name
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.model = PlotGenerationModel(self.model_name, self.model_name)
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.training_args = TrainingArguments(output_dir=f'results__{kw_type}', **args.T5.train_args)
        self.input_cols = self.args.T5.input_cols+[kw_type]
        self.organize_dataset(self.input_cols)
        self.repetitions = repetitions(self.tokenizer)
        self.trainer = Trainer(
        model=self.model, args=self.training_args,
        train_dataset = self.tokenized_datasets['train'],
        eval_dataset = self.tokenized_datasets['validation'],
        compute_metrics = self.repetitions.eval,
        # data_collator = self.collate_fn
        )


    def organize_dataset(self, input_cols):
        """
        :param input_cols: data columns in the dataframe
        :return: saves original dataframe and tokenized datasets for the model
        """
        self.df = pd.read_csv(self.args.data_paths[self.args.T5.run_ds])
        train_ds = self.df[self.df['row_class']=='train'][input_cols+['clean_Plot']]
        test_ds = self.df[self.df['row_class']=='test'][input_cols+['clean_Plot']]
        val_ds = self.df[self.df['row_class']=='val'][input_cols+['clean_Plot']]
        train_ds = Dataset.from_pandas(train_ds)
        test_ds = Dataset.from_pandas(test_ds)
        val_ds = Dataset.from_pandas(val_ds)
        ds = DatasetDict({'train': train_ds, 'test': test_ds, 'validation': val_ds})
        self.tokenized_datasets = ds.map(self.tokenize_fn, fn_kwargs={'input_cols': input_cols},
                                          remove_columns=ds["train"].column_names)
        self.tokenized_datasets.set_format('torch')

    def tokenize_fn(self, examples, input_cols):
        """
        :param examples: examples to tokenize
        :param input_cols: data columns
        :return:
        """
        tokenized_examples = \
        self.tokenizer(
        ' '.join([f'<extra_id_{i}> ' + examples[col] for i, col in
                              enumerate(input_cols)]), truncation=True
        )
        plot = examples['clean_Plot']
        tok_plot = self.tokenizer(
            plot, truncation=True, padding='max_length'
        )['input_ids']
        tokenized_examples['labels'] = tok_plot
        return tokenized_examples

    def collate_fn(self, data):
        """
        :param data:
        :return:
        """
        out = {}
        # num_labels = []
        num_input_ids = []
        for sen in data:
            # labels = sen['labels']
            # num_labels.append(len(labels))
            num_input_ids.append(len(sen['input_ids']))
        # max_num = max(max(num_labels),max(num_input_ids))
        num_input_ids = max(num_input_ids)
        # num_input_ids = max(num_input_ids)
        for sen in data:
            # labels = sen['labels']
            # add_labels = max_num - len(labels)
            # add_labels = -100*torch.ones(add_labels, device=labels.device, dtype=labels.dtype)
            # labels = torch.cat([labels, add_labels])
            # sen['labels'] = labels
            input_ids = sen['input_ids']
            attention_mask = sen['attention_mask']
            add_input_ids = num_input_ids - len(input_ids)
            add_input_ids = torch.zeros(add_input_ids, device=input_ids.device, dtype=input_ids.dtype)
            input_ids = torch.cat([input_ids, add_input_ids])
            attention_mask = torch.cat([attention_mask, add_input_ids])
            sen['input_ids'] = input_ids
            sen['attention_mask'] = attention_mask
        for k in data[0]:
            out[k] = torch.stack([f[k] for f in data])
        return out



class PlotGenerationModel(nn.Module):

    def __init__(self, model_path, model_name):
        super(PlotGenerationModel, self).__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        # self.model: AutoModelWithLMHead
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask, labels=None):
        if self.model.training: ##TODO check
        # if labels is not None:
            labels = labels.squeeze(1)
            ans = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels,return_dict=True)
            return ans
        else:
            gen_pred = self.model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                           return_dict_in_generate=True)
            gen_pred['loss'] = torch.zeros(0).to(self.model.device)
            return gen_pred



class repetitions():
    ##TODO: adjust to T5 output
    def __init__ (self, tokenizer=None):
        self.sent_delimiter = '</s>'
        self.tokenizer = tokenizer


    def get_ngrams(self,text, n=3, sent_delimiter="</s>"):
        """takes file with text and an optional sentence delimiter, returns counter of ngrams"""
        ngrams = Counter()
        sentences = [sent.split() for sent in text.strip().split(sent_delimiter)]  # nested list words in sents
        for sent in sentences:
            for i in range(len(sent) - n + 1):
                ngrams[' '.join(sent[i:i + n])] += 1
        return ngrams

    def intra_repetitions(self,n,plots):
        repetition_array = []
        for plot in plots:
            ngrams = self.get_ngrams(plot,n)
            unique_ngrams = len(ngrams)
            total_ngrams = sum(ngrams.values())
            if total_ngrams==0:
                repetition_array.append(0)
                continue
            ngrams_repetition = (total_ngrams - unique_ngrams) / total_ngrams
            repetition_array.append(ngrams_repetition)
        return {'plots_number':len(plots),'mean':mean(repetition_array), 'min':min(repetition_array),'max':max(repetition_array)}

    def inter_repetitions(self,plots):
        all_plots = ''
        for plot in plots:
            all_plots += plot + ' '
        unigrams = self.get_ngrams(all_plots, n=1)
        bigrams = self.get_ngrams(all_plots, n=2)
        trigrams = self.get_ngrams(all_plots, n=3)
        results = {}
        for n,res in zip([1,2,3],[unigrams,bigrams,trigrams]):
            num = len(res)
            cur_sum = sum([res[x] for x in res])
            results[f'inter_{n}'] = 1.0-(float(num)/float(cur_sum))
        return results

    def eval(self, output):
        # loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        # loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        output = np.maximum(output.predictions,0)
        plots = self.tokenizer.batch_decode(output)
        res = self.inter_repetitions(plots)
        for n in [1,2,3]:
            tmp = self.intra_repetitions(n, plots)
            res[f'intra_{n}_mean'] = tmp['mean']
            res[f'intra_{n}_min'] = tmp['min']
            res[f'intra_{n}_max'] = tmp['max']
        return res

