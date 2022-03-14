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
        """

        print('pretrained model: ', args.T5.pretrained_model)
        print('from checkpoint: ', args.T5.from_checkpoint)
        self.args = args
        self.model_name = args.T5.model_name
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.model = PlotGenerationModel(args.T5.pretrained_model, self.model_name)
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        training_args_dict = args.T5.train_args if not args.T5.from_checkpoint else args.T5.train_args_checkpoint
        output_dir = f'results__{kw_type}' if not args.T5.from_checkpoint else f'{kw_type}_from_checkpoint'
        self.training_args = TrainingArguments(output_dir=output_dir, **training_args_dict)
        self.input_cols = self.args.T5.input_cols+[kw_type]
        self.organize_dataset(self.input_cols)
        self.repetitions = repetitions(self.tokenizer)
        eval_data =  self.tokenized_datasets['validation'] if not args.T5.from_checkpoint else self.tokenized_datasets['test']
        self.trainer = Trainer(
        model=self.model, args=self.training_args,
        train_dataset = self.tokenized_datasets['train'],
        eval_dataset = eval_data,
        compute_metrics = self.repetitions.eval,
        # data_collator = self.collate_fn
        )

    def change_model_beams(self, num):
        self.model.update_num_beams(num)


    def organize_dataset(self, input_cols):
        """
        :param input_cols: data columns in the dataframe
        :return: saves original dataframe and tokenized datasets for the model
        """
        self.df = pd.read_csv(self.args.data_paths[self.args.T5.run_ds])
        train_ds = self.df[self.df['row_class']=='train'][input_cols+['clean_Plot']]
        val_ds = self.df[self.df['row_class']=='val'][input_cols+['clean_Plot']]
        if self.args.T5.from_checkpoint:
            train_ds=train_ds[0:5]
            val_ds = val_ds[:5]
        self.test_ds = self.df[self.df['row_class']=='test'][input_cols+['clean_Plot']]
        train_ds = Dataset.from_pandas(train_ds)
        test_ds = Dataset.from_pandas(self.test_ds)
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

    def __init__(self, model_path, model_name,config=None, num_beams=10):
        super(PlotGenerationModel, self).__init__()
        if not config:
            self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        else:
            self.model = T5ForConditionalGeneration.from_pretrained(model_path, config=config)

        # self.model: AutoModelWithLMHead
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.num_beams = num_beams

    def update_num_beams(self, num_beams):
        self.num_beams = num_beams

    def forward(self, input_ids, attention_mask, labels=None):
        if self.model.training: ##TODO check
        # if labels is not None:
            labels = labels.squeeze(1)
            ans = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels,return_dict=True)
            return ans
        else:
            gen_pred = self.model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                           return_dict_in_generate=True, max_length=300,num_beams=self.num_beams,
                                           no_repeat_ngram_size=3,num_return_sequences=1)
            gen_pred['loss'] = torch.zeros(0).to(self.model.device)
            return gen_pred

    def generate_plot(self, txt):
        self.model.eval()
        # device=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        with torch.no_grad():
            txt = self.tokenizer(txt, return_tensors="pt")
            txt= {k:v.to(self.model.device) for k,v in txt.items()}
            beam_outputs = self.model.generate(
                **txt,
                max_length=300,
                num_beams=self.num_beams,
                no_repeat_ngram_size=3,
                num_return_sequences=1
            )
            res = self.tokenizer.decode(beam_outputs[0], skip_special_tokens=True)
            res = res.replace('title2008-10-08workNY Times','').replace('Retrieved 2010-10-08workNY Times','')
            return res

class repetitions():
    """
    This class evaluates model results using inter and intre repetitions measures, based on the code is adjusted from the plan and write paper code
    that cited in our paper.
    """
    def __init__ (self, tokenizer):
        self.tokenizer = tokenizer
        self.order='sorted'

    def get_ngrams(self,text, n=3, sent_delimiter="."):
        """takes file with text and an optional sentence delimiter, returns counter of ngrams"""
        ngrams = Counter()
        sentences = [sent.split() for sent in text.strip().split(sent_delimiter)]  # nested list words in sents
        for sent in sentences:
            for i in range(len(sent) - n + 1):
                cur_ngram = sent[i:i + n]
                if self.order=='sorted':
                    cur_ngram = sorted(cur_ngram)
                ngrams[' '.join(cur_ngram)] += 1
        return ngrams

    def intra_repetitions(self,n,plots): ## per plot
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


    def inter_repetitions(self,plots): ##between plots
        all_plots = ''
        for plot in plots:
            all_plots += plot + '.'
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
        output = np.maximum(output.predictions,0)
        self.plots = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        lower_plots = [plot.lower() for plot in self.plots]
        results = {}
        for order in ['sorted','original']:
            self.order=order
            print(self.order)
            res = self.inter_repetitions(lower_plots)
            for n in [1,2,3,4]:
                if n==1 and order=='original':
                    continue
                tmp = self.intra_repetitions(n, lower_plots)
                res[f'intra_{n}_mean'] = tmp['mean']
                res[f'intra_{n}_min'] = tmp['min']
                res[f'intra_{n}_max'] = tmp['max']
            for k,v in res.items():
                results[f'{k}_{self.order}'] = v
        print(results)
        return results