#!/usr/bin/env python3

import os
from fairseq import hub_utils
from fairseq.models.transformer_lm import TransformerLanguageModel
from fairseq.models.roberta import RobertaModel, RobertaHubInterface
import sys
import torch
import requests
from pathlib import Path
import py7zr
import argparse
import itertools as it
import regex as re
from transformers import T5Tokenizer, T5ForConditionalGeneration
import transformers as trans
from zipfile import ZipFile


trans.logging.set_verbosity_error()
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


alt_names = {
    'gpt2.small': 'gpt2',
    'gpt2.medium': 'gpt2-medium',
    'gpt2.large': 'gpt2-large',
    'gpt2.xl': 'gpt2-xl',

    't5.small': 't5-small',
    't5.base': 't5-base',
    't5.large': 't5-large',

    't5.v1_1.large': 'google/t5-v1_1-base',

    'mt5.small': 'google/mt5-small',
    'mt5.base': 'google/mt5-base',
    'mt5.large': 'google/mt5-large',

    'plt5.small': 'allegro/plt5-small',
    'plt5.base': 'allegro/plt5-base',
    'plt5.large': 'allegro/plt5-large'
}

def normalize_model_name(model_name):
    if model_name in alt_names:
        return alt_names[model_name]

    return model_name



device = 'cuda' if torch.cuda.is_available() else 'cpu'
# jeśli nie działa na karcie:
# device = 'cpu'

model_cache = str(Path.home()) + '/' + '.txtplz'


def check_files(dir_to_look, files):
    for f in files:
        if not os.path.isfile(f'{dir_to_look}/{f}'):
            return False
    return True


def download_model(compressor, model_name, url, file_list):
    model_dir_path = f'{model_cache}/{model_name}'
    Path(model_dir_path).mkdir(parents=True, exist_ok=True)
    if not check_files(model_dir_path, file_list):
        print('Downloading model...', file=sys.stderr)
        downloadable_file = f'{model_dir_path}/downloadable'
        downloadable = requests.get(url)
        open(downloadable_file, 'wb').write(downloadable.content)
        if compressor == 'zip':
            with ZipFile(downloadable_file, 'r') as zip:
                zip.extractall(path=model_dir_path)
        else:
            with py7zr.SevenZipFile(downloadable_file, mode='r') as z:
                z.extractall(path=model_dir_path)
        os.remove(downloadable_file)
        print('... done', file=sys.stderr)
    return model_dir_path


def download_polish_gpt2_model(model_size):
    return download_model(
        '7z',
        f'polish-gpt2-{model_size}',
        f'https://github.com/sdadas/polish-nlp-resources/releases/download/gpt-2/gpt2_{model_size}_fairseq.7z',
        ['model.pt'])


def download_polish_roberta_model(version, model_size):
    return download_model(
        'zip',
        f'polish-roberta{version}-{model_size}',
        f'https://github.com/sdadas/polish-roberta/releases/download/models{version}/roberta_{model_size}_fairseq.zip',
        ['model.pt'])


def grouper(n, iterable):
    iterable = iter(iterable)
    return iter(lambda: list(it.islice(iterable, n)), [])


def unquote(t):
    if t is None:
        return t

    return t.replace('\\n', '\n').replace('\\r', '\r')


def prepare_input(line, opts):
    line = line.rstrip('\n')

    inp_fields = line.split('\t')

    inp = opts.input_pattern.replace('{}', line)

    inp = re.sub(r'\{(\d+)\}', lambda m: inp_fields[int(m.group(1)) - 1], inp)

    if opts.prompt is not None:
        inp = opts.prompt + inp

    return inp


def get_polish_gpt2(variant):
    model_dir = download_polish_gpt2_model(variant)

    loaded = hub_utils.from_pretrained(
        model_name_or_path=model_dir,
        checkpoint_file="model.pt",
        data_name_or_path=model_dir,
        bpe="hf_byte_bpe",
        bpe_merges=os.path.join(model_dir, "merges.txt"),
        bpe_vocab=os.path.join(model_dir, "vocab.json"),
        load_checkpoint_heads=True,
        archive_map=TransformerLanguageModel.hub_models()
    )
    model = hub_utils.GeneratorHubInterface(
        loaded["args"], loaded["task"], loaded["models"])
    return model


def get_polish_roberta(version, model_size):
    model_dir = download_polish_roberta_model(version, model_size)

    loaded = hub_utils.from_pretrained(
        model_name_or_path=model_dir,
        data_name_or_path=model_dir,
        bpe="sentencepiece",
        sentencepiece_vocab=os.path.join(model_dir, "sentencepiece.bpe.model"),
        load_checkpoint_heads=True,
        archive_map=RobertaModel.hub_models()
    )
    model = RobertaHubInterface(loaded['args'], loaded['task'], loaded['models'][0])
    return model


class PolishGPT2:
    def __init__(self, device, variant):
        self.model = get_polish_gpt2(variant)
        self.model.to(device)
        self.model.eval()

    def run(self, line_batch):
        results = self.model.sample(
            line_batch,
            beam=5, sampling=(not opts.no_sampling), sampling_topk=opts.topk, sampling_topp=opts.topp,
            temperature=opts.temperature, max_len_a=2, max_len_b=300, no_repeat_ngram_size=3)
        return results


class PolishRoberta:
    def __init__(self, device, version, model_size):
        self.model = get_polish_roberta(version, model_size)
        self.model.to(device)
        self.model.eval()

    def run(self, line_batch):
        return [self.run_for_line(line) for line in line_batch]

    def run_for_line(self, line):
        result = (self.model.fill_mask(
            line.replace('<>', '<mask>'),
            topk=1))[0][0]
        return result


class GPT2:
    def __init__(self, device, opts, variant):
        self.model = torch.hub.load('huggingface/transformers', 'modelForCausalLM', variant).eval().to(device)
        self.device = device
        self.tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'gpt2')
        self.pad_token_id = 50256
        self.opts = opts

    def _detok(self, ids):
        if self.opts.mark_tokens:
            return '|'.join(self.tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=True))
        else:
            return self.tokenizer.decode(ids, skip_special_tokens=True)

    def run(self, line_batch):
        tokens = [self.tokenizer(line, add_special_tokens=True).input_ids for line in line_batch]
        padded_tokens = list(zip(*it.zip_longest(*tokens, fillvalue=self.pad_token_id)))

        t = torch.tensor(padded_tokens).to(self.device)
        results = self.model.generate(t, beam=5, do_sample=(not opts.no_sampling), sampling_topk=opts.topk, sampling_topp=opts.topp,
                                      max_length=100,
                                      temperature=opts.temperature, max_len_a=2, max_len_b=300, no_repeat_ngram_size=3)
        return (self._detok(result) for result in results)


class T5:
    def __init__(self, device, opts, variant):
        self.tokenizer = T5Tokenizer.from_pretrained(variant)
        self.model = T5ForConditionalGeneration.from_pretrained(variant).eval().to(device)

        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def _preprocess_batch(self, line_batch):
        return [self._preprocess_line(line) for line in line_batch]

    def _preprocess_line(self, line):
        counter = 0
        def inc(m, i=[-1]):
            i[0] += 1
            return f'<extra_id_{str(i[0])}>'
        return re.sub(r'<>', inc, line)
        
    def run(self, line_batch):
        inputs = self.tokenizer(self._preprocess_batch(line_batch), return_tensors="pt", padding=True).to(device)

        output_sequences = self.model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            num_beams=1,
#            top_k=opts.topk,
#            top_p=opts.topp,
#            temperature=opts.temperature,
#            no_repeat_ngram_size=3,
            do_sample=(not opts.no_sampling))

        return self.tokenizer.batch_decode(output_sequences, skip_special_tokens=False)


def get_model(device, opts, model_name):
    normalized_model_name = normalize_model_name(model_name)

    if m := re.search(r'^polish\.gpt2\.(.*)$', normalized_model_name):
        return PolishGPT2(device, m.group(1))
    elif normalized_model_name in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']:
        return GPT2(device, opts, normalized_model_name)
    elif normalized_model_name in ['t5-small', 't5-base', 't5-large', 'google/t5-v1_1-base', 'google/mt5-small', 'google/mt5-base', 'google/mt5-large', 'allegro/plt5-small', 'allegro/plt5-base', 'allegro/plt5-large']:
        return T5(device, opts, normalized_model_name)
    elif normalized_model_name == 'polish.roberta.large':
        return PolishRoberta(device, '', 'large')
    elif normalized_model_name == 'polish.roberta.base':
        return PolishRoberta(device, '', 'base')
    elif normalized_model_name == 'polish.roberta.v2.base':
        return PolishRoberta(device, '-v2', 'base')
    else:
        print(f'Unknown model {model_name}', file=sys.stderr)
        exit(1)


def remove_prefix(prefix, s):
    prefix_len = len(prefix)
    if s[0:prefix_len] == prefix:
        return s[prefix_len:]

    return s


def process_output(opts, inp, out):
    final_out = out

    removed_all_input = False
    if opts.remove_input:
        s = remove_prefix(inp, final_out)
        if s != final_out:
            removed_all_input = True
            final_out = s

    if not removed_all_input and opts.prompt:
        final_out = remove_prefix(opts.prompt, final_out)

    if opts.delimiter is not None:
        m = re.search(opts.delimiter, final_out)
        if m:
            final_out = final_out[0:m.start()]

    if opts.search is not None:
        m = re.search(opts.search, final_out)
        if m:
            if m.groups():
                final_out = '\t'.join(m.groups())
            else:
                final_out = m.group(0)

    return final_out


parser = argparse.ArgumentParser(description='Generate text.')
parser.add_argument('model', metavar='MODEL', type=str,
                    help='model name')
parser.add_argument('--batch-size',
                    type=int, default=1,
                    help='batch size')
parser.add_argument('--prompt',
                    type=str, default=None,
                    help='prompt')
parser.add_argument('--topk',
                    type=int, default=50,
                    help='topk')
parser.add_argument('--topp',
                    type=float, default=0.95,
                    help='topp')
parser.add_argument('--temperature',
                    type=float, default=1.0,
                    help='temperature')
parser.add_argument('--mark-tokens', help='mark tokens', action='store_true')
parser.add_argument('--remove-input', help='remove input', action='store_true')
parser.add_argument('--no-sampling', help='switch off sampliong', action='store_true')
parser.add_argument('--delimiter', help='end delimiter', type=str, default=None)
parser.add_argument('--input-pattern', help='end delimiter', type=str, default='{}')
parser.add_argument('--search', help='pattern',type=str,default=None)
opts = parser.parse_args()

if opts.no_sampling:
    opts.topk = -1
    opts.topp = -1.0

opts.prompt = unquote(opts.prompt)
opts.delimiter = unquote(opts.delimiter)
opts.input_pattern = unquote(opts.input_pattern)

model_name = opts.model

model = get_model(device, opts, model_name)

for line_batch in grouper(opts.batch_size, (prepare_input(line, opts) for line in sys.stdin)):
    results = model.run(line_batch)
    for input_line, output_line in zip(line_batch, results):
        print(process_output(opts, input_line, output_line))
