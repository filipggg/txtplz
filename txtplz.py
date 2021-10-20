#!/usr/bin/env python3

import os
from fairseq import hub_utils
from fairseq.models.transformer_lm import TransformerLanguageModel
import sys
import torch
import requests
from pathlib import Path
import py7zr
import argparse
import itertools as it
import re
from transformers import AutoTokenizer, AutoModelWithLMHead

alt_names = {
    'gpt2.small': 'gpt2',
    'gpt2.medium': 'gpt2-medium',
    'gpt2.large': 'gpt2-large',
    'gpt2.xl': 'gpt2-xl'
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


def download_model(model_name, url, file_list):
    model_dir_path = f'{model_cache}/{model_name}'
    Path(model_dir_path).mkdir(parents=True, exist_ok=True)
    if not check_files(model_dir_path, file_list):
        print('Downloading model...', file=sys.stderr)
        downloadable_file = f'{model_dir_path}/downloadable'
        downloadable = requests.get(url)
        open(downloadable_file, 'wb').write(downloadable.content)
        with py7zr.SevenZipFile(downloadable_file, mode='r') as z:
            z.extractall(path=model_dir_path)
        os.remove(downloadable_file)
    return model_dir_path


def download_polish_gpt2_model(model_size):
    return download_model(
        f'polish-gpt2-{model_size}',
        f'https://github.com/sdadas/polish-nlp-resources/releases/download/gpt-2/gpt2_{model_size}_fairseq.7z',
        ['model.pt'])


def grouper(n, iterable):
    iterable = iter(iterable)
    return iter(lambda: list(it.islice(iterable, n)), [])


def unquote(t):
    if t is None:
        return t

    return t.replace('\\n', '\n').replace('\\r', '\r')


def prepare_input(line, opts):
    inp = line.rstrip('\n')

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


class PolishGPT2:
    def __init__(self, device, variant):
        self.model = get_polish_gpt2(variant)
        self.model.to(device)
        self.model.eval()

    def run(self, line_batch):
        results = self.model.sample(
            line_batch,
            beam=5, sampling=True, sampling_topk=50, sampling_topp=0.95,
            temperature=0.5, max_len_a=2, max_len_b=300, no_repeat_ngram_size=3)
        return results


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
        results = self.model.generate(t, beam=5, do_sample=True, sampling_topk=50, sampling_topp=0.95,
                                      max_length=100,
                                      temperature=0.5, max_len_a=2, max_len_b=300, no_repeat_ngram_size=3)
        return (self._detok(result) for result in results)


def get_model(device, opts, model_name):
    normalized_model_name = normalize_model_name(model_name)

    if m := re.search(r'^polish\.gpt2\.(.*)$', normalized_model_name):
        return PolishGPT2(device, m.group(1))
    elif normalized_model_name in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']:
        return GPT2(device, opts, normalized_model_name)
    else:
        print(f'Unknown model {model_name}', file=sys.stderr)
        exit(1)


parser = argparse.ArgumentParser(description='Generate text.')
parser.add_argument('model', metavar='MODEL', type=str,
                    help='model name')
parser.add_argument('--batch-size',
                    type=int, default=1,
                    help='batch size')
parser.add_argument('--prompt',
                    type=str, default=None,
                    help='prompt')
parser.add_argument('--mark-tokens', help='mark tokens', action='store_true')
opts = parser.parse_args()

opts.prompt = unquote(opts.prompt)

model_name = opts.model

model = get_model(device, opts, model_name)

for line_batch in grouper(opts.batch_size, (prepare_input(line, opts) for line in sys.stdin)):
    results = model.run(line_batch)
    for output_line in results:
        print(output_line)
