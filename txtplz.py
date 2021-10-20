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


parser = argparse.ArgumentParser(description='Generate text.')
parser.add_argument('model', metavar='MODEL', type=str,
                    help='model name')
parser.add_argument('--batch-size',
                    type=int, default=1,
                    help='batch size')
args = parser.parse_args()


model = args.model


if model == 'polish.gpt2.medium':
    model_dir = download_polish_gpt2_model('medium')
elif model == 'polish.gpt2.large':
    model_dir = download_polish_gpt2_model('large')
else:
    print(f'Unknown model {model}', file=sys.stderr)
    exit(1)


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
model.to(device)
model.eval()

for line in sys.stdin:
    line = line.rstrip('\n')
    result = model.sample(
        [line],
        beam=5, sampling=True, sampling_topk=50, sampling_topp=0.95,
        temperature=0.5, max_len_a=2, max_len_b=300, no_repeat_ngram_size=3)
    print(result[0])
