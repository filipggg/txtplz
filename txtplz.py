#!/usr/bin/env python3

import os
from fairseq import hub_utils
from fairseq.models.transformer_lm import TransformerLanguageModel
import sys
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# jeśli nie działa na karcie:
# device = 'cpu'

model_dir = "."
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
model = hub_utils.GeneratorHubInterface(loaded["args"], loaded["task"], loaded["models"])
model.to(device)
model.eval()

for line in sys.stdin:
    line = line.rstrip('\n')
    result = model.sample(
        [line],
        beam=5, sampling=True, sampling_topk=50, sampling_topp=0.95,
        temperature=0.5, max_len_a=2, max_len_b=300, no_repeat_ngram_size=3)
    print(result[0])
