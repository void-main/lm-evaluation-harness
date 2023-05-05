""" Triton API
Run the evaluation with Triton API
Example usage:
    python main.py --model triton --model_args tokenizer_dir=/path/to/tokenizer --no_cache --tasks piqa
Homepage: https://github.com/triton-inference-server/server
"""
import logging
import os
import requests as _requests
import time
from tqdm import tqdm
from lm_eval.base import BaseLM

import numpy as np

import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype

import torch
from transformers import AutoTokenizer
from transformers import LlamaTokenizer

logger = logging.getLogger(__name__)

TRITON_NAME_TO_DTYPE = {
    "input_ids": "uint32",
    "input_lengths": "uint32",
    "request_output_len": "uint32",
    "stop_words_list": "int32",
    "bad_words_list": "int32",
    "beam_search_diversity_rate": "float32",
    "temperature": "float32",
    "len_penalty": "float32",
    "repetition_penalty": "float32",
    "random_seed": "uint64",
    "is_return_log_probs": "bool",
    "beam_width": "uint32",
    "runtime_top_k": "uint32",
    "runtime_top_p": "float32",
    "start_id": "uint32",
    "end_id": "uint32",
}

TRITON_NAME_DEFAULT_VALS = {
    "beam_search_diversity_rate": [0.0],
    "temperature": [1.0],
    "len_penalty": [1.0],
    "repetition_penalty": [1.0],
    "random_seed_": [0],
    "is_return_log_probs": [False],
    "beam_width": [1],
    "runtime_top_k": [1],
    "runtime_top_p": [0.0],
}

def prepare_tensor(client, name, input):
    t = client.InferInput(name, input.shape, np_to_triton_dtype(input.dtype))
    t.set_data_from_numpy(input)
    return t

def triton_completion(**kwargs):
    """Query Triton API for completion.
    Retry with back-off until they respond.
    """
    cl = kwargs['client']
    model_name = kwargs['model_name']
    requests = kwargs['requests']
    batched_request = {}

    for req in requests:
        for key, dtype in TRITON_NAME_TO_DTYPE.items():
            if key not in req and key not in TRITON_NAME_DEFAULT_VALS:
                continue

            if key not in batched_request:
                batched_request[key] = []

            val = req[key] if key in req else TRITON_NAME_DEFAULT_VALS[key]
            batched_request[key].append(val)

    for key, dtype in TRITON_NAME_TO_DTYPE.items():
        if key not in batched_request:
            continue

        val = batched_request[key]
        val = np.array(val, dtype=dtype)
        batched_request[key] = val

    payload = [prepare_tensor(httpclient, key, batched_request[key])
            for key in batched_request.keys()]
    backoff_time = 3
    while True:
        try:
            return cl.infer(model_name, payload)
        except _requests.exceptions.RequestException:
            import traceback

            traceback.print_exc()
            time.sleep(backoff_time)
            backoff_time *= 1.5


class TritonLM(BaseLM):
    def __init__(self, tokenizer_dir):
        """
        :param engine: str
            Triton API engine (e.g. `ensemble`, `fastertransformer`)
        :param truncate: bool
            Truncate input if too long (if False and input is too long, throw error)
        """
        super().__init__()

        self.client = httpclient.InferenceServerClient('localhost:8000', concurrency=1)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        self.vocab_size = self.tokenizer.vocab_size

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        # NOTE: Turn on truncation to avoid errors on long inputs.
        return 2048

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return 16

    @property
    def device(self):
        return 'cpu'

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call
        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        assert inps.dim() == 2, 'input shape should be [batch, seqlen]'
        requests = [{
            "input_ids": inp,
            "input_lengths": [inps.size(dim=1)],
            "request_output_len": [1],
            "is_return_log_probs": [True],
        } for inp in inps.tolist()]

        result = triton_completion(
            client=self.client,
            model_name='fastertransformer',
            requests=requests,
        )

        out_tensor = torch.from_numpy(result.as_numpy('logits'))
        return out_tensor

    def _model_generate(self, context, max_length, eos_token_id):
        requests = [{
            "input_ids": context,
            "input_lengths": [len(context)],
            "request_output_len": [max_length],
            "stop_words_list": [eos_token_id],
            "is_return_log_probs": [False],
        }]

        response = triton_completion(
            client=self.client,
            model_name=self.engine,
            requests=requests,
        )

        return response['output_ids']
