import torch
import numpy as np
import pandas as pd
import openai
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, GPT2Config


from app import webapp
from typing import List, Dict, Tuple


openai.api_key = webapp.config['OPENAI_KEY']


class Classification:

    def __init__(
        self, 
        model='Zero-shot',
        temperature=1.0, 
        max_length=16, 
        top_p=1.0, 
        frequency_penalty=0.0, 
        presence_penalty=0.0, 
        best_of=1,
        prompt=''
    ) -> None:

        if model == 'Zero-shot':
            selected_model = 'text-davinci-002'
        elif model == 'One/Few-shot':
            selected_model = 'text-davinci-002'
        else:
            selected_model = 'text-davinci-002'

        self.configs = {
            "engine": selected_model,
            "temperature": temperature,
            "max_tokens": max_length,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "best_of": best_of
        }
        self.inputstart = "[TEXT]"
        self.outputstart = "[ANSWER]"
        self.prompt = prompt.strip()

    def ask(self, input_text: str):
        prompt_text = f"{self.prompt}\n\n{self.inputstart}: {input_text}\n{self.outputstart}: "
        response = openai.Completion.create(
            prompt=prompt_text,
            stop=[" {}:".format(self.outputstart)],
            logprobs=1,
            **self.configs
        )

        print(response)

        top_words = response["choices"][0]["logprobs"]["top_logprobs"][0]
        _, top_logprob = list(top_words.items())[0]
        text = response["choices"][0]["text"]

        return text, self._compute_prob_gpt3(top_logprob)

    def _compute_prob_gpt3(self, logprob: float) -> Tuple[float, float]:
        return 100 * np.e**logprob


class GPT2Classification(Classification):

    def __init__(
        self, 
        model='Zero-shot',
        temperature=1.0, 
        max_length=16, 
        top_p=1.0, 
        num_beams=1,
        diversity_penalty=0.0,
        repetition_penalty=1.0,
        length_penalty=1.0,
        prompt=''
    ) -> None:

        # setup baseline model
        self.configs = {
            "temperature": temperature,
            "top_p": top_p,
            "max_length": max_length,
            "diversity_penalty": diversity_penalty,
            "repetition_penalty": repetition_penalty,
            "length_penalty": length_penalty,
            "num_beams": num_beams
        }

        # TODO: push a finetuned gpt2 image to Huggingface and then pull it here
        self.model_config = GPT2Config.from_pretrained('gpt2', num_labels=2, **self.configs) # Binary Classification
        self.model = GPT2ForSequenceClassification.from_pretrained('gpt2', config=self.model_config)

        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.config.pad_token_id = self.model.config.eos_token_id
        self.inputstart = "[TEXT]"
        self.outputstart = "[ANSWER]"
        self.prompt = prompt.strip()

    def ask(self, input_text: str):
        prompt_text = f"{self.prompt}\n\n{self.inputstart}: {input_text}\n{self.outputstart}: "
        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        outputs = self.model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        logits = outputs.logits
        probs, indices = self._compute_prob_baseline(logits)
        prob = probs.item()
        cls = indices.item()
        if cls == 1:
            return "suicide", prob
        else:
            return "non-suicide", prob
        
    def _compute_prob_baseline(self, logits):
        softmax = torch.nn.Softmax(dim=1)
        probs = softmax(logits)
        return torch.topk(probs, k=1)
