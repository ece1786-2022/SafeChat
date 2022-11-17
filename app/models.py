import numpy as np
import pandas as pd
import openai


from app import webapp
from typing import List, Dict, Tuple


openai.api_key = webapp.config['OPENAI_KEY']


class Classification:

    def __init__(
        self, 
        model: str,
        temperature: float, 
        max_length: int, 
        top_p: float, 
        frequency_penalty: float, 
        presence_penalty: float, 
        best_of: float,
        prompt: str
    ) -> None:

        self.configs = {
            "engine": model,
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
        prompt_text = f"{self.prompt}\n\n{self.inputstart}:{input_text}\n{self.outputstart}: "
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

        return text, self._compute_prob(top_logprob)
    
    def _compute_prob(self, logprob: float) -> Tuple[float, float]:
        return 100*np.e**logprob

                


