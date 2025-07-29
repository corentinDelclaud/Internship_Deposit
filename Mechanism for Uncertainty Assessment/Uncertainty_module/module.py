from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, ContextualPrecisionMetric,ContextualRelevancyMetric, ContextualRecallMetric
from deepeval.test_case import LLMTestCase
from deepeval.models.base_model import DeepEvalBaseLLM
import transformers
import torch
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from deepeval.models import DeepEvalBaseLLM


import transformers
from deepeval.models.base_model import DeepEvalBaseLLM
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import (
    build_transformers_prefix_allowed_tokens_fn,
)
from pydantic import BaseModel
import json


class CustomModel(DeepEvalBaseLLM):
    def __init__(self):
        self.model_id = "mistralai/Mistral-7B-v0.1"
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            # Load the model with quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
        )

    def load_model(self):
        return self.model

    def generate(self, prompt: str, schema: BaseModel = None) -> BaseModel | str:
        model = self.load_model()

        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
            use_cache=True,
            device_map="auto",
            max_new_tokens=1024,
            do_sample=True,
            top_k=5,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        if schema is not None:
            # Create parser required for JSON confinement using lmformatenforcer
            parser = JsonSchemaParser(schema.schema())
            prefix_function = build_transformers_prefix_allowed_tokens_fn(
                pipeline.tokenizer, parser
            )

            # Output and load valid JSON
            output_dict = pipeline(prompt, prefix_allowed_tokens_fn=prefix_function)
            output = output_dict[0]["generated_text"][len(prompt) :]
            json_result = json.loads(output)

            # Return valid JSON object according to the schema DeepEval supplied
            return schema(**json_result)
        return pipeline(prompt)

    async def a_generate(self, prompt: str, schema: BaseModel = None) -> BaseModel | str:
        return self.generate(prompt, schema)

    def get_model_name(self):
        return "mistralai/Mistral-7B-v0.1"