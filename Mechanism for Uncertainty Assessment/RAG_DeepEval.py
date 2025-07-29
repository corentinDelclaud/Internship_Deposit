from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRelevancyMetric,
    ContextualRecallMetric,
)
from deepeval.test_case import LLMTestCase
from deepeval.models.base_model import DeepEvalBaseLLM
import transformers
import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import (
    build_transformers_prefix_allowed_tokens_fn,
)
from pydantic import BaseModel
import json
import csv
import os

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

        # Enhanced prompt template
        enhanced_prompt = f"""
    You are an impartial evaluator. Your task is to judge the quality of a response given a user input and, if available, additional context or an expected answer.

    Instructions:
    - Carefully read the user input, the actual response, and any provided context or expected answer.
    - Evaluate the response according to the specified metric.
    - Assign a score between 0 (poor) and 1 (excellent).
    - Provide a brief explanation for your score.

    Format your answer as:
    Score: <score>
    Reason: <your explanation>

    User Input: {prompt}
    """

        # Use enhanced_prompt instead of prompt
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
            parser = JsonSchemaParser(schema.schema())
            prefix_function = build_transformers_prefix_allowed_tokens_fn(
                pipeline.tokenizer, parser
            )
            output_dict = pipeline(enhanced_prompt, prefix_allowed_tokens_fn=prefix_function)
            output = output_dict[0]["generated_text"][len(enhanced_prompt):]
            print("Model output before JSON parsing:", repr(output))
            try:
                json_result = json.loads(output)
            except json.JSONDecodeError as e:
                print("JSON decode error:", e)
                json_result = {}
            return schema(**json_result)
        return pipeline(enhanced_prompt)

    async def a_generate(self, prompt: str, schema: BaseModel = None) -> BaseModel | str:
        return self.generate(prompt, schema)

    def get_model_name(self):
        return "mistralai/Mistral-7B-v0.1"

# Logging setup
log_file = "test_results.csv"
write_header = not os.path.exists(log_file)

def log_result(metric_name, test_input, actual_output, expected_output, retrieval_context, score, reason):
    global write_header
    with open(log_file, "a", newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "metric", "input", "actual_output", "expected_output",
                "retrieval_context", "score", "reason"
            ])
            write_header = False
        writer.writerow([
            metric_name,
            test_input,
            actual_output,
            expected_output if expected_output else "",
            "; ".join(retrieval_context) if retrieval_context else "",
            score,
            reason
        ])

custom_llm = CustomModel()

# Define test variables once
actual_output = "We offer a 30-day full refund at no extra cost."
expected_output = "You are eligible for a 30 day full refund at no extra cost."
retrieval_context = ["All customers are eligible for a 30 day full refund at no extra cost."]
test_input = "What if these shoes don't fit?"

# Answer Relevancy Metric
metric = AnswerRelevancyMetric(
    threshold=0.7,
    model=custom_llm,
    include_reason=True
)
test_case = LLMTestCase(
    input=test_input,
    actual_output=actual_output
)
evaluate(test_cases=[test_case], metrics=[metric])
log_result(
    "AnswerRelevancyMetric",
    test_input,
    actual_output,
    "",
    None,
    getattr(metric, "score", ""),
    getattr(metric, "reason", "")
)

# Faithfulness Metric
metric = FaithfulnessMetric(
    threshold=0.7,
    model=custom_llm,
    include_reason=True
)
test_case = LLMTestCase(
    input=test_input,
    actual_output=actual_output,
    retrieval_context=retrieval_context
)
metric.measure(test_case)
print(metric.score, metric.reason)
evaluate(test_cases=[test_case], metrics=[metric])
log_result(
    "FaithfulnessMetric",
    test_input,
    actual_output,
    "",
    retrieval_context,
    getattr(metric, "score", ""),
    getattr(metric, "reason", "")
)

# Contextual Precision Metric
metric = ContextualPrecisionMetric(
    threshold=0.7,
    model=custom_llm,
    include_reason=True
)
test_case = LLMTestCase(
    input=test_input,
    actual_output=actual_output,
    expected_output=expected_output,
    retrieval_context=retrieval_context
)
metric.measure(test_case)
print(metric.score, metric.reason)
evaluate(test_cases=[test_case], metrics=[metric])
log_result(
    "ContextualPrecisionMetric",
    test_input,
    actual_output,
    expected_output,
    retrieval_context,
    getattr(metric, "score", ""),
    getattr(metric, "reason", "")
)

# Contextual Recall Metric
metric = ContextualRecallMetric(
    threshold=0.7,
    model=custom_llm,
    include_reason=True
)
test_case = LLMTestCase(
    input=test_input,
    actual_output=actual_output,
    expected_output=expected_output,
    retrieval_context=retrieval_context
)
metric.measure(test_case)
print(metric.score, metric.reason)
evaluate(test_cases=[test_case], metrics=[metric])
log_result(
    "ContextualRecallMetric",
    test_input,
    actual_output,
    expected_output,
    retrieval_context,
    getattr(metric, "score", ""),
    getattr(metric, "reason", "")
)

# Contextual Relevancy Metric
metric = ContextualRelevancyMetric(
    threshold=0.6,
    model=custom_llm,
    include_reason=True
)
test_case = LLMTestCase(
    input=test_input,
    actual_output=actual_output,
    retrieval_context=retrieval_context
)
metric.measure(test_case)
print(metric.score, metric.reason)
evaluate(test_cases=[test_case], metrics=[metric])
log_result(
    "ContextualRelevancyMetric",
    test_input,
    actual_output,
    "",
    retrieval_context,
    getattr(metric, "score", ""),
    getattr(metric, "reason", "")
)