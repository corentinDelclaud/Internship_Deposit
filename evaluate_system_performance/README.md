How to Use

For automatic metrics (EM, F1):

python generalRAGevaluator.py --predictions example\predictions.json --references example\references.json

python generalRAGevaluator.py --predictions example\fakepredictionsquestions_2WikiMultihopQA_structured.json --references example\questions2WikiMultihopQA_structured.json --max_eval 500
&&
python generalRAGevaluator.py --predictions example\fakepredictionsquestions_questionsIIRC_structured.json --references example\questionsIIRC_structured.json --max_eval 500
&&
python generalRAGevaluator.py --predictions example\fakepredictionsquestions_questionsStrategyQA_structured.json --references example\questionsStrategyQA_structured.json --max_eval 500



For LLM-based comparative evaluation (between two RAGs):

python generalRAGevaluator.py --predictions example\predictions.json --references \example\references.json --comparative --other_predictions example\other_predictions.json --model_name meta-llama/Llama-3.2-1B-Instruct

