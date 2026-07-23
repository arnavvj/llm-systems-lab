from engine import InferenceEngine
from evaluator import Evaluator


MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"

engine = InferenceEngine(MODEL_NAME)
evaluator = Evaluator(engine)


dataset = [
    {
        "prompt": "What is the capital of France?",
        "reference": "Paris",
    },
    {
        "prompt": "Who discovered gravity?",
        "reference": "Isaac Newton",
    },
    {
        "prompt": "What is 2 + 2?",
        "reference": "4",
    },
]


def main():
    results = evaluator.evaluate(dataset)
    evaluator.print_report(results)


if __name__ == "__main__":
    main()