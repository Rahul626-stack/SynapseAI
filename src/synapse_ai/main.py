#!/usr/bin/env python
"""
CLI entry point for synapse.ai.
Runs the crew locally with a sample PDF for testing.
"""
import warnings
from synapse_ai.crew import SynapseAICrew
from synapse_ai.blooms import build_bloom_prompt_instructions

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


def run():
    """
    Run the crew with a test PDF.
    Requires a PDF file — uses test.pdf from project root by default.
    """
    from synapse_ai.rag import ingest_pdf, retrieve_context
    from pathlib import Path

    # Look for test.pdf in project root
    project_root = Path(__file__).resolve().parents[2]
    test_pdf = project_root / "test.pdf"

    if not test_pdf.exists():
        raise FileNotFoundError(
            f"No test PDF found at {test_pdf}. "
            "Place a PDF named 'test.pdf' in the project root to test."
        )

    topic = "AI LLMs"
    num_questions = 10
    question_types = "multiple_choice, true_false, short_answer"

    # Ingest and retrieve
    print(f"Ingesting {test_pdf.name}...")
    col_name = ingest_pdf(str(test_pdf))

    print(f"Retrieving context for topic: {topic}...")
    context = retrieve_context(col_name, topic, top_k=5)

    bloom_instructions = build_bloom_prompt_instructions(num_questions)

    print("Running crew...")
    inputs = {
        "topic": topic,
        "context": context,
        "num_questions": num_questions,
        "question_types": question_types,
        "bloom_instructions": bloom_instructions,
        "randomness_instruction": "Generate creative questions.",
    }

    try:
        generator = SynapseAICrew()
        result = generator.crew().kickoff(inputs=inputs)
        print("\nQuiz generated successfully!")
        if hasattr(result, "pydantic") and result.pydantic:
            for i, q in enumerate(result.pydantic.questions, 1):
                print(f"\nQ{i} [{q.bloom_level}] ({q.difficulty}): {q.question}")
        return result
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


if __name__ == "__main__":
    run()
