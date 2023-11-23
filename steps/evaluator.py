from typing import Any, Dict
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from zenml import step


@step()
def create_evaluator(dataset: Dict[str, Any]) -> InformationRetrievalEvaluator:
    """Generate training examples from the dataset.

    Args:
        dataset: Dataset containing the corpus, queries and relevant docs.

    Returns:
        InformationRetrievalEvaluator for the dataset.
    """
    corpus = dataset["corpus"]
    queries = dataset["queries"]
    relevant_docs = dataset["relevant_docs"]

    return InformationRetrievalEvaluator(queries, corpus, relevant_docs)