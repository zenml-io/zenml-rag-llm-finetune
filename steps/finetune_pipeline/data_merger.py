from typing import Any, Dict, List, Tuple

from zenml import step


@step()
def merge_data(
    train_corpus: Dict[str, str],
    train_queries: Dict[str, str],
    train_relevant_docs: Dict[str, List[str]],
    val_corpus: Dict[str, str],
    val_queries: Dict[str, str],
    val_relevant_docs: Dict[str, List[str]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    train_dataset = {
        "queries": train_queries,
        "corpus": train_corpus,
        "relevant_docs": train_relevant_docs,
    }

    val_dataset = {
        "queries": val_queries,
        "corpus": val_corpus,
        "relevant_docs": val_relevant_docs,
    }

    return train_dataset, val_dataset
