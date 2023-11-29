from typing import Annotated, Any, Dict, List, Tuple

from zenml import step


@step(enable_cache=True)
def generate_metrics(
    finetuned_results: List[Dict[str, Any]],
    standard_results: List[Dict[str, Any]],
) -> Tuple[Annotated[int, "finetuned_hits"], Annotated[int, "standard_hits"]]:
    """Generates metrics for the vector stores.

    Args:
        finetuned_results: Results to evaluate.
        standard_results: Results to evaluate.

    Returns:
        A tuple of hits for the vector stores.
    """
    finetuned_hits = _generate_hits(finetuned_results)
    standard_hits = _generate_hits(standard_results)

    return finetuned_hits, standard_hits


def _generate_hits(results: List[Dict[str, Any]]) -> int:
    """Generates hits for the vector stores.

    Args:
        results: Results to evaluate.

    Returns:
        Hits for the vector stores.
    """
    hits = 0
    for result in results:
        if result["is_hit"]:
            hits += 1
    return hits