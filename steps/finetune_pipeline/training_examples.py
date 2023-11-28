from typing import Any, Dict
from torch.utils.data import DataLoader
from sentence_transformers import InputExample
from zenml import step


@step()
def generate_training_examples(
    dataset: Dict[str, Any], batch_size: int = 10
) -> DataLoader:
    """Generate training examples from the dataset.
    
    Args:
        dataset: Dataset containing the corpus, queries and relevant docs.
        batch_size: Batch size for the dataloader.
        
    Returns:
        DataLoader containing the training examples.
    """
    corpus = dataset['corpus']
    queries = dataset['queries']
    relevant_docs = dataset['relevant_docs']

    examples = []
    for query_id, query in queries.items():
        node_id = relevant_docs[query_id][0]
        text = corpus[node_id]
        example = InputExample(texts=[query, text])
        examples.append(example)

    return DataLoader(examples, batch_size=batch_size)