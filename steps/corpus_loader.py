from typing import Dict, List

from llama_index import SimpleWebPageReader
from llama_index.node_parser import SimpleNodeParser
from zenml import step
from llama_index.schema import MetadataMode


@step()
def load_corpus(urls: List[str], verbose=False) -> Dict[str, str]:
    if verbose:
        print(f"Loading URLs {urls}")

    reader = SimpleWebPageReader(html_to_text=True)
    docs = reader.load_data(urls)
    if verbose:
        print(f"Loaded {len(docs)} docs")

    parser = SimpleNodeParser.from_defaults()
    nodes = parser.get_nodes_from_documents(docs, show_progress=verbose)

    if verbose:
        print(f"Parsed {len(nodes)} nodes")

    corpus = {
        node.node_id: node.get_content(metadata_mode=MetadataMode.NONE)
        for node in nodes
    }
    return corpus
