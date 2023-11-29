#  Copyright (c) ZenML GmbH 2023. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at:
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
#  or implied. See the License for the specific language governing
#  permissions and limitations under the License.

from typing import Annotated, Any, Dict, List, Tuple

# initialize simple vector indices
from llama_index import VectorStoreIndex

import tqdm
from zenml import step


@step(enable_cache=True)
def evaluate_vector_stores(
    enhanced_vector_store: VectorStoreIndex,
    standard_vector_store: VectorStoreIndex,
) -> Tuple[Annotated[List, "finetuned_results"], Annotated[List, "standard_results"]]:
    """Evaluates the vector stores.

    Args:
        enhanced_vector_store: Vector store to evaluate.
        standard_vector_store: Vector store to evaluate.
        val_corpus: Validation corpus.

    Returns:
        A list of evaluation results for all queries.
    """
    queries, relevant_docs = _fetch_queries_and_relevant_docs()
    finetuned_results = _evaluate(enhanced_vector_store, queries, relevant_docs)
    standard_results = _evaluate(standard_vector_store, queries, relevant_docs)

    return finetuned_results, standard_results


def _fetch_queries_and_relevant_docs() -> Tuple[Dict[str, str], Dict[str, str]]:
    """Fetches the queries and relevant docs for evaluation.

    Returns:
        A tuple of queries and relevant docs.
    """
    from zenml.client import Client

    pipeline_model = Client().get_pipeline(name_id_or_prefix="finetuning_pipeline")

    if pipeline_model.runs is not None:
        # get the last run
        last_run = pipeline_model.runs[0]
        # get the agent_creator step
        queries_steps = last_run.steps["val_queries_generator"]

        try:
            queries = queries_steps.outputs["output_0"].load()
            relevant_docs = queries_steps.outputs["output_1"].load()
        except ValueError:
            pass

    return queries, relevant_docs


def _evaluate(
    vector_store: VectorStoreIndex,
    queries: Dict[str, str],
    relevant_docs: Dict[str, str],
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """Evaluates the vector stores.

    Args:
        vector_store: Vector store to evaluate.
        queries: Queries to evaluate.
        relevant_docs: Known relevant docs for each query.

    Returns:
        A list of evaluation results for all queries.
    """
    retriever = vector_store.as_retriever(similarity_top_k=top_k)

    eval_results = []
    for query_id, query in tqdm.tqdm(queries.items()):
        retrieved_nodes = retriever.retrieve(query)
        retrieved_ids = [node.node.node_id for node in retrieved_nodes]
        expected_id = relevant_docs[query_id][0]
        is_hit = expected_id in retrieved_ids  # assume 1 relevant doc

        eval_result = {
            "is_hit": is_hit,
            "retrieved": retrieved_ids,
            "expected": expected_id,
            "query": query_id,
        }
        eval_results.append(eval_result)
    return eval_results