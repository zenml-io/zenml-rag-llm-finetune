import re
from typing import Dict, List, Tuple
import uuid

from llama_index.llms import OpenAI
from tqdm.notebook import tqdm
from zenml import step


@step()
def generate_queries(
    corpus: Dict[str, str],
    num_questions_per_chunk: int = 2,
    prompt_template: str = "",
    verbose=False,
) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """
    Automatically generate hypothetical questions that could be answered with
    doc in the corpus.
    """
    import os
    llm = OpenAI(model="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY"))

    prompt_template = (
        prompt_template
        or """\
    Context information is below.
    
    ---------------------
    {context_str}
    ---------------------
    
    Given the context information and not prior knowledge.
    generate only questions based on the below query.
    
    You are a Teacher/ Professor. Your task is to setup \
    {num_questions_per_chunk} questions for an upcoming \
    quiz/examination. The questions should be diverse in nature \
    across the document. Restrict the questions to the \
    context information provided."
    """
    )

    queries = {}
    relevant_docs = {}
    for node_id, text in tqdm(corpus.items()):
        query = prompt_template.format(
            context_str=text, num_questions_per_chunk=num_questions_per_chunk
        )
        response = llm.complete(query)

        result = str(response).strip().split("\n")
        questions = [
            re.sub(r"^\d+[\).\s]", "", question).strip() for question in result
        ]
        questions = [question for question in questions if len(question) > 0]

        for question in questions:
            question_id = str(uuid.uuid4())
            queries[question_id] = question
            relevant_docs[question_id] = [node_id]
    return queries, relevant_docs