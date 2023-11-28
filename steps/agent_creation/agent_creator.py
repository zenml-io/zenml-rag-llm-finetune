import logging

from agent.prompt import PREFIX
from zenml import step

from llama_index import VectorStoreIndex
from llama_index.agent import OpenAIAgent
from llama_index.tools import QueryEngineTool, ToolMetadata

PIPELINE_NAME = "zenml_agent_creation_pipeline"


@step(enable_cache=False)
def agent_creator(vector_store: VectorStoreIndex) -> OpenAIAgent:
    """Create an agent from a vector store.

    Args:
        vector_store: Vector store to create agent from.

    Returns:
        An OpenAIAgent.
    """
    tools = [
        QueryEngineTool(
            query_engine=vector_store.as_query_engine(),
            metadata=ToolMetadata(
                name="zenml",
                description="Use this tool to answer questions about ZenML. "
                "How to debug errors in ZenML, how to answer conceptual "
                "questions about ZenML like available features, existing abstractions, "
                "and other parts from the documentation.",
            ),
        ),
    ]

    my_agent = OpenAIAgent.from_tools(tools=tools, system_prompt=PREFIX, verbose=True)
    logging.info("Agent created.")

    return my_agent
