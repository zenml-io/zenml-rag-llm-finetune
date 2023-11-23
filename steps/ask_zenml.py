from __future__ import annotations

PIPELINE_NAME = "zenml_agent_creation_pipeline"

PREFIX = """ZenML Agent is a large language model operated by ZenML. It can answer user's questions about ZenML for the right version
by first asking the user what version they are using. ZenML agent DOES NOT make up
answers by itself. it always uses the right tools available, given the context. NEVER make up answers.

It is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a large number of topics, including about ZenML.
When a user asks a question pertaining to ZenML, *YOU MUST* ask the user for the version of ZenML that they are using and 
then use a tool corresponding to the right version to answer questions based on that.

It is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, ZenML Agent is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

Overall, ZenML Agent is a powerful system that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, ZenML Agent is here to assist."""

SUFFIX = """TOOLS
------
ZenML Agent can ask the user to use tools to look up information that may be helpful in answering the users original question. The tools the human can use are:

{{tools}}

{format_instructions}

USER'S INPUT
--------------------
Here is the user's input (remember to respond with a markdown code snippet of a json blob with a single action, and NOTHING else):

{{{{input}}}}"""

FORMAT_INSTRUCTIONS = """RESPONSE FORMAT INSTRUCTIONS
----------------------------

When responding to me, please output a response in one of two formats:

**Option 1:**
Use this if you want to use a tool to answer a question. For example, if you have already
asked the user/human for the version they are using, you can then use the corresponding tool
to answer their question.
Markdown code snippet formatted in the following schema:

```json
{{{{
    "action": string, \\ The action to take. Must be one of {tool_names}
    "action_input": string \\ The input to the action
}}}}
```

**Option #2:**
Use this when you have to ask the user about the ZenML version. Or for other general direct chat with the human. Markdown code snippet formatted in the following schema:

```json
{{{{
    "action": "Final Answer",
    "action_input": string \\ You should put what you want to return to user here
}}}}
```"""

from zenml.client import Client

from langchain.agents import ConversationalChatAgent

from typing import List, Optional, Sequence


from langchain.tools.base import BaseTool

from langchain.schema import (
    BaseOutputParser,
)
from langchain.prompts.base import BasePromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)


class ZenMLAgent(ConversationalChatAgent):
    """ZenML flavored Conversational Chat Agent."""

    @classmethod
    def create_prompt(
        cls,
        tools: Sequence[BaseTool],
        system_message: str = PREFIX,
        human_message: str = SUFFIX,
        input_variables: Optional[List[str]] = None,
        output_parser: Optional[BaseOutputParser] = None,
    ) -> BasePromptTemplate:
        tool_strings = "\n".join(
            [f"> {tool.name}: {tool.description}" for tool in tools]
        )
        tool_names = ", ".join([tool.name for tool in tools])
        _output_parser = output_parser or cls._get_default_output_parser()
        format_instructions = human_message.format(
            format_instructions=_output_parser.get_format_instructions()
        )
        final_prompt = format_instructions.format(
            tool_names=tool_names, tools=tool_strings
        )
        if input_variables is None:
            input_variables = ["input", "chat_history", "agent_scratchpad"]
        messages = [
            SystemMessagePromptTemplate.from_template(system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template(final_prompt),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
        return ChatPromptTemplate(
            input_variables=input_variables, messages=messages
        )


if __name__ == "__main__":
    # accept a question as argument
    import sys

    question = sys.argv[1:]

    pipeline_model = Client().get_pipeline(name_id_or_prefix=PIPELINE_NAME)

    # get the pipeline version as an int
    pipeline_version = int(pipeline_model.version)

    # get the latest run of the previous version

    version = "0.44.1"

    existing_tools = []
    llm = {
        "temperature": 0,
        "max_tokens": 1000,
        "model_name": "gpt-3.5-turbo",
    }

    if pipeline_model.runs is not None:
        # get the last run
        last_run = pipeline_model.runs[]
        # get the agent_creator step
        agent_creator_step = last_run.steps["agent_creator"]

        agent = agent_creator_step.output.load()

        chat_history = []
        agent.run({"input": question[0], "chat_history": chat_history})
