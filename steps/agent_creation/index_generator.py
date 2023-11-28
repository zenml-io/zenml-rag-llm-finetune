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

from typing import List, Optional
from sentence_transformers import SentenceTransformer

# initialize simple vector indices
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.embeddings.openai import OpenAIEmbedding

from llama_index.schema import Document
from zenml import step


@step(enable_cache=True)
def index_generator(
    documents: List[Document], model: Optional[SentenceTransformer] = None
) -> VectorStoreIndex:
    if model is None:
        embed_model = OpenAIEmbedding()
    else:
        # write model to a file in current directory
        model.save("./finetuned_model")
        embed_model = "local:finetuned_model"

    service_context = ServiceContext.from_defaults(embed_model=embed_model)
    return VectorStoreIndex.from_documents(
        documents=documents, service_context=service_context
    )