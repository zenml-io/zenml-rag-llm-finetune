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


from steps.agent_creation.agent_creator import agent_creator
from steps.agent_creation.evaluator import evaluate_vector_stores
from steps.agent_creation.index_generator import index_generator
from steps.agent_creation.index_generator2 import index_generator2
from steps.agent_creation.metrics import generate_metrics
from steps.agent_creation.url_scraper.url_scraper import url_scraper
from steps.agent_creation.web_url_loader import web_url_loader
from zenml import pipeline, get_pipeline_context
from zenml.artifacts.external_artifact import ExternalArtifact
from zenml.config import DockerSettings
from zenml.integrations.constants import OPEN_AI, PILLOW
from zenml.model.model_version import ModelVersion

PIPELINE_NAME = "zenml_agent_creation_pipeline"

docker_settings = DockerSettings(
    requirements="requirements.txt",
    required_integrations=[OPEN_AI, PILLOW],
)


@pipeline(
    name=PIPELINE_NAME,
    enable_cache=True,
    settings={"docker": docker_settings},
    model_version=ModelVersion(
        name="finetuned-sentence-transformer",
    ),
    extra={"trained_embeddings": "finetuned-sentence-transformer"},
)
def docs_to_agent_pipeline(
    docs_url: str = "",
    repo_url: str = "",
    release_notes_url: str = "",
    website_url: str = "",
    model_version: int = None,
) -> None:
    """Generate index for ZenML.

    Args:
        docs_url: URL to the documentation.
        repo_url: URL to the repository.
        release_notes_url: URL to the release notes.
        website_url: URL to the website.
        model_version: Version of the model to be used.
    """
    urls = url_scraper(docs_url, repo_url, release_notes_url, website_url)
    documents = web_url_loader(urls)
    trained_embeddings = get_pipeline_context().extra["trained_embeddings"]
    enhanced_vector_store = index_generator(
        model=ExternalArtifact(
            name=trained_embeddings,
        ),
        documents=documents,
        id="enhanced_index_generator",
    )
    standard_vector_store = index_generator2(
        documents=documents,
        id="standard_index_generator",
    )
    finetuned_results, standard_results = evaluate_vector_stores(
        enhanced_vector_store=enhanced_vector_store,
        standard_vector_store=standard_vector_store,
        id="evaluator",
    )
    finetuned_hits, standard_hits = generate_metrics(
        finetuned_results, standard_results, id="metrics"
    )
    finetuned_agent = agent_creator(
        vector_store=enhanced_vector_store, id="finetuned_agent_creator"
    )
    standard_agent = agent_creator(
        vector_store=standard_vector_store, id="standard_agent_creator"
    )
