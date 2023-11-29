from zenml import pipeline
from zenml.model.model_version import ModelVersion

from steps.finetune_pipeline.url_scraper.url_scraper import url_scraper
from steps.finetune_pipeline.corpus_loader import load_corpus
from steps.finetune_pipeline.data_merger import merge_data
from steps.finetune_pipeline.evaluator import create_evaluator
from steps.finetune_pipeline.finetune_embeddings import (
    finetune_sentencetransformer_model,
)
from steps.finetune_pipeline.query_generator import generate_queries
from steps.finetune_pipeline.training_examples import generate_training_examples
from zenml.config import DockerSettings
from zenml.integrations.constants import OPEN_AI, PILLOW


docker_settings = DockerSettings(
    requirements="requirements.txt",
    required_integrations=[OPEN_AI, PILLOW],
)

@pipeline(
    name="finetuning_pipeline",
    enable_cache=True,
    settings={"docker": docker_settings},
    model_version=ModelVersion(
        name="finetuned-sentence-transformer",
        license="Apache",
        description="Custom Embeddings model",
        create_new_model_version=True,
        delete_new_version_on_failure=True,
    ),
)
def finetuning_pipeline(
    docs_url: str = "",
    repo_url: str = "",
    release_notes_url: str = "",
    website_url: str = "",
    model_id: str = "paraphrase-albert-small-v2",
    num_epochs: int = 2,
):
    train_urls, val_urls = url_scraper(
        docs_url, repo_url, release_notes_url, website_url
    )
    train_corpus = load_corpus(train_urls, id="train_loader")
    val_corpus = load_corpus(val_urls, id="val_loader")
    train_queries, train_relevant_docs = generate_queries(
        train_corpus, id="train_queries_generator"
    )
    val_queries, val_relevant_docs = generate_queries(
        val_corpus, id="val_queries_generator"
    )
    train_dataset, val_dataset = merge_data(
        train_corpus,
        train_queries,
        train_relevant_docs,
        val_corpus,
        val_queries,
        val_relevant_docs,
    )
    training_examples = generate_training_examples(train_dataset)
    evaluator = create_evaluator(val_dataset)
    model = finetune_sentencetransformer_model(
        loader=training_examples,
        evaluator=evaluator,
        model_id=model_id,
        EPOCHS=num_epochs,
    )
