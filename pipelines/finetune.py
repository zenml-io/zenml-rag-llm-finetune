from zenml import pipeline

from steps.url_scraper import url_scraper
from steps.corpus_loader import load_corpus
from steps.data_merger import merge_data
from steps.evaluator import create_evaluator
from steps.finetune_embeddings import finetune_sentencetransformer_model
from steps.query_generator import generate_queries
from steps.training_examples import generate_training_examples


@pipeline(name="finetuning_pipeline", enable_cache=True)
def finetuning_pipeline(
    docs_url: str = "",
    repo_url: str = "",
    release_notes_url: str = "",
    website_url: str = "",
):
    train_urls, val_urls = url_scraper.url_scraper(docs_url, repo_url, release_notes_url, website_url)
    train_corpus = load_corpus(train_urls, id="train_loader")
    val_corpus = load_corpus(val_urls, id="val_loader")
    train_queries, train_relevant_docs = generate_queries(train_corpus, id="train_queries_generator")
    val_queries, val_relevant_docs = generate_queries(val_corpus, id="val_queries_generator")
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
    model = finetune_sentencetransformer_model(training_examples, evaluator)
    