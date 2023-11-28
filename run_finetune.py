from pipelines.finetune import finetuning_pipeline


if __name__ == "__main__":
    version = "0.47.0"
    docs_url = f"https://docs.zenml.io/v/{version}/"
    website_url = "https://zenml.io"
    repo_url = f"https://github.com/zenml-io/zenml/tree/{version}/examples"
    release_notes_url = (
        f"https://github.com/zenml-io/zenml/blob/{version}/RELEASE_NOTES.md"
    )

    import os
    os.environ["OPENAI_API_KEY"] = "API_KEY"
    finetuning_pipeline(
        website_url=website_url,
        docs_url=docs_url,
        repo_url=repo_url,
        release_notes_url=release_notes_url,
    )