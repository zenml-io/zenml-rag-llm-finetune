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

from typing import Annotated, List, Tuple

from steps.finetune_pipeline.url_scraper.url_scraping_utils import get_all_pages, get_nested_readme_urls
from zenml import step


@step(enable_cache=True)
def url_scraper(
    docs_url: str = "",
    repo_url: str = "",
    release_notes_url: str = "",
    website_url: str = "",
) -> Tuple[Annotated[List, "train_urls"], Annotated[List, "val_urls"]]:
    """Generates a list of relevant URLs to scrape.

    Args:
        docs_url: URL to the documentation.
        repo_url: URL to the repository.
        release_notes_url: URL to the release notes.
        website_url: URL to the website.

    Returns:
        List of URLs to scrape.
    """
    # examples_readme_urls = get_nested_readme_urls(repo_url)
    # docs_urls = get_all_pages(docs_url, finetuning=True)
    # website_urls = get_all_pages(website_url, finetuning=True)
    # all_urls = docs_urls + website_urls + [release_notes_url]

    # # split into train and val sets
    # train_urls = all_urls[: int(0.8 * len(all_urls))]
    # val_urls = all_urls[int(0.8 * len(all_urls)) :]

    return [website_url], [website_url]

    return train_urls, val_urls
