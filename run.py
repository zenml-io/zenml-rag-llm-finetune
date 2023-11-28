# Apache Software License 2.0
#
# Copyright (c) ZenML GmbH 2023. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os

import click
from zenml.logger import get_logger

from pipelines.finetune import finetuning_pipeline

logger = get_logger(__name__)


@click.command(
    help="""
ZenML NLP project CLI v0.0.1.

Run the ZenML NLP project model training pipeline with various
options.

Examples:


  \b
  # Run the pipeline with default options
  python run.py
               
  \b
  # Run the pipeline without cache
  python run.py --no-cache

  \b
  # Run the pipeline without NA drop and normalization, 
  # but dropping columns [A,B,C] and keeping 10% of dataset 
  # as test set.
  python run.py --num-epochs 3 --train-batch-size 8 --eval-batch-size 8

  \b
  # Run the pipeline with Quality Gate for accuracy set at 90% for train set 
  # and 85% for test set. If any of accuracies will be lower - pipeline will fail.
  python run.py --min-train-accuracy 0.9 --min-test-accuracy 0.85 --fail-on-accuracy-quality-gates


"""
)
@click.option(
    "--no-cache",
    is_flag=True,
    default=False,
    help="Disable caching for the pipeline run.",
)
@click.option(
    "--num-epochs",
    default=1,
    type=click.INT,
    help="Number of epochs to train the model for.",
)
@click.option(
    "--finetune-pipeline",
    is_flag=True,
    default=False,
    help="Whether to run the pipeline that creates the dataset.",
)
@click.option(
    "--agent-creation-pipeline",
    is_flag=True,
    default=False,
    help="Whether to run the pipeline that trains the model.",
)
@click.option(
    "--website-url",
    default="https://zenml.io",
    type=click.STRING,
    help="URL of the website you'd like to train on.",
)
@click.option(
    "--model-id",
    default="paraphrase-albert-small-v2",
    type=click.STRING,
    help="Name of the Sentence Transformers Model to finetune.",
)
@click.option(
    "--model-version",
    default=0,
    type=click.INT,
    help="Version of the model to be used for agent.",
)
def main(
    no_cache: bool = True,
    num_epochs: int = 1,
    finetune_pipeline: bool = False,
    agent_creation_pipeline: bool = False,
    model_id: str = "paraphrase-albert-small-v2",
    model_version: int = 0,
    website_url: str = "https://zenml.io",
):
    """Main entry point for the pipeline execution.

    This entrypoint is where everything comes together:

      * configuring pipeline with the required parameters
        (some of which may come from command line arguments)
      * launching the pipeline
    """

    # Run a pipeline with the required parameters. This executes
    # all steps in the pipeline in the correct order using the orchestrator
    # stack component that is configured in your active ZenML stack.
    config_folder = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "configs",
    )

    pipeline_args = {}

    if no_cache:
        pipeline_args["enable_cache"] = False

    # Execute Feature Engineering Pipeline
    if finetune_pipeline:
        # pipeline_args["config_path"] = os.path.join(config_folder, "feature_engineering_config.yaml")
        run_args_finetune = {
            "model_id": model_id,
            "num_epochs": num_epochs,
            "website_url": website_url,
        }
        finetuning_pipeline.with_options(**pipeline_args)(
            **run_args_finetune
        )
        logger.info("Finetuning pipeline finished successfully!")

    # Execute Training Pipeline
    if agent_creation_pipeline:
        # pipeline_args["config_path"] = os.path.join(config_folder, "trainer_config.yaml")

        run_args_agent = {
            "website_url": website_url,
            "model_version": model_version,
        }

        agent_creation_pipeline.with_options(**pipeline_args)(
            **run_args_agent
        )
        logger.info("Agent creation pipeline finished successfully!")


if __name__ == "__main__":
    main()
