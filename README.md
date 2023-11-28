<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

<div align="center">

  <!-- PROJECT LOGO -->
  <br />
    <a href="https://zenml.io">
      <img src="assets/header.png" alt="ZenML Logo">
    </a>
  <br />

  [![PyPi][pypi-shield]][pypi-url]
  [![PyPi][pypiversion-shield]][pypi-url]
  [![PyPi][downloads-shield]][downloads-url]
  [![Contributors][contributors-shield]][contributors-url]
  [![License][license-shield]][license-url]
  <!-- [![Build][build-shield]][build-url] -->
  <!-- [![CodeCov][codecov-shield]][codecov-url] -->

</div>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[pypi-shield]: https://img.shields.io/pypi/pyversions/zenml?color=281158

[pypi-url]: https://pypi.org/project/zenml/

[pypiversion-shield]: https://img.shields.io/pypi/v/zenml?color=361776

[downloads-shield]: https://img.shields.io/pypi/dm/zenml?color=431D93

[downloads-url]: https://pypi.org/project/zenml/

[codecov-shield]: https://img.shields.io/codecov/c/gh/zenml-io/zenml?color=7A3EF4

[codecov-url]: https://codecov.io/gh/zenml-io/zenml

[contributors-shield]: https://img.shields.io/github/contributors/zenml-io/zenml?color=7A3EF4

[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors

[license-shield]: https://img.shields.io/github/license/zenml-io/zenml?color=9565F6

[license-url]: https://github.com/zenml-io/zenml/blob/main/LICENSE

[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555

[linkedin-url]: https://www.linkedin.com/company/zenml/

[twitter-shield]: https://img.shields.io/twitter/follow/zenml_io?style=for-the-badge

[twitter-url]: https://twitter.com/zenml_io

[slack-shield]: https://img.shields.io/badge/-Slack-black.svg?style=for-the-badge&logo=linkedin&colorB=555

[slack-url]: https://zenml.io/slack-invite

[build-shield]: https://img.shields.io/github/workflow/status/zenml-io/zenml/Build,%20Lint,%20Unit%20&%20Integration%20Test/develop?logo=github&style=for-the-badge

[build-url]: https://github.com/zenml-io/zenml/actions/workflows/ci.yml

<div align="center">
  <h3 align="center">Build portable, production-ready MLOps pipelines.</h3>
  <p align="center">
    <div align="center">
      Join our <a href="https://zenml.io/slack-invite" target="_blank">
      <img width="18" src="https://cdn3.iconfinder.com/data/icons/logos-and-brands-adobe/512/306_Slack-512.png" alt="Slack"/>
    <b>Slack Community</b> </a> and be part of the ZenML family.
    </div>
    <br />
    <a href="https://zenml.io/features">Features</a>
    ·
    <a href="https://zenml.io/roadmap">Roadmap</a>
    ·
    <a href="https://github.com/zenml-io/zenml/issues">Report Bug</a>
    ·
    <a href="https://zenml.io/discussion">Vote New Features</a>
    ·
    <a href="https://zenml.io/blog">Read Blog</a>
    ·
    <a href="https://www.zenml.io/company#team">Meet the Team</a>
    <br />
  </p>
</div>

---

<!-- TABLE OF CONTENTS -->
<!-- TABLE OF CONTENTS -->
<details>
  <summary>🏁 Table of Contents</summary>
  <ol>
    <li><a href="#-huggingface-model-to-sagemaker-endpoint-mlops-with-zenml">Introduction</a></li>
    <li><a href="#👋-get-started">Get Started</a></li>
    <li>
      <a href="#🧑‍💻-how-to-run-this-project">How To Run This Project</a>
      <ul>
        <li><a href="#📓-either-use-a-jupyter-notebook">EITHER: Use a Jupyter notebook</a></li>
        <li><a href="#✍️-or-run-it-locally">OR: Run it locally</a>
          <ul>
            <li><a href="#👶-step-1-start-with-fine-tuning-the-embeddings-model">Step 1: Start with feature engineering</a></li>
            <li><a href="#💪-step-2-create-an-ai-agent-using-your-embeddings-model">Step 2: Train the model</a></li>
          </ul>
        </li>
      </ul>
    </li>
    <li><a href="#🆘-getting-help">Getting Help</a></li>
  </ol>
</details>
<br />

# 🧬 Finetune your RAG pipeline with ZenML

RAG pipelines are all the rage these days. They allow you to ask LLMs any questions, grounded by your own data. There is now a more-or-less established way of doing this and it involves creating vector stores of your data and retreiving the top-k results using a query model, when the question is asked. This is a great way to get started, but it has been observed that finetuning the embeddings model leads to better results. In this project, you'll build a ZenML pipeline that can help you not just finetune your own model but also use it in a different pipeline to build AI Agents out of it, all tracked and versioned through ZenML and accessible through the ZenML Dashboard.

This project showcases one way of using [ZenML](https://zenml.io) pipelines to achieve this:

- Train/Fine-tune a Sentence Transformers embeddings model using your data in `finetuning_pipeline`.
- Use the Model Control Plane (MCP) to make this model available to the index creation step of the `agent_creation` pipeline.
- Create an AI Agent using the `agent_creation` pipeline that uses the vector store built with your embeddings model to answer questions based on your data.

Here is an overview of the entire process:

TODO: Add image

The above flow is achieved in a repeatable, fully tracked pipeline that is observable across the organization. Let's
see how this works.

## 👋 Get started

What to do first? You can start by giving the the project a quick run. The
project is ready to be used and can run as-is without any further code
changes! You can try it right away by installing ZenML, the needed
ZenML integration and then calling the CLI included in the project.

<details>
<summary><h3>Install requirements</h3></summary>

```bash
# Clone this repo
git clone git@github.com:zenml-io/zenml-rag-llm-finetune.git

# Set up a Python virtual environment, if you haven't already
python3 -m venv .venv
source .venv/bin/activate

# Install requirements & integrations
# Alternatively see the Makefile for commands to use
make setup
```

</details>

<details>
<summary><h3>Connect to a deployed ZenML server</h3></summary>

After this, you should have ZenML and all of the requirements of the project installed locally.
Next thing to do is to connect to a [deployed ZenML instance](https://docs.zenml.io/deploying-zenml/). You can
create a free trial using [ZenML Cloud](https://cloud.zenml.io) to get setup quickly.

Once you have your deployed ZenML ready, you can connect to it using:

```shell
zenml connect --url YOUR_ZENML_SERVER_URL
```

This will open up the browser for your to connect to a deployed ZenML!

You will be able to see your models and related artifacts and pipeline runs in the Model Control Plane now (available only on ZenML Cloud).

</details>

<details>
<summary><h3>Set up your local stack</h3></summary>

To run this project, you need to create a [ZenML Stack](https://docs.zenml.io/user-guide/starter-guide/understand-stacks) with the required components to run the pipelines.

```shell
make install-stack
```

</details>

## 🧑‍💻 How To Run This Project

There are two paths you can take this with the project. You can either
use a notebook or run it in scripts. Choose whichever path suits your learning
style.

You can also watch a full video walkthrough on YouTube:

TODO: Add video

### 📓 EITHER: Use a Jupyter notebook

```shell
# Install jupyter
pip install notebook

# Go to finetune.ipynb and agent_creation.ipynb
jupyter notebook
```

### ✍️ OR: Run it locally

If you're note the notebook type, you can use this README to run the pipelines one by one.

<details>

<summary><h3>Instructions to run locally</h3></summary>

At any time, you can look at the CLI help to see what you can do with the project:
  
```shell
python run.py --help
```

Let's walk through the process one by one:

#### 👶 Step 1: Start with fine-tuning the embeddings model

The first pipeline is the fine-tune embeddings pipeline. This pipeline loads some of your data and trains a Sentence Transformers model on it. This model is tracked through the Model Control Plane and is associated with the pipeline.

TODO: image

Run it as follows:

```shell
python run.py --finetune-pipeline --no-cache --num-epochs 1 --model-id "paraphrase-albert-small-v2"
```

This will train a model from the Sentence Transformers library and register a new ZenML model on the Model Control Plane:

TODO: MCP image

Please note the above screens are a cloud-only feature in [ZenML Cloud](https://zenml.io/cloud), and
the CLI `zenml models list` should be used instead for OSS users.
In the dashboard, you can click on these artifacts and look at the lineage of the model.

TODO: section video

#### 💪 Step 2: Create an AI Agent using your embeddings model.

The `agent_creation` pipeline uses the embeddings model from the previous pipeline to create a vector store index which an AI Agent can use to answer questions. This pipeline is also tracked through the Model Control Plane and is associated with the embeddings model.

TODO: image

Run it as follows:

```shell
python run.py --agent-creation-pipeline --no-cache
```

Or if you'd like to use a specific version of the model, you can specify it as follows:

```shell
python run.py --agent-creation-pipeline --no-cache --model-version <INT>
```

TODO: section video

## 🆘 Getting Help

Something didn't work? No problem!

The first point of call should
be [the ZenML Slack group](https://zenml.io/slack/).
Ask your questions about bugs or specific use cases, and someone from
the [core team](https://zenml.io/company#CompanyTeam) will respond.
Or, if you
prefer, [open an issue](https://github.com/zenml-io/zenml-huggingface-sagemaker/issues/new/choose) on
this GitHub repo.
