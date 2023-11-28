from typing import Annotated, Optional
from sentence_transformers import SentenceTransformer, losses
from torch.utils.data import DataLoader
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from zenml import step
from zenml.artifacts.artifact_config import ArtifactConfig


@step()
def finetune_sentencetransformer_model(
    loader: DataLoader,
    evaluator: InformationRetrievalEvaluator,
    EPOCHS: int = 2,
    model_id: Optional[str] = "BAAI/bge-small-en",
) -> Annotated[SentenceTransformer, ArtifactConfig(name="finetuned-sentence-transformer", is_model_artifact=True)]:
    model = SentenceTransformer(model_id)
    loss = losses.MultipleNegativesRankingLoss(model=model)

    warmup_steps = int(len(loader) * EPOCHS * 0.1)

    model.fit(
        train_objectives=[(loader, loss)],
        epochs=EPOCHS,
        warmup_steps=warmup_steps,
        show_progress_bar=True,
        evaluator=evaluator, 
        evaluation_steps=50,
    )

    return model