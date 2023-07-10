import sys
from train import train
from inference import inference
from evaluation import evaluation

if __name__ == "__main__":
    if not sys.argv[1] in [
        "train",
        "inference",
        "evaluation",
    ]:
        raise Exception("Prompt Unknown")

    if len(sys.argv) > 2:
        if not sys.argv[2] in ["kaggle_models"]:
            raise Exception("Prompt Unknown")

    if sys.argv[1] == "train":
        train()

    if sys.argv[1] == "inference":
        if len(sys.argv) > 2 and sys.argv[2] == "kaggle_models":
            inference(kaggle_models=True)
        else:
            inference()

    if sys.argv[1] == "evaluation":
        if len(sys.argv) > 2 and sys.argv[2] == "kaggle_models":
            evaluation(kaggle_models=True)
        else:
            evaluation()
