import comet_ml
from comet_ml import Experiment
from time import sleep
experiment = Experiment(project_name="test-comet")
experiment.set_name("testing comet")
experiment.log_other("finished", False)
for y in range(10):
   experiment.log_metric("a", y)
   experiment.log_parameter("paramA", 0.31000005)
   sleep(0.1)
experiment.log_other("finished", True)
experiment.log_confusion_matrix(
    ["dog", "cat", "frog"],
    ["dog", "cat", "frog"],
    title="Confusion Matrix",
    file_name="confusion-matrix.png",
)
comet_ml.get_global_experiment()


import comet_ml

# Create a Comet Artifact
artifact = comet_ml.Artifact(
    name="california",
    artifact_type="dataset",
    aliases=["raw"],
    metadata={"task": "regression"},
)

# Add files to the Artifact
for split, asset in zip(
    ["train", "test"], ["./datasets/train.csv", "./datasets/test.csv"]
):
    artifact.add(asset, metadata={"dataset_stage": "raw", "dataset_split": split})

experiment = comet_ml.Experiment(
    api_key="<Your API Key>",
    project_name="<Your Project Name>"
)
experiment.add_tag("upload")
experiment.log_artifact(artifact)

experiment.end()