# run = wandb.init(project="cat-classification", notes="", tags=["baseline", "paper1"])
# config = dict(
#     learning_rate=0.01, momentum=0.2, architecture="CNN", dataset_id="cats-0192"
# )


import random  # for demo script

# train.py
import wandb

wandb.login()

epochs = 10
lr = 0.01

run = wandb.init(
    # Set the project where this run will be logged
    project="my-awesome-project",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": lr,
        "epochs": epochs,
    })

offset = random.random() / 5
print(f"lr: {lr}")

# simulating a training run
acc_list = []
loss_list = []
for epoch in range(2, epochs):
    acc = 1 - 2 ** -epoch - random.random() / epoch - offset
    loss = 2 ** -epoch + random.random() / epoch + offset
    print(f"epoch={epoch}, accuracy={acc}, loss={loss}")
    wandb.log({"accuracy": acc, "loss": loss})
    acc_list.append(acc)
    loss_list.append(loss)
epoch_list = list(range(2,epochs))

data = [[x, y] for (x, y) in zip(epoch_list, loss_list)]
table = wandb.Table(data=data, columns = ["x", "y"])
wandb.log({"my_custom_plot_id" : wandb.plot.line(table, "xx", "yy",
    title="Custom Y vs X Line Plot")})
wandb.run.name = "my_run_name"
run.log_code()

# add table
artifact_obj = wandb.Artifact("my_artifact", type="dataset")
artifact_obj.add_file("bala.txt")
run.log_artifact(artifact_obj)
