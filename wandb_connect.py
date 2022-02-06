import wandb

wandb.init(project="test-project", entity="nlp_final_project")
wandb.log({"loss": loss})
