# NLP-final-Project
## This is our NLP final Project

### requirements:  <br>
pip install box <br>
pip install keybert <br>
pip install python-rake==1.4.4 <br>
pip install wandb <br>
To run the web app - pip install pywebio <br>

### Reproduction Instructions - 
Run main.py to clean the data set, extract keywords and train the model. 
Edit the config.yaml file with the right paths for reading the data and saving the model. You can also change the training arguments
To compare the model at different checkpoints, you can run load_from_checkpoint.py and see the results logged to wandb.

### webapp instructions
To run on our machine with our model - start the machine and then run try_pywebio.py, enter the link that the script outputs. (or just send us an email)
To run with a different trained model - change the model path in  try_pywebio.py and run the script
