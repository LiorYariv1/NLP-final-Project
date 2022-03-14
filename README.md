# NLP-final-Project
## This is our NLP final Project

###requirements:  <br>
pip install box <br>
pip install python-rake==1.4.4 <br>
pip install wandb <br>
pip install pywebio (only needed to run the web app)<br> 
pip install keybert (optional for kw extraction. our final model did not use this and the relevant code is commented)
<br>
### Reproduction Instructions: 
Run main.py to clean the data set, extract keywords and train the model. 
<br>
If needed, edit the config.yaml file with the right paths for reading the data and saving the model. <br>
You can also change the training arguments, but the config is ready to reproduct our results <br>
To compare the model at different checkpoints, you can run comparisonscript.py and see the results logged to wandb.

### To activate the web interface:
To run on our machine with our model:
1. start the azure vm
2. conda activate py38_pytorch
3. run pywebio_interface.py
(or just send us an email)

To run with a different trained model: <br>
change the model path in pywebio_interface.py and run the script

