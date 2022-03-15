# NLP-final-Project


## Requirements:  <br>
pip install box <br>
pip install python-rake==1.4.4 <br>
pip install wandb <br>
pip install pywebio (only needed to run the web app)<br> 
pip install keybert (optional for kw extraction. our final model did not use this and the relevant code is commented)
<br>

## Reproduction Instructions: 
1. open a "data" directory for the movie datasets (files can be found on  https://technionmail-my.sharepoint.com/:f:/g/personal/atar_cohen_campus_technion_ac_il/EmeDCbfBN4xFqs-2LY4FOn0BSXiCMy0K7oTZsZRhLbP1rA?e=Z1qEEG)
2. Run main.py to clean the data set, extract keywords and train the model. 
The model will be saved and the training data will be logged into WandB. 
<br>
If needed, edit the config.yaml file with the right paths for reading the data and saving the model. <br>
You can also change the training arguments, but the config is ready to reproduct our results <br>
3. To obtain all comparison result, run comparisons.py The objective metrics will be logged to WandB, the plots will be saved to csv files
4. For the graphs and tables showed in the experiments part, run the  Experiments_Results.ipynb notebook
<br>
   

## To activate the web interface:
To run on our machine with our model:
1. start the azure vm
2. conda activate py38_pytorch
3. run pywebio_interface.py
(or just send us an email)
   
Note: The website reads the full_model_beams.csv file, if you dont have it, it can be reproduced in comparisons.py

To run with a different trained model: <br>
change the model path in pywebio_interface.py and run the script

![Web interface](https://github.com/LiorYariv1/NLP-final-Project/blob/main/img/ui.jpg?raw=true)
