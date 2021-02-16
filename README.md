# What is it?
This is an application that predicts the score of basketball games using the RandomForestRegressor model.

# Python Version
Python 3.9

# Libraries used
<ul>
  <li>sklearn</li>
  <li>pandas</li>
  <li>numpy</li>
  <li>tkinter</li>
 </ul>

# How it works
You will first be asked to choose a dataset file from a tkinter box. The dataset is provided in this repository. If you dont want to use the dataset there is a function to download the data from the API. This will take about 20 minutes as you are downloading the complete schedule and statistics for every NCAA basketball team. 

Once the data has been retrieved it will create two CSV's one is the actual score of the game, the other is the predicted. If you compare the two you will see that the algorithm is only wrong 5% of the time. This could be a sign of over fitting and is on my todo's.

# todo improvements
1. Ability to input two team names and let the machine predict. 
2. Cleanup the functions
3. Overfitting
