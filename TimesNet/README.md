
## Overview of files 

+ The folder custom_scripts contains a file with shell scripts, which stores the timesNet parameters that were used during training. The python file data-prep.py creates custom .csv files out of the training data, which are fed as input to TimesNet. The other python files conduct the rolling 1-day or 20-day forecast, either with or without exogenous variables. 
+ The folder plots_and_metrics stores the summarized results (plots and metrics tables) of the rolling forecasts performed by the aforementioned python scripts, which are referenced in the thesis. 
+ The folder results stores the raw results of the TimesNet forecast as .npy files. 
+ The folder checkpoints stores the TimesNet models after training, so that they can be accessed by the python files. 
+ The folder data contains the prepared input files for TimesNet.  
+ All folders are to be put into the Time-Series-Library git repository. 
