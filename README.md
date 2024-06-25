# Description
This is the data pipeline repository for group 15 for CS4295-Release Engineering for Machine Learning Applications.

# Pipeline design
The pipeline is divided into six stages:

1. get data
2. data preperation 
3. train
4. predict
5. evalaute
6. release  

Each stage has a corresponding file in the `src` folder. Additionally, the `dvc.yaml` file specifies the exact dependencies and outputs for each stage. We chose to divide the pipeline into these stages because each has a well-defined input and output, adhering to standard ML practices.

# Prerequisites
 * Python 3.10 is required to run the code.
 * Create a conda environment :

   ```conda create --name remlapy310 python=3.10 ```

   ``` conda activate remlapy310 ```
  
   (alternatively , look at the `run.sh` file in the repository)

 * To install the required packages , we use [poetry](https://python-poetry.org/docs/). To install and run poetry , use the following commands.

   ```pip install poetry ```
 *  For DVC , please refer [official docs](https://dvc.org/doc)


# Setup
1. Once the virtual env, dvc and poetry is setup ,  run  ```poetry install``` to install the required dependecies.
2. To get the data from kaggle , sign up on [kaggle](https://www.kaggle.com) . Then go to the 'Account' tab of your user profile (https://www.kaggle.com/<username>/account) and select 'Create API Token'. This will trigger the download of kaggle.json, a file containing your API credentials. Place this file in the location ~/.kaggle/kaggle.json.
3. For Linters 
Install Pylint  : ```pip install pylint```
Install Flake   : ```pip install flake8 ``` 




