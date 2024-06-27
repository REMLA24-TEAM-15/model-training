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

# Run Instructions 

###### _If you want to download the data before running the pipeline run_:
 ```poetry run python -m src.data.get_data ```

1. To run the data pipeline use , ` dvc repro ` or `dvc exp run` 
2. The metrics and results are stored in the reports and models directories.
3. To run pylint: ```pylint <file/dir> > reports/pylintreport.txt```
4. To run flake8: ```flake8 <file/dir>```
5. To run tests : ```pytest --cov-report term --cov=src tests/ | tee reports/coverage_report.txt ```


# Model Performance

[![DVC-metrics](https://img.shields.io/badge/dynamic/json?style=flat-square&colorA=grey&colorB=99ff99&label=Accuracy&url=https://raw.githubusercontent.com/REMLA24-TEAM-15/model-training/main/reports/metrics/statistics.json&query=accuracy)](https://raw.githubusercontent.com/REMLA24-TEAM-15/model-training/main/reports/metrics/statistics.json) 

[![DVC-metrics](https://img.shields.io/badge/dynamic/json?style=flat-square&colorA=grey&colorB=99ff99&label=F1&url=https://raw.githubusercontent.com/REMLA24-TEAM-15/model-training/main/reports/metrics/statistics.json&query=f1)](https://raw.githubusercontent.com/REMLA24-TEAM-15/model-training/main/reports/metrics/statistics.json) 

[![DVC-metrics](https://img.shields.io/badge/dynamic/json?style=flat-square&colorA=grey&colorB=99ff99&label=ROC_AUC&url=https://raw.githubusercontent.com/REMLA24-TEAM-15/model-training/main/reports/metrics/statistics.json&query=roc_auc)](https://raw.githubusercontent.com/REMLA24-TEAM-15/model-training/main/reports/metrics/statistics.json) 


# Configuration Decisions

Below, we outline the project decisions.

## Remote
We configured a Google Drive folder as remote storage. It is accessible, [here](https://drive.google.com/drive/u/0/folders/1akOAvoKDwCRZWbWxBE_EmybK1hlHUDlL).
To modify the remote in DVC, use:
```
> dvc remote add --default myremote \
                           gdrive://<gdrive-endpoint>
> dvc remote modify myremote gdrive_acknowledge_abuse true
> dvc push
```

## Project structure
We used the standard [Cookiecutter](https://drivendata.github.io/cookiecutter-data-science/) to ease the creation of directories and configuration files.

## Package management
We used [Poetry](https://python-poetry.org/) as our dependency management tool. Poetry ensures that users with different environments use the same versions of dependencies. 

## Linters
We used [Pylint](https://pypi.org/project/pylint/) to check for errors in our codebase, enforce a coding standard in our project, and look for bad code smells [^1] https://docs.pylint.org/intro.html#what-is-pylint.

Additionaly, we use flake8 as a linter as it also includes complexity analysis.

### Pylint configuration
We use Pylint to check for code quality. The configuration file can be found in `pylintrc`. 
Pylint was configured to accept variable names starting with a capital, such as            Run,
           train_file,
           val_file,
           test_file,
           raw_x_train,
           raw_y_train,
           raw_x_test,
           raw_y_test,
           raw_x_val,
           raw_y_val, 
because these are common ML variable names. 



```
Report
======
185 statements analysed.

Statistics by type
------------------

+---------+-------+-----------+-----------+------------+---------+
|type     |number |old number |difference |%documented |%badname |
+=========+=======+===========+===========+============+=========+
|module   |7      |7          |=          |100.00      |0.00     |
+---------+-------+-----------+-----------+------------+---------+
|class    |0      |NC         |NC         |0           |0        |
+---------+-------+-----------+-----------+------------+---------+
|method   |0      |NC         |NC         |0           |0        |
+---------+-------+-----------+-----------+------------+---------+
|function |0      |NC         |NC         |0           |0        |
+---------+-------+-----------+-----------+------------+---------+



Raw metrics
-----------

+----------+-------+------+---------+-----------+
|type      |number |%     |previous |difference |
+==========+=======+======+=========+===========+
|code      |210    |66.25 |210      |=          |
+----------+-------+------+---------+-----------+
|docstring |18     |5.68  |18       |=          |
+----------+-------+------+---------+-----------+
|comment   |20     |6.31  |20       |=          |
+----------+-------+------+---------+-----------+
|empty     |69     |21.77 |69       |=          |
+----------+-------+------+---------+-----------+



Duplication
-----------

+-------------------------+------+---------+-----------+
|                         |now   |previous |difference |
+=========================+======+=========+===========+
|nb duplicated lines      |0     |0        |0          |
+-------------------------+------+---------+-----------+
|percent duplicated lines |0.000 |0.000    |=          |
+-------------------------+------+---------+-----------+



Messages by category
--------------------

+-----------+-------+---------+-----------+
|type       |number |previous |difference |
+===========+=======+=========+===========+
|convention |0      |0        |0          |
+-----------+-------+---------+-----------+
|refactor   |0      |0        |0          |
+-----------+-------+---------+-----------+
|warning    |0      |0        |0          |
+-----------+-------+---------+-----------+
|error      |0      |0        |0          |
+-----------+-------+---------+-----------+



Messages
--------

+-----------+------------+
|message id |occurrences |
+===========+============+




--------------------------------------------------------------------
Your code has been rated at 10.00/10 (previous run: 10.00/10, +0.00)


```

# Project Structure 

```
│   .dvcignore
│   .flake8
│   .gitignore
│   dvc.lock
│   dvc.yaml
│   params.yaml
│   poetry.lock
│   pylintrc
│   pyproject.toml
│   README.md
│   release.joblib
│   run.sh
│
├───.dvc
│       .gitignore
│       config
│
├───.github
│   └───workflows
│           pylint.yaml
│           pytest.yaml
│           realease.yaml
│           tag.yaml
│
├───.idea
│   │   .gitignore
│   │   misc.xml
│   │   model-training.iml
│   │   modules.xml
│   │   vcs.xml
│   │   workspace.xml
│   │
│   └───inspectionProfiles
│           profiles_settings.xml
│           Project_Default.xml
│
├───models
│   │   metrics.json
│   │   phishing_model.h5
│   │   predictions.joblib
│   │
│   └───plots
│       └───metrics
│           ├───eval
│           │       accuracy.tsv
│           │       loss.tsv
│           │
│           └───train
│                   accuracy.tsv
│                   loss.tsv
│
├───notebooks
│       phishing-detection-cnn.ipynb
│
├───reports
│   │   pylintreport.txt
│   │
│   └───metrics
│           .gitignore
│           statistics.json
│
├───src
│   │   release.py
│   │   __init__.py
│   │
│   ├───data
│   │       data_prep.py
│   │       get_data.py
│   │       __init__.py
│   │
│   └───models
│           evaluate.py
│           load_parameters.py
│           model.py
│           predict.py
│           train.py
│           __init__.py
│
└───tests
        data_slices_test.py
        features_cost_test.py
        feature_test.py
        mutamorphic_test.py
        non_determinism_test.py
        __init__.py
```
