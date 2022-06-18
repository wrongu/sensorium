<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
[![hub](https://img.shields.io/badge/powered%20by-hub%20-ff5a1f.svg)](https://github.com/activeloopai/Hub)

# SENSORIUM 2022 Competition

![plot](figures/Fig1.png)
SENSORIUM is a competition on predicting large scale mouse primary visual cortex activity. We will provide large scale datasets of neuronal activity in the visual cortex of mice. Participants will train models on pairs of natural stimuli and recorded neuronal responses, and submit the predicted responses to a set of test images for which responses are withheld.

Join our challenge and compete for the best neural predictive model!

For more information about the competition, vist our [website](https://sensorium2022.net/).

# Important Dates
**June 20, 2022**: Start of the competition and data release. The data structure is similar to the data available at https://gin.g-node.org/cajal/Lurz2020.
<br>**Oct 15, 2022**: Submission deadline.
<br>**Oct 22, 2022**: Validation of all submitted scores completed. Rank 1-3 in both competition tracks are contacted to provide the code for their submission.
<br>**Nov 5, 2022**: Deadline for top-ranked entries to provide the code for their submission.
<br>**Nov 15, 2022**: Winners contacted to contribute to the competition summary write-up.

# Starter-kit

Below we provide a step-by-step guide for getting started with the competition.

## 1. Pre-requisites
- install [**docker**](https://docs.docker.com/get-docker/) and [**docker-compose**](https://docs.docker.com/compose/install/)
- install git
- clone the repo via `git clone https://github.com/sinzlab/sensorium.git`

## 2. Download neural data

You can download the data from [https://gin.g-node.org/cajal/Sensorium2022](https://gin.g-node.org/cajal/Sensorium2022) and place it in `sensorium/notebooks/data`.
**Note:** Downloading the files all at once as a directory does lead to unfortunate errors. Thus, all datastes have to be downloadede individually.

## 3. Run the example notebooks

### **Start Jupyterlab environment**
```
cd sensorium/
docker-compose run -d -p 10101:8888 jupyterlab
```

### **Competition example notebooks**
We provide four notebooks that illustrate the structure of our data, our baselines models, and how to make a submission to the competition.
<br>[**Notebook 1**](notebooks/legacy/1_inspect_data.ipynb): Inspecting the Data
<br>[**Notebook 2**](notebooks/legacy/2_model_training.ipynb): Re-train our Baseline Models
<br>[**Notebook 3**](notebooks/legacy/3_submission_and_evaluation.ipynb): Use our API to make a submission to our competition
<br>[**Notebook 4**](notebooks/legacy/4_cloud_based_data_demo.ipynb): A full submission in 4 easy steps using our cloud-based DataLoaders (using toy data)

### **Generic example notebooks**
Below we provide example notebooks for general usage of this repository for training and evaluating system identification models on publicly available datasets.
<br>[**Notebook 1**](notebooks/legacy/5_model_training_and_evaluation.ipynb): Train a model (here we used the baseline model in the competition) on publicly available dataset and evaluate on the test set