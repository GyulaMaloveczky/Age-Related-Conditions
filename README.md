# Age-Related-Conditions
Predicting age related cognitive decline in elderly patients using LightGBM.
# Project Overview

This repository contains a detailed exploration and analysis of a dataset, highlighting key findings during the Exploratory Data Analysis (EDA) process and the subsequent modeling phase.

## Key Findings during EDA

1. **Class Imbalance in Target Variable**: 
    - During the EDA, a significant class imbalance was discovered in the target variable. This imbalance could potentially influence the accuracy and reliability of predictive models. Therefore, consideration was given to this aspect during the modeling phase.

2. **High Correlation Between 'Epsilon' and Target**:
    - A noteworthy high correlation was observed between the feature 'Epsilon' and the target variable. As correlation may imply a possible causative relationship or might just represent an underlying pattern, it was vital to consider this in the subsequent steps of data preprocessing and modeling.

3. **Data Merging**:
    - 'Epsilon' was merged with the rest of the dataset for a holistic analysis.

## Modeling

Various models were tried and their performance was gauged, keeping in mind the intricacies discovered during the EDA.

### Custom Evaluation Metric: Balanced Log Loss

During the modeling process, a custom evaluation metric named "Balanced Log Loss" was used. This metric provides a more nuanced performance evaluation, especially relevant given the class imbalance in the target variable.

### Hyperparameter Tuning

To ensure that each model operated at its optimal capability, hyperparameter tuning was conducted for each of the tried models. The Bayesian optimization was performed using an intermediate value for the parameters `kappa` and `xi` to strike a balance between exploration and exploitation. The optimization ran for a total of 5000 iterations.

### Best Model: LightGBM

After rigorous testing and tuning, the **LightGBM** model emerged as the top performer. The choice to proceed with LightGBM was based on its superior performance metrics compared to the other models.

## How to Use

1. Clone the repository:
    ```bash
    git clone <repository_link>
    ```

2. Navigate to the directory:
    ```bash
    cd <repository_name>
    ```

3. Follow instructions provided in the respective notebooks/scripts for data preprocessing, EDA, and modeling.

## Contribution

Feel free to fork the repository, raise issues, or submit Pull Requests if you think there's any way this analysis can be improved or if you have any questions.
