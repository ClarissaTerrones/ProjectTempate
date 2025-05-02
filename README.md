![](UTA-DataScience-Logo.png)

# Protein Function Prediction

* This repository explores the use of a Histogram-based Gradient Boosting Classifier to predict the functional class of proteins based on their sequence-derived features and physicochemical properties.
* https://www.kaggle.com/datasets/gallo33henrique/bioinformatics-protein-dataset-simulated?select=proteinas_test.csv

## Overview
This repository addresses the task of classifying proteins into one of five functional classes (Estrutural, Receptora, Enzima, Transporte, Outras) using a tabular dataset. The approach formulates this as a multi-class classification problem, leveraging various protein features. We focused on making a single algorithm work effectively as per the initial request. Our chosen model, a Histogram-based Gradient Boosting Classifier, achieved a validation accuracy of approximately 20%.

## Summary of Workdone


### Data

* Data:
    * Type: Tabular data (CSV format)
        * Input: Protein features such as molecular weight, isoelectric point, hydrophobicity, charge, amino acid proportions, and sequence length.
        * Output: Multi-class target column ("Classe"), with five possible protein functional classes.
    * Size:
        * Train: 16,000 rows × 10 columns (including the "Classe" target)
        * Test: 4,000 rows × 10 columns (including the "Classe" target for evaluation)
    * Instances (Train, Test, Validation Split): The initial training data was split into:
        * 60% training (9,600 samples)
        * 20% validation (3,200 samples)
        * 20% testing (3,200 samples)


#### Preprocessing / Clean up

* **Missing Values:** Checked for missing values in numerical features; none were found.
* **Sequence Length Validation:** Verified that the length of the protein sequence matched the provided sequence length feature.
* **Amino Acid Composition:** Engineered new numerical features based on the proportion of different types of amino acids in the protein sequence (Hydrophobic, Charged, Polar, Small, Aromatic, Proline Content).
* **Unnecessary Columns:** Removed 'ID_Proteína' and 'Sequência' columns after feature engineering.
* **Feature Scaling:** Applied `StandardScaler` to the numerical features to standardize their ranges.


#### Data Visualization


* **Bar Chart of Target Variable ('Classe'):** A bar chart showing the distribution of protein functional classes in the dataset.

    ![Bar Chart of Target Variable](Bar%20charts.png)

* **Comparison of Numerical Features Across Classes:** Histograms comparing the distributions of numerical features across different protein classes. The Kolmogorov-Smirnov (KS) test is used to quantify the difference between the distributions.

    ![Comparison of Numerical Features](Compares%20numerical%20features%20across%20classes.png)

### Performance Comparison

* **Confusion Matrix (Test Set):** A confusion matrix visualizing the model's performance on the test set, showing the distribution of predicted versus actual classes.

    ![Confusion Matrix (Test Set)](Model%20Performance.png)



### Problem Formulation

* Define:
    * Input: Numerical features derived from protein sequences and physicochemical properties.
    * Output: Prediction of the protein's functional class (one of five categories).
    * Models:
        * Histogram-based Gradient Boosting Classifier: Chosen for its efficiency and performance on tabular data.

### Training

* Describe the training:
    * Trained using scikit-learn in a standard Python environment.
    * Training time was minimal due to the dataset size and the efficiency of the algorithm.
    * Training curves (loss vs epoch for test/train): Not explicitly tracked as the `HistGradientBoostingClassifier` is not trained in epochs like neural networks. Performance was evaluated on the validation set after training.
    * How did you decide to stop training: Training stopped when the `fit` method of the classifier completed. For more advanced usage, one might monitor performance on a validation set during training and use early stopping.
    * Any difficulties? How did you resolve them? Initially encountered a `ValueError` due to non-numeric data. This was resolved by ensuring only numeric columns were used for training and prediction. Feature name mismatches between training and validation/test sets after selecting numeric columns were addressed by ensuring consistent column selection.


### Performance Comparison

* Clearly define the key performance metric(s):
    * Accuracy: Overall percentage of correctly classified proteins.
    * Classification Report: Includes precision, recall, and F1-score for each class.
    * Confusion Matrix: Shows the distribution of predicted vs. actual classes.
* Show/compare results in one table:


| Metric        | Value (Validation Set) |
|---------------|------------------------|
| Accuracy      | ~0.20                  |
| Macro Precision | ~0.20                  |
| Macro Recall    | ~0.20                  |
| Macro F1-Score  | ~0.20                  |

### Conclusions

* The initial implementation of the Histogram-based Gradient Boosting Classifier achieved an accuracy of approximately 20% on the validation set. This performance is close to random guessing given the five classes, indicating that further work is needed to build a more effective model. The confusion matrix shows a distribution of predictions across all classes, with no single class being consistently well-predicted.

### Future Work

* Investigate more advanced feature engineering techniques, potentially exploring protein sequence embeddings or other biological features.
* Address the class imbalance, although it appeared relatively minor in the initial analysis, by using techniques like oversampling or undersampling.

## How to reproduce results
*  To reproduce the results, follow the steps in a Python environment with the necessary libraries installed.

### Overview of files in repository

* This README.md: Provides an overview of the project.
  
### Software Setup
* pandas
* numpy
* scikit-learn (for data splitting, scaling, model, and metrics)
* matplotlib (for visualization)


### Data
* The `proteinas_train.csv` and `proteinas_test.csv` files should be in the same directory as the Python scripts.


### Training

* Trained using scikit-learn.

* Training time was minimal.

* Performance evaluated on the validation set.

* Training stopped after the fit method completed.

* Difficulties with data type and feature name mismatches were resolved.

#### Performance Evaluation

* Run the provided Python code to calculate and display metrics (accuracy, classification report, confusion matrix) on the validation set.


## Citations

Gallo, H. (2023). *Bioinformatics protein dataset simulated*. Kaggle. https://www.kaggle.com/datasets/gallo33henrique/bioinformatics-protein-dataset-simulated/code
