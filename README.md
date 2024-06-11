# README

## Project Overview
This project, titled **2024 NLP Project: Semantic Analysis on Song Lyrics**, aims to develop a system that performs semantic analysis on song lyrics to classify them based on genre and emotion. By leveraging Natural Language Processing (NLP) techniques and machine learning models, the project seeks to create an automated solution for organizing and categorizing song lyrics, providing valuable insights into the linguistic and cultural aspects of music.

## Features
- **Feature Extraction**: Extracts lexical features (TF-IDF, word embeddings) and semantic features (sentiment analysis, figurative language detection) from song lyrics.
- **Model Training**: Implements and evaluates machine learning models, including Logistic Regression and Support Vector Machines (SVM), to classify lyrics by genre and emotion.
- **Evaluation**: Provides a comprehensive evaluation of model performance, including accuracy, F1-score, and other relevant metrics.


## Requirements
- Python 3.7+
- Required Python packages (listed in `requirements.txt`)

You can install the necessary packages using:

```bash
pip install -r requirements.txt
```


## Running the Code

To run the models, follow these steps:

### 1. Navigate to the Project Directory
Ensure that your current directory is set to `Final-Project`.

### 2. Select and Run a Model
Each model (SVM and Logistic Regression) has a `run_all.py` script. Uncomment the desired model (genre or emotion classification) in the respective script.

To run the SVM model:

```bash
python svm/run_all.py
```

To run the Logistic Regression model:

```bash
python Logistic_Regression/run_all.py
```

3. Classify Lyrics
After running the desired model, you can classify new lyrics using the corresponding predict_{model}.py script. Edit the script to include the lyrics you want to classify.

To classify lyrics using the SVM model:

```bash
python svm/predict_svm.py
```
To classify lyrics using the Logistic Regression model:

```bash
python logistic_regression/predict_logistic.py
```

Data, Software, and Ethics Policy

The dataset used in this project is sourced from publicly available repositories on Kaggle, ensuring compliance with copyright and licensing regulations. The software developed during this research is open-source and will be made available through appropriate channels (e.g., GitHub, open-access publications).

Ethical considerations are paramount, ensuring that data collection and analysis processes do not infringe on any privacy or intellectual property rights. The methods developed are intended solely for research and academic purposes, with no commercial or unauthorized applications.

## References

Papers and articles from the ACL Anthology: ACL Anthology

Scikit-learn documentation: Scikit-learn

Kaggle Competitions: Kaggle

Mahedero, Jose Martinez, et al. (2005). Natural language processing of lyrics. 475-478. DOI: 10.1145/1101149.1101255

## Contact

For any questions or further information, please contact:

Evan Doster
edoster@uoregon.edu
University of Oregon, CS 410