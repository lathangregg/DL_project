# DL_project
Predictive Model for NBA Data

nba-betting-prediction/
│── data/                        # Raw & processed datasets
│
│── notebooks/                    # Jupyter notebooks for EDA & experiments
│   ├── 01_eda.ipynb              # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb # Feature selection & creation
│   ├── 03_model_training.ipynb   # Model training/testing
│   ├── 04_evaluation.ipynb       # Performance analysis
│
│── src/                          # Source code
│   ├── data_loader.py            # Data fetching & preprocessing
│   ├── feature_engineering.py     # Feature transformation & encoding
│   ├── model.py                  # Deep learning model definitions
│   ├── train.py                  # Model training script
│   ├── evaluate.py               # Performance evaluation
│   ├── predict.py                # Model inference script
│
│── scripts/                      # Utility scripts for automation
│   ├── fetch_data.py             # Scraping or API data collection
│   ├── preprocess_data.py        # Data cleaning and formatting
│
│── models/                       # Saved models & checkpoints
│
│
│── tests/                        # Unit & integration tests
│
│── configs/                      # Configuration files
│   ├── config.yaml               # Model & training parameters
│
│── requirements.txt              # Required packages
│── README.md                     # Project overview & instructions
│── .gitignore                     # Files to exclude from version control

For all tasks related to a Deep Learning Model for NBA Predictions, please adhere to the following guidelines:
1. Provide python code
2. Cumulative Building: Each new section of code should be designed to
seamlessly integrate with the previous sections, building the code base
progressively. Please instruct me on exactly where to insert each new
block of code within the existing structure, avoiding repetition of
universal elements unless necessary for the task.
3. Clear comments. Provide comments that describe the purpose and function of code sections
4. Avoidance of Redundancy
Model Objective: Identify NBA betting market inefficiencies by predicting game outcomes with a calibrated probability model.
Model Type: Multilayer Perceptron (MLP); tested with various architectures (hidden layers, dimensions).
Input Features:
Team stats per game (e.g., points, rebounds)
Team records to date
Betting data (e.g., favorite team, bet percentages)
Preprocessing Steps:
Normalize stats using season-to-date mean values
Standardize all features
Calculate feature differences between teams to reduce dimensionality
Training Setup:
Train: 2012–2016 seasons
Validation: 2017 season
Test: 2018–2019
Final model retrained with train + validation before testing
Loss Function: Class-wise ECE to produce well-calibrated probabilities.
Evaluation Metrics:
Calibration accuracy
Return on Investment (ROI) via a betting simulation across full seasons