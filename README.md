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
