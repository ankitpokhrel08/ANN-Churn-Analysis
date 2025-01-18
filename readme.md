# Customer Churn Prediction using ANN

This project aims to predict customer churn using an Artificial Neural Network (ANN). The dataset used for this project is `Churn_Modelling.csv`.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Experiments](#experiments)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/Churn_ANN.git
    cd Churn_ANN
    ```

2. **Create and activate a virtual environment**:
    ```bash
    virtualenv -p python3 myenv
    source myenv/bin/activate
    ```

3. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Run the Jupyter Notebook**:
    ```bash
    jupyter notebook experiments.ipynb
    ```

2. **Run the Streamlit app**:
    ```bash
    streamlit run app.py
    ```

## Dataset

The dataset `Churn_Modelling.csv` contains the following columns:
- RowNumber
- CustomerId
- Surname
- CreditScore
- Geography
- Gender
- Age
- Tenure
- Balance
- NumOfProducts
- HasCrCard
- IsActiveMember
- EstimatedSalary
- Exited

## Experiments

All experiments and model training are documented in the `experiments.ipynb` Jupyter Notebook.

## Requirements

The project requires the following packages:
- numpy
- pandas
- scikit-learn
- tensorboard
- matplotlib
- streamlit
- tensorflow
- ipykernel

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License.
