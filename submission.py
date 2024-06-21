"""
This is an example script to generate the outcome variable given the input dataset.

This script should be modified to prepare your own submission that predicts 
the outcome for the benchmark challenge by changing the clean_df and predict_outcomes function.

The predict_outcomes function takes a Pandas data frame. The return value must
be a data frame with two columns: nomem_encr and outcome. The nomem_encr column
should contain the nomem_encr column from the input data frame. The outcome
column should contain the predicted outcome for each nomem_encr. The outcome
should be 0 (no child) or 1 (having a child).

clean_df should be used to clean (preprocess) the data.

run.py can be used to test your submission.
"""

# List your libraries and modules here. Don't forget to update environment.yml!
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_selector as selector
import joblib

def clean_df(df, background_df=None):
    """
    Preprocess the input dataframe to feed the model.
    # If no cleaning is done (e.g. if all the cleaning is done in a pipeline) leave only the "return df" command

    Parameters:
    df (pd.DataFrame): The input dataframe containing the raw data (e.g., from PreFer_train_data.csv or PreFer_fake_data.csv).
    background (pd.DataFrame): Optional input dataframe containing background data (e.g., from PreFer_train_background_data.csv or PreFer_fake_background_data.csv).

    Returns:
    pd.DataFrame: The cleaned dataframe with only the necessary columns and processed variables.
    """

    # Select variables 
    # Gender
    gender_cols = "cf20m003"	

    # Age
    age_cols = "cf20m004"	

    # Has partner
    haspartner_cols = "cf20m024"

    # Has children
    haschild_cols = "cf20m454"	

    # Number of children
    nchild_cols = "cf20m455"	

    # Intention to have children
    intent_cols = "cf20m128" 

    # Educational level
    educ_cols = "cw20m005"

    keepcols = [
        "nomem_encr",
        "gender_bg",
        "age_bg",
        "cf20m024",
        "cf20m454",
        "cf20m455",
        "cf20m128",
        "oplcat_2020",
        "nettohh_f_2020",
        "migration_background_bg"
    ] 

    # Keeping data with variables selected
    df = df[keepcols]

    # Rename columns
    df.columns = ['nomem_encr', 'gender', 'age', 'partner', 'haschild', 'nchild', 'fert_int', 'educ_level', 'hh_inc_2020', 'migration_bg']

    # Correction of variables 
    # Correct number of children
    df.loc[df['haschild'] == 2, 'nchild'] = 0
    df = df.drop(columns=["haschild"])

    # Drop missing
    df = df.dropna()

    # Change column type
    df['gender'] = df['gender'].astype('object')
    df['partner'] = df['partner'].astype('object')
    df['fert_int'] = df['fert_int'].astype('object')
    df['educ_level'] = df['educ_level'].astype('object')
    df['migration_bg'] = df['migration_bg'].astype('object')
    df['age'] = df['age'].astype(int)
    df['nchild'] = df['nchild'].astype(int)
    df['hh_inc_2020'] = df['hh_inc_2020'].astype(int)

    return df



def predict_outcomes(df, background_df=None, model_path="model.joblib"):
    """Generate predictions using the saved model and the input dataframe.

    The predict_outcomes function accepts a Pandas DataFrame as an argument
    and returns a new DataFrame with two columns: nomem_encr and
    prediction. The nomem_encr column in the new DataFrame replicates the
    corresponding column from the input DataFrame. The prediction
    column contains predictions for each corresponding nomem_encr. Each
    prediction is represented as a binary value: '0' indicates that the
    individual did not have a child during 2021-2023, while '1' implies that
    they did.

    Parameters:
    df (pd.DataFrame): The input dataframe for which predictions are to be made.
    background_df (pd.DataFrame): The background dataframe for which predictions are to be made.
    model_path (str): The path to the saved model file (which is the output of training.py).

    Returns:
    pd.DataFrame: A dataframe containing the identifiers and their corresponding predictions.
    """

    ## This script contains a bare minimum working example
    if "nomem_encr" not in df.columns:
        print("The identifier variable 'nomem_encr' should be in the dataset")

    # Load the model
    model = joblib.load(model_path)

    # Preprocess the fake / holdout data
    df = clean_df(df, background_df)
    
    # Exclude the variable nomem_encr if this variable is NOT in your model
    vars_without_id = df.columns[df.columns != 'nomem_encr']
    
    # Generate predictions from model, should be 0 (no child) or 1 (had child)
    predictions = model.predict(df[vars_without_id])

    # Output file should be DataFrame with two columns, nomem_encr and predictions
    df_predict = pd.DataFrame(
        {"nomem_encr": df["nomem_encr"], "prediction": predictions}
    )

    # Return only dataset with predictions and identifier
    return df_predict
