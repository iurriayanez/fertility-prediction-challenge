"""
This is an example script to train your model given the (cleaned) input dataset.

This script will not be run on the holdout data, 
but the resulting model model.joblib will be applied to the holdout data.

It is important to document your training steps here, including seed, 
number of folds, model, et cetera
"""

def train_save_model(cleaned_df, outcome_df):
    """
    Trains a model using the cleaned dataframe and saves the model to a file.

    Parameters:
    cleaned_df (pd.DataFrame): The cleaned data from clean_df function to be used for training the model.
    outcome_df (pd.DataFrame): The data with the outcome variable (e.g., from PreFer_train_outcome.csv or PreFer_fake_outcome.csv).
    """
    
    ## This script contains a bare minimum working example
    #random.seed(1) # not useful here because logistic regression deterministic

    # Combine cleaned_df and outcome_df
    model_df = pd.merge(cleaned_df, outcome_df, on="nomem_encr")

    # Filter cases for whom the outcome is not available
    model_df = model_df[~model_df['new_child'].isna()]  
    
    # Define preprocessor
    # Categorical and numerical variables
    numerical_columns_selector = selector(dtype_exclude=object)
    categorical_columns_selector = selector(dtype_include=object)
    
    numerical_columns = numerical_columns_selector(model_df[['gender', 'age', 'partner', 'nchild', 'fert_int', 'educ_level']])
    categorical_columns = categorical_columns_selector(model_df[['gender', 'age', 'partner', 'nchild', 'fert_int', 'educ_level']])

    categorical_preprocessor = OneHotEncoder(handle_unknown="ignore")
    numerical_preprocessor = StandardScaler()

    preprocessor = ColumnTransformer([
        ('one-hot-encoder', categorical_preprocessor, categorical_columns),
        ('standard_scaler', numerical_preprocessor, numerical_columns)])

    # Load the model
    model = make_pipeline(preprocessor, LogisticRegression(max_iter=500))
    
    # Fit the model
    model.fit(model_df[['gender', 'age', 'partner', 'nchild', 'fert_int', 'educ_level']], model_df['new_child'])

    # Save the model
    joblib.dump(model, "model.joblib")
