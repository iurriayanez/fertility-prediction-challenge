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
    numerical_columns = ['age', 'nchild', 'hh_inc_2020']
    categorical_columns = ['gender', 'partner', 'fert_int', 'educ_level']

    categorical_preprocessor = Pipeline(
        steps = [("encoder", OneHotEncoder(handle_unknown="ignore"))]
    )
    numerical_preprocessor = Pipeline(
        steps = [("imputer", SimpleImputer(strategy = "mean")), ("scaler", StandardScaler())]
    )

    preprocessor = ColumnTransformer([
        ("cat", categorical_preprocessor, categorical_columns),
        ("num", numerical_preprocessor, numerical_columns)])

    # Load the model
    model = Pipeline(
        steps = [("preprocessor", preprocessor),
                 ("classifier", RandomForestClassifier(random_state = 42, 
                                                       class_weight = "balanced",
                                                       max_depth = 30,
                                                       max_features = 'sqrt',
                                                       min_samples_split = 10,
                                                       min_samples_leaf = 2,
                                                       bootstrap = True)
                 )
                ])
    
    # Fit the model
    model.fit(model_df[['gender', 'age', 'partner', 'nchild', 'fert_int', 'educ_level', 'hh_inc_2020', 'migration_bg']], model_df['new_child'])

    # Save the model
    joblib.dump(model, "model.joblib")