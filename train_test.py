# %%
from preprocessing import get_sup_df
from sklearn.model_selection import train_test_split

# %%
def prepare_X_y():
    """
    Establishes feature and target columns for the supervised dataset.

    Returns:
        X: A dataframe containing feature columns
        y: A series object containing target values
    """
    df = get_sup_df()

    X = df[['Age Group', 'Heart Disease', 'Asthma', 'Kidney Disease', 'Diabetes', 'Obesity', 'Population']]

    y = df['Rate of COVID Deaths Due to Conditions']

    return X, y

# %%
def get_train_test():
    """
    Splits dataset into training and testing data

    Args:
        X: Feature data to be split
        y: Target data to be split
    Returns:
        X_train: Feature data to be used in training
        X_test: Feature data to be used in testing
        y_train: Target data to be used in training
        y_test: Target data to be used in testing
    """
    X, y = prepare_X_y()

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    return X_train, X_test, y_train, y_test

# %%



