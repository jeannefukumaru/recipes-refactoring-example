"""
This module defines the following routines used by the 'transform' step of the regression recipe:

- ``transformer_fn``: Defines customizable logic for transforming input data before it is passed
  to the estimator during model inference.
"""

def transformer_fn():
    """
    Returns an *unfitted* transformer that defines ``fit()`` and ``transform()`` methods.
    The transformer's input and output signatures should be compatible with scikit-learn
    transformers.
    """
    # our old friend, onehotencoder and scaler
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import make_column_transformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.compose import make_column_selector
    import numpy as np

    # get the list of columns names of categorical variables
    cat_selector = make_column_selector(dtype_include=object)
    # get the list of column names of numerical variables
    num_selector = make_column_selector(dtype_include=np.number)


    cat_linear_processor = OneHotEncoder(handle_unknown="ignore")

    num_linear_processor = make_pipeline(
        StandardScaler(), SimpleImputer(strategy="mean", add_indicator=True)
    )

    linear_preprocessor = make_column_transformer(
        (num_linear_processor, num_selector), (cat_linear_processor, cat_selector)
    )

    return linear_preprocessor
