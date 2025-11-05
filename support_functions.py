import random
random.seed(0)

import numpy as np
import pandas as pd

from IPython.display import display

# graphic packages
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots
import seaborn as sns

import statsmodels.api as sm
from statsmodels.stats.outliers_influence \
    import variance_inflation_factor as VIF
from statsmodels.stats.anova import anova_lm

from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from ISLP import load_data

def prepare_data(data, predictors, response=None, add_intercept=True, interaction=None, normalize=False, encoding=None):
    """Prepare data from dataframe
    Parameters:
        data (pandas.DataFrame): Dataframe to extract data from
        predictors (list[str]): List of numerical predictors which are columns from data
        response (str): Response variable which is a column from data
        add_intercept (bool): If True, add a column of intercept to the data
        interaction (list[list[str]]): List of interaction variables which are column from data. The format is: ((col1, col2, etc.), (col3,col4,etc.), etc.)
        normalize (bool): If True, normalize the data
        encoding (list): List of categorical predictors to be encoded from data
    Returns:
        If response is None, return a tuple, the first element is the predictor data as a pandas.DataFrame, the second element is the response as a pandas.Series
        Else, return the predictor data as a pandas.DataFrame"""

    # extract predictor data
    X = data[predictors]

    if isinstance(X, pd.Series):
        X = X.to_frame()

    # add intercept
    if add_intercept:
        X.insert(loc=0, column='intercept', value=np.ones(X.shape[0]))

    # create interactive term
    if interaction is not None:
        for term in interaction:
            buf = data[term[0]]
            for col in term[1:]:
                buf = buf.multiply(data[col])
            X.insert(loc=len(X.columns), column=':'.join(term), value=buf)

    # normalize data
    if normalize:
        array = preprocessing.normalize(X, axis=0)
        X = pd.DataFrame(array, columns=X.columns)

    if encoding is not None:
        dummies = pd.get_dummies(data[encoding], prefix=encoding, drop_first=True)
        [X.insert(loc=len(X.columns), column=col, value=dummies[f'{col}'].astype(int)) for col in dummies.columns]

    # return
    if response is not None:
        return X, data[response]
    else:
        return X

def prepare_data_categorical(data, predictors, response=None, add_intercept=True):
    """
    Prepare data from dataframe
    Parameters:
        data (pd.DataFrame): Dataframe to extract data from
        predictors (list[str]): List of categorical predictors which are columns from data
        response (str): Response variable which is a column from data
        add_intercept (bool): If True, add a column of intercept to the data
    Returns:
        If response is None, return a tuple, the first element is the predictor data as a pandas.DataFrame, the second element is the response as a pandas.Series
        Else, return the predictor data as a pandas.DataFrame
    :param data:
    :param predictors:
    :param response:
    :param add_intercept:
    :return:
    """
    X = pd.get_dummies(data[predictors], prefix=predictors, drop_first=True, dtype=int)
    # [X.insert(loc=len(X.columns), column=col, value=dummies[f'{col}'].astype(int)) for col in dummies.columns]

    if add_intercept:
        X.insert(loc=0, column='intercept', value=np.ones(X.shape[0]))

    # return
    if response is not None:
        return X, data[response]
    else:
        return X

def plot_scatter_simple_linear(x, y, bi_estimate=None, bi_true=None, subplot=None):
    """
    Plot a scatter plot of y against x.
    Parameters:
        x (np.array): a 1D vector of x values
        y (np.array): a 1D vector of y values
        bi_estimate (list[float, float]): the estimated values of b0 and b1
        bi_true (list[float, float]): the true values of b0 and b1
        subplot (tuple [Figure, Axes]): the subplot number
    Return:
        Figure, Axes (tuple [Figure, Axes]): figure and axes objects
    """
    if subplot is None:
        fig, ax = plt.subplots(1,1, figsize=(8,8))
    else:
        fig, ax = subplot

    # scatter plot of x and y
    ax.scatter(x, y)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('y against x')

    # least squares line
    if bi_estimate is not None:
        y_estimate = bi_estimate[0] + bi_estimate[1] * x
        ax.plot(x, y_estimate, color='r', label='least square line')

    # population regression line
    if bi_true is not None:
        x_min = x.min(axis=0)
        x_max = x.max(axis=0)
        x_vals = np.linspace(x_min, x_max, x.shape[0])
        y_vals = bi_true[0] + bi_true[1] * x_vals
        ax.plot(x_vals, y_vals, color='b', linestyle='--', label='population regression line')

    ax.legend()

    return fig, ax

def plot_coefficient_interval(data, true_val=None, subplot=None):
    """
    Plot coefficient interval.
    Parameters:
        data (Dict): a dictionary with keys is 'bi model <name>', value is a tuple [coefficient, lower_bound, upper_bound]
        subplot (List(Figure, Axes)): figure and axes objects to plot
    :return:
    """
    if subplot is None:
        fig, ax = plt.subplots(1,1, figsize=(8,8))
    else:
        fig, ax = subplot

    # plot the vertical line of true bi value
    if true_val is not None:
        ax.axvline(true_val, color='r', linestyle='--', label=f'true value')

    # Extract the point estimates and confidence interval errors
    points = [val[0] for val in data.values()]
    # The error is the difference between the bounds and the point estimate
    lower_errors = [val[0] - val[1] for val in data.values()]
    upper_errors = [val[2] - val[0] for val in data.values()]
    asymmetric_error = [lower_errors, upper_errors]

    # The y-positions for the horizontal lines
    y_pos = np.arange(len(data), 0, -1)

    # Plot the horizontal confidence intervals
    ax.errorbar(points, y_pos, xerr=asymmetric_error, fmt='o', capsize=5)

    # Annotate data points with their values
    for i, txt in enumerate(points):
        ax.annotate(f'{txt:.3f}', (points[i], y_pos[i]), textcoords="offset points", xytext=(0,10), ha='center')

    # Set the y-axis labels to the point names
    ax.set_yticks(y_pos)
    ax.set_yticklabels(data.keys())

    # Add labels and title
    ax.set_xlabel("Value")
    ax.set_title("Comparison of Confidence Intervals")
    ax.legend()

    return fig, ax

def create_DataFrame_from_model_results(models, predictors):
    """ Create a pandas DataFrame holding the results of models
    Parameters:
        models (list[statsmodels.regression.linear_model.RegressionResultsWrapper]):
        predictors (list[str]): list of predictors
    Returns:
        DataFrame:
    """
    #Create a list to hold the summary results of each model
    results_list = []

    # Iterate through the models and predictors
    for i, model in enumerate(models):
        predictor_name = predictors[i]

        # Extract the required statistics
        r_squared = model.rsquared
        f_statistic = model.fvalue
        p_value_f = model.f_pvalue

        # Extract coefficients and related stats for the predictor
        coefficient = model.params[predictor_name]
        std_err = model.bse[predictor_name]
        t_statistic = model.tvalues[predictor_name]
        p_value_t = model.pvalues[predictor_name]

        # Extract the confidence interval for the predictor
        conf_int = model.conf_int().loc[predictor_name]

        # Append the results to the list
        results_list.append({
            'predictor': predictor_name,
            'R-squared': r_squared,
            'F-statistic': f_statistic,
            'p-value F-stats': p_value_f,
            'coefficient': coefficient,
            'standard error': std_err,
            't-statistic': t_statistic,
            'P>|t|': p_value_t,
            '0.025': conf_int[0],
            '0.975': conf_int[1]
        })

    # Create the pandas DataFrame
    results_df = pd.DataFrame(results_list).set_index('predictor')
    return results_df

def create_DataFrame_from_model_results_nonlinear(models, predictors):

    # Create an empty list to store the data for each row
    results_list = []

    # Iterate through each model and its corresponding base predictor name
    for i, model in enumerate(models):
        # Get the base predictor name for the current model
        base_predictor = predictors[i]

        # --- 1. Extract MODEL-LEVEL statistics ---
        # These are the same for all terms from this one model
        r_squared = model.rsquared
        f_statistic = model.fvalue
        p_value_f = model.f_pvalue

        # --- 2. Define the predictor terms for this model ---
        predictor_terms = [
            base_predictor,
            f'{base_predictor}_pow_2',
            f'{base_predictor}_pow_3'
        ]

        # --- 3. Loop through each term to extract its specific stats ---
        for term in predictor_terms:
            # Extract coefficient-level statistics for the specific term
            coefficient = model.params[term]
            std_err = model.bse[term]
            t_statistic = model.tvalues[term]
            p_value_t = model.pvalues[term]

            # Extract the confidence interval for this specific term
            conf_int = model.conf_int().loc[term]

            # Append all the information as a dictionary to our list
            # This dictionary represents one row in the final DataFrame
            results_list.append({
                'predictor': term,
                'R-squared': r_squared,
                'F-statistic': f_statistic,
                'p-value F-stats': p_value_f,
                'coefficient': coefficient,
                'standard error': std_err,
                't-statistic': t_statistic,
                'P>|t|': p_value_t,
                '0.025': conf_int[0],
                '0.975': conf_int[1]
            })

    # --- 4. Create the final DataFrame ---
    results_df = pd.DataFrame(results_list).set_index('predictor')
    return results_df

def confusion_matrix_binary(y_true, y_pred, positive_label, negative_label):
    """
    :param y_true: np.array true labels
    :param y_pred: np.array predicted labels
    :param positive_label: str positive label
    :param negative_label: str negative label
    :return: pd.DataFrame confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    df = pd.DataFrame(cm.T, columns=[negative_label, positive_label])
    df.index = pd.Index([negative_label, positive_label], name='Predicted')
    df.columns.name = 'Truth'
    return df