import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as linear_model
import datetime
import random
import xgboost as xgb
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.model_selection import train_test_split, TimeSeriesSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from dateutil.relativedelta import relativedelta
from statsmodels.tsa.stattools import adfuller
from MMSC_BasketHedging import *
import warnings
from warnings import simplefilter

# Disable DeprecationWarnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

def calculate_covariance_matrix(pricing_date, returns, num_months=12):
    """
    Calculate a covariance matrix from returns.

    Parameters:
    - pricing_date (datetime): Pricing date.
    - returns (pandas.DataFrame): DataFrame containing the entire return series.
    - num_months (int): Number of months for covariance estimation.

    Returns:
    - cov (ndarray): Covariance matrix.
    """
    t_minus_n = pricing_date + relativedelta(months=-num_months)
    relevant_returns = returns[t_minus_n:pricing_date]
    cov = np.cov(relevant_returns.values, rowvar=False)
    return cov

def drop_highly_correlated_columns(data, cutoff=0.9):
    """
    Drop highly correlated columns from data.

    Parameters:
    - data (pandas.DataFrame): Input data.
    - cutoff (float): Correlation threshold for dropping columns.

    Returns:
    - data_filtered (pandas.DataFrame): Data with highly correlated columns removed.
    """
    corr_data = np.triu(np.corrcoef(data, rowvar=False), k=1)
    drop_indices = np.nonzero(corr_data > cutoff)
    data_filtered = data.drop(data.columns[drop_indices[1]], axis=1)
    return data_filtered

def select_random_stocks(df, num_stocks):
    """
    Select 'num_stocks' randomly from the given DataFrame and return a new DataFrame
    containing the cumulative sum of the time series data for the selected stocks,
    and the original DataFrame with those stocks removed.

    Parameters:
    - df (pandas.DataFrame): Input DataFrame with stock data.
    - num_stocks (int): Number of stocks to select.

    Returns:
    - new_df (pandas.DataFrame): New DataFrame with cumulative sum of selected stocks' time series.
    - remaining_df (pandas.DataFrame): Original DataFrame with selected stocks removed.
    """
    if num_stocks <= 0 or num_stocks > len(df.columns):
        raise ValueError("Invalid value for 'num_stocks'. Must be between 1 and the number of stocks in the DataFrame.")

    all_stocks = df.columns.tolist()
    random.seed(42)
    selected_stocks = random.sample(all_stocks, num_stocks)
    print("Selected stocks: " + str(selected_stocks))
    new_df = pd.Series(df[selected_stocks].sum(axis=1), name='Sum')
    remaining_stocks = [stock for stock in all_stocks if stock not in selected_stocks]
    remaining_df = df[remaining_stocks]
    return new_df, remaining_df

def select_best_stocks(X, y, limit):
    """
    Select the best stocks for hedging using Linear Regression.

    Parameters:
    - X (pandas.DataFrame): Feature data for stock selection.
    - y (pandas.Series): Target data for stock selection.
    - limit (int): Limit on the number of features to select.

    Returns:
    - weights (ndarray): Regression weights.
    - selected_features (pandas.DataFrame): DataFrame with the selected features.
    """
    n_best = 20
    val_fold = 5
    test_cv = 5

    anova_filter = SelectKBest(f_regression, k=n_best)
    lasso_cv = linear_model.LassoCV(cv=val_fold)
    feature_selection = RFE(lasso_cv, n_features_to_select=limit)
    linreg = linear_model.LinearRegression()

    estimators = [('anova_filter', anova_filter), ('fs', feature_selection), ('regressor', linreg)]
    pipe = Pipeline(estimators)

    tss = TimeSeriesSplit(test_cv)
    y_predict = pd.Series().reindex_like(y)
    for train_index, test_index in tss.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        pipe.fit(X_train, y_train)
        y_predict[test_index] = pipe.predict(X_test)
        s_train_fold = pipe.score(X_train, y_train)
        s_test_fold = pipe.score(X_test, y_test)

    s_train = s_train_fold.mean()
    s_test = s_test_fold.mean()
    pos = y_predict - y

    af_features = X_train.columns[pipe.named_steps['anova_filter'].get_support()]
    fs_features = af_features[pipe.named_steps['fs'].get_support()]
    weights = pipe.named_steps['regressor'].coef_
    print("std=%0.2f, r2_train=%.2f, r2_test=%.2f" % (pos.std(), s_train, s_test),
          "Selected features:", list(fs_features),
          "with weights:", ["{0:0.2f}".format(i) for i in weights])

    return weights, X[fs_features]

def xgboost_hedge(X, y):
    """
    Perform hedging using XGBoost.

    Parameters:
    - X (pandas.DataFrame): Feature data for hedging.
    - y (pandas.Series): Target data for hedging.

    Returns:
    - feature_importances (ndarray): Feature importances from XGBoost.
    """
    tscv = TimeSeriesSplit(n_splits=5)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }

    xgb_regressor = xgb.XGBRegressor()
    random_search = RandomizedSearchCV(
        xgb_regressor, param_distributions=param_grid, n_iter=10, scoring='neg_mean_squared_error',
        cv=tscv, n_jobs=-1, random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    best_params = random_search.best_params_
    feature_importances = best_model.feature_importances_
    stock_names = X.columns

    return feature_importances

def run_hedging():
    #read data
    df_Nikkei = pd.read_csv("nikkei_stock_data.csv",index_col=0)

    # Convert the index to datetime
    df_Nikkei.index = pd.to_datetime(df_Nikkei.index.str.split(' ').str[0])

    #to delete
    df_Nikkei = df_Nikkei.head(400)
    #--------

    # Check for null or NaN values for each column
    null_summary = df_Nikkei.isnull().sum()

    # Remove columns with more than 7 empty values
    columns_to_remove = null_summary[null_summary > 7].index
    df_Nikkei = df_Nikkei.drop(columns=columns_to_remove)

    # Fill empty values with the preceding value for the remaining columns
    df_Nikkei.fillna(method='ffill', inplace=True)

    #Drop highly correlated stocks
    df_Nikkei = drop_highly_correlated_columns(df_Nikkei, cutoff=0.95)
    print(df_Nikkei.shape)

    #Select randomly stocks to create ptf
    df_ptf, df_Nikkei = select_random_stocks(df_Nikkei, 5)

    retVal = df_Nikkei.copy()
    numOfMons = 12
    previous_date = df_Nikkei.index[0]
    list_df_returns = []
    for date in df_Nikkei.index:
        tminusN = date + relativedelta(months=-numOfMons)
        if tminusN < min(df_Nikkei.index):
            retVal.loc[date] = np.full((df_Nikkei.shape[1],), np.nan)
        elif previous_date.month != date.month:
            lists_weights = []
            tnextM = (date.replace(day=1) + datetime.timedelta(days=32)).replace(day=1)
            #Use last 12 months to computes weights
            df_Nikkei_previous12M = df_Nikkei[(df_Nikkei.index >= tminusN) & (df_Nikkei.index <= date)]
            df_ptf_previous12M = df_ptf[(df_ptf.index >= tminusN) & (df_ptf.index <= date)]

            #Select best stocks for hedging using LinearRegression
            linear_regression_weights, df_hedgeStocks = select_best_stocks(df_Nikkei_previous12M, df_ptf_previous12M, 5)
            list_stocks = df_hedgeStocks.columns
            lists_weights.append(['Linear',linear_regression_weights.ravel().tolist()])

            df_merged = pd.concat([df_ptf_previous12M, df_hedgeStocks], axis=1)
            df_merged_returns = df_merged.pct_change().dropna()
            cov = calculate_covariance_matrix(date, df_merged_returns, num_months=numOfMons)
            cov = np.matrix(cov)

            #XGBoost weights
            xgboost_weights = xgboost_hedge(df_hedgeStocks, df_ptf_previous12M)
            lists_weights.append(['XGBoost',xgboost_weights.ravel().tolist()])

            #MMSC weights
            MMSC_weights = getMMSC_weights(cov)
            MMSC_weights = MMSC_weights.ravel().tolist()[1:]
            MMSC_weights = [-x for x in MMSC_weights]
            lists_weights.append(['MMSC',MMSC_weights])

            #trim hedge stock and ptf for next moth to compute return
            df_hedgeStocks_nextM = df_Nikkei[(df_Nikkei.index >= previous_date) & (df_Nikkei.index < tnextM)][list_stocks]
            df_ptf_NextM = df_ptf[(df_ptf.index >= previous_date) & (df_ptf.index < tnextM)]

            df_strategies = pd.DataFrame()
            list_analysis_dollars = []
            # Iterate through the list of weights and calculate the equity returns for each strategy
            for list_weights in lists_weights:
                name = list_weights[0]
                weights = list_weights[1]

                #******* Dollars *******
                df_portfolio_hedging = df_hedgeStocks_nextM * weights

                sum_ptf = df_portfolio_hedging.sum(axis=1)
                hedging_strategy = df_ptf_NextM - sum_ptf

                hedging_strategy_return = hedging_strategy.pct_change().dropna()
                df_strategies[name] = hedging_strategy_return

                df_portfolio_hedging.columns = [name + "_" + 'stock_' + str(i) for i in range(len(df_portfolio_hedging.columns))]
                list_analysis_dollars.append(df_portfolio_hedging)

            list_df_returns.append(df_strategies)
        previous_date = date

    concatenated_df = pd.concat(list_df_returns)

    # Number of trading days in a year (adjust as needed)
    trading_days_per_year = 252

    # Initialize lists to store the results
    strategy_names = concatenated_df.columns
    std_list = []
    mean_list = []
    sharpe_list = []
    adf_pvalue_list = []

    # Calculate std, mean, Sharpe ratio, and ADF test for each strategy
    for strategy in strategy_names:
        returns = concatenated_df[strategy]

        # Annualized metrics
        std = np.std(returns) * np.sqrt(trading_days_per_year)
        mean = np.mean(returns)

        # Sharpe ratio without a risk-free rate (annualized)
        sharpe = mean / std * trading_days_per_year

        # ADF test to check for stationarity
        adf_result = adfuller(returns, autolag='AIC')
        adf_pvalue = adf_result[1]

        std_list.append(std * 100)
        mean_list.append(mean * 100)
        sharpe_list.append(sharpe)
        adf_pvalue_list.append(adf_pvalue)

    # Create a summary DataFrame for the results
    results_data = {
        'Strategy': strategy_names,
        'Daily Mean Return %': mean_list,
        'Annualized Vol %': std_list,
        'Sharpe Ratio': sharpe_list,
        'ADF Test P-Value': adf_pvalue_list
    }

    results_df = pd.DataFrame(results_data)

    # Display the results
    print(results_df)


if __name__=="__main__":
    run_hedging()