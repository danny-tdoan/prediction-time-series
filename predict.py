import itertools
import warnings
from statsmodels.api import tsa
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error as mse

warnings.filterwarnings("ignore")

def find_params_arima_grid_search(series,p,d,q):
    """This function derives the parameters p and q for ARIMA algorithm.

    Given the error margin is +/- 1.96*std, p and q take the value of the point where
    the acf and pacf cross the horizontal error margin the first time
    """
    pdq = list(itertools.product(p, d, q))
    best_params=pdq[0]
    smallest_aic=float("inf")

    for params in pdq:
        try:
            #the time series is already stationary
            model= tsa.statespace.SARIMAX(series.values, order=params)

            results = model.fit()
            if results.aic<smallest_aic:
                smallest_aic=results.aic
                best_params=params
        except Exception as e:
            continue
            
    return best_params

def find_params_arima_acf_pacf(shifted_series, nlags=20):
    """This function derives the parameters p and q for ARIMA algorithm.

    Given the error margin is +/- 1.96*std, p and q take the value of the point where
    the acf and pacf cross the horizontal error margin the first time
    """
    lag_acf = acf(shifted_series, nlags=nlags)
    lag_pacf = pacf(shifted_series, nlags=nlags, method='ols')

    # p is the point where the pacf cross the error margin the first time
    p = np.argmax(lag_acf < -1.96 / np.sqrt(len(shifted_series)))

    # q is the point where the acf cross the error margin the first time
    q = np.argmax(lag_pacf < -1.96 / np.sqrt(len(shifted_series)))

    return p, q


def predict_stock(stock_close_rtn, n_steps=5, plot=False):
    """Given the close returns of a stock (as a dataframe), predict the next n_step values"""
    diff_series = (stock_close_rtn - stock_close_rtn.shift()).dropna()
    p, q = find_params_arima(diff_series)

    model = ARIMA(stock_close_rtn.values, (p, q, 0)).fit(disp=plot)
    predicted = model.predict(end=n_steps)

    if plot:
        model.plot_predict(len(stock_close_rtn) - 10, len(stock_close_rtn) + n_steps)
    plt.axhline(y=0, linestyle='--', color='gray')

    return predicted


def evaluate_prediction_for_stock(stock_close_rtn, split=0.7):
    """Split the stock close returns into training (70%) and test set (30%)
    Train ARIMA on the training set, perform predict on the test set, and compute
    RMSE to evaluate the performance of the model

    return: the model, predicted values and RMSE
    """

    total_length = stock_close_rtn.shape[0]
    split_point = int(np.ceil(total_length * split))
    training_set = stock_close_rtn.iloc[:split_point]
    test_set = stock_close_rtn.iloc[split_point:]

    # get the p, q values
    diff_series = (stock_close_rtn - stock_close_rtn.shift()).dropna()
    #p, q = find_params_arima(diff_series)
    p, q = find_params_arima_acf_pacf(diff_series)

    model = ARIMA(training_set.values, (p, q, 0)).fit()

    predicted = model.predict(end=len(test_set))

    rmse = np.sqrt(mse(test_set.values, predicted))

    return predicted, rmse