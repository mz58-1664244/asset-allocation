from WindPy import *
import numpy as np
from datetime import datetime
from scipy.optimize import minimize


def _asset_cov_matrix(assets="000905.SH,HSI.HI,SPX.GI,AU9999.SGE,H11008.CSI"):
    w.start()

    # extract needed data
    history_data = w.wsd(assets, "pct_chg", "ED-120TD", datetime.today().strftime("%Y-%m-%d"),
                         "")

    # turn NAN in the table into 0
    assets_data = np.nan_to_num(history_data.Data)

    # calculate the assets covariance matrix
    assets_cov = np.cov(assets_data, rowvar=True)
    return assets_cov


def _allocation_risk(weights, covariances):
    # We calculate the risk of the weights distribution
    portfolio_risk = np.sqrt(weights.dot(covariances.dot(weights.T)))

    # It returns the risk of the weights distribution
    return portfolio_risk


def _assets_risk_contribution_to_allocation_risk(weights, covariances):
    # We calculate the risk of the weights distribution
    portfolio_risk = _allocation_risk(weights, covariances)

    # We calculate the contribution of each asset to the risk of the weights
    # distribution
    assets_risk_contribution = weights * weights.dot(covariances) / portfolio_risk

    # It returns the contribution of each asset to the risk of the weights
    # distribution
    return assets_risk_contribution


def _risk_budget_objective_error(weights, args):
    # The covariance matrix occupies the first position in the variable
    covariances = args[0]

    # The desired contribution of each asset to the portfolio risk occupies the
    # second position
    assets_risk_budget = args[1]

    # We calculate the risk of the weights distribution
    portfolio_risk = _allocation_risk(weights, covariances)

    # We calculate the contribution of each asset to the risk of the weights
    # distribution
    assets_risk_contribution = _assets_risk_contribution_to_allocation_risk(weights, covariances)

    # We calculate the desired contribution of each asset to the risk of the
    # weights distribution
    assets_risk_target = portfolio_risk * assets_risk_budget

    # Error between the desired contribution and the calculated contribution of
    # each asset
    error = sum((assets_risk_contribution - assets_risk_target)**2)

    # It returns the calculated error
    return error


def _get_risk_parity_weights(covariances, assets_risk_budget, initial_weights):
    # Restrictions to consider in the optimisation: only long positions whose
    # sum equals 100%
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},
                   {'type': 'ineq', 'fun': lambda x: x})

    # Optimisation process in scipy
    optimize_result = minimize(fun=_risk_budget_objective_error,
                               x0=initial_weights,
                               args=[covariances, assets_risk_budget],
                               method='SLSQP',
                               constraints=constraints,
                               tol=1e-9,
                               options={'disp': False})

    # Recover the weights from the optimised object
    weights = optimize_result.x

    # It returns the optimised weights
    return weights


cov_matrix = _asset_cov_matrix()
risk_budget = np.array([1. / 5] * 5)
initial_weights = np.array([1. / 5] * 5)

asset_allocation = _get_risk_parity_weights(cov_matrix, risk_budget, initial_weights)



