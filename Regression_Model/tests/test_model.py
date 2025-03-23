import sys
import os

# Add the parent directory to the system path so we can import model_utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from model_utils import log_transform, fit_model, get_vif, get_durbin_watson, mean_residuals

def test_model_metrics():
    df = pd.read_csv("Metrics_Trial_Median.csv")
    df = log_transform(df)
    model, X = fit_model(df)

    # Test 1: R-squared
    print(f"\nModel R-squared: {model.rsquared:.4f}")
    assert model.rsquared > 0.1, f"R-squared too low: {model.rsquared:.3f}"

    # Test 2: Mean residuals
    mean_resid = mean_residuals(model)
    print(f"Mean of residuals: {mean_resid:.6f}")
    assert abs(mean_resid) < 1e-4, f"Mean of residuals too far from zero: {mean_resid:.6f}"

    # Test 3: VIF
    vif = get_vif(X)
    print("VIF values (excluding 'const'):\n", vif)
    assert all(vif['VIF'] < 5), f"High multicollinearity detected:\n{vif}"

    # Test 4: Durbin-Watson
    dw = get_durbin_watson(model)
    print(f"Durbin-Watson statistic: {dw:.3f}")
    assert 1.5 < dw < 2.5, f"Durbin-Watson statistic abnormal: {dw:.3f}"
