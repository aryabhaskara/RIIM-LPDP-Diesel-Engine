from bayes_opt import BayesianOptimization
import numpy as np
import xgboost as xgb

def predict(speed, load, bio_d, bio_bt):
    model1 = xgb.XGBRegressor()
    model1.load_model('model/XGB_TQ.model')    
    model2 = xgb.XGBRegressor()
    model2.load_model('model/XGB_SFC_MAE_final.model')
    model3 = xgb.XGBRegressor()
    model3.load_model('model/XGB_TE.model')
    
    input_data = np.array([[speed, load, bio_d, bio_bt]])
    
    torque = model1.predict(input_data)[0]  # Extract scalar value from 1D array
    sfc = model2.predict(input_data)[0]     # Extract scalar value from 1D array
    thermal_efficiency = model3.predict(input_data)[0]  # Extract scalar value from 1D array
    
    return torque, sfc, thermal_efficiency


def objective(speed, load, bio_d, bio_bt):
    torque, sfc, thermal_efficiency = predict(speed, load, bio_d, bio_bt)
    return torque + thermal_efficiency - sfc

pbounds = {
    'speed': (800, 1200),     # Adjust these bounds according to your problem
    'load': (1000, 4000),    # Speed bounds
    'bio_d': (0, 50),         # Bio_d bounds
    'bio_bt': (26, 60),       # Bio_bt bounds
}

optimizer = BayesianOptimization(
    f=objective,              # The function we are optimizing
    pbounds=pbounds,          # Bounds for the parameters
    random_state=1,           # Random state for reproducibility
    verbose=2                 # Verbosity level (0: silent, 1: minimal, 2: detailed)
)

optimizer.maximize(
    init_points=10,     
    n_iter=100,         
)

best_params = optimizer.max['params']
best_value = optimizer.max['target']

print(f"Best Parameters: {best_params}")
print(f"Best Objective Value: {best_value}")
