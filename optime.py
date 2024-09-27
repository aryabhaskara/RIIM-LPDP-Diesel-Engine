from bayes_opt import BayesianOptimization
import numpy as np
import xgboost as xgb

# Assuming 'predict' is your model's prediction function 
# and it returns [torque, sfc, thermal_efficiency]

def predict(speed, load, bio_d, bio_bt):
    # Load the models
    model1 = xgb.XGBRegressor()
    model1.load_model('model/XGB_TQ.model')    
    model2 = xgb.XGBRegressor()
    model2.load_model('model/XGB_SFC_MAE_final.model')
    model3 = xgb.XGBRegressor()
    model3.load_model('model/XGB_TE.model')
    
    # Prepare the input in the correct format (2D array with one row)
    input_data = np.array([[speed, load, bio_d, bio_bt]])
    
    # Predict and extract scalar values
    torque = model1.predict(input_data)[0]  # Extract scalar value from 1D array
    sfc = model2.predict(input_data)[0]     # Extract scalar value from 1D array
    thermal_efficiency = model3.predict(input_data)[0]  # Extract scalar value from 1D array
    
    return torque, sfc, thermal_efficiency

# Objective function for Bayesian optimization
def objective(speed, load, bio_d, bio_bt):
    torque, sfc, thermal_efficiency = predict(speed, load, bio_d, bio_bt)
    
    # Maximize torque and thermal_efficiency, minimize sfc
    return torque + thermal_efficiency - sfc

# Define bounds for each parameter
pbounds = {
    'speed': (800, 1200),     # Adjust these bounds according to your problem
    'load': (1000, 4000),    # Speed bounds
    'bio_d': (0, 50),         # Bio_d bounds
    'bio_bt': (26, 60),       # Bio_bt bounds
}

# Create a Bayesian optimizer
optimizer = BayesianOptimization(
    f=objective,              # The function we are optimizing
    pbounds=pbounds,          # Bounds for the parameters
    random_state=1,           # Random state for reproducibility
    verbose=2                 # Verbosity level (0: silent, 1: minimal, 2: detailed)
)

# Run optimization
optimizer.maximize(
    init_points=10,     # Number of random starting points
    n_iter=100,         # Number of iterations to explore the parameter space
)

# Best result found
best_params = optimizer.max['params']
best_value = optimizer.max['target']

print(f"Best Parameters: {best_params}")
print(f"Best Objective Value: {best_value}")
