import numpy as np
from scipy.optimize import minimize
import xgboost as xgb

# Example: Assuming 'predict' is your XGB model's prediction function
# It takes input parameters: load, speed, bio_bt, bio_d and returns [torque, sfc, thermal_efficiency]

def predict(load, speed, bio_d, bio_bt):
    model1 = xgb.XGBRegressor()
    model1.load_model('model/XGB_TQ.model')    
    #model2 = xgb.XGBRegressor()
    #model2.load_model('model/XGB_SFC_MAE_final.model')
    #model3 = xgb.XGBRegressor()
    #model3.load_model('model/XGB_TE.model')
    torque = model1.predict(np.array([[speed, load, bio_d, bio_bt]]))
    #sfc = model2.predict(np.array([[load, speed, bio_d, bio_bt]]))
    #thermal_efficiency = model3.predict(np.array([[load, speed, bio_d, bio_bt]]))
    #return torque, sfc, thermal_efficiency
    return torque

# Objective function: we aim to maximize torque and thermal efficiency, minimize sfc
def objective(params):
    speed, load, bio_d, bio_bt = params
    #torque, sfc, thermal_efficiency = predict(load, speed, bio_d, bio_bt)
    torque = predict(speed, load, bio_d, bio_bt)
    #return -torque + sfc - thermal_efficiency
    return -torque

# Initial guesses for load, speed, bio_bt, bio_d
initial_guess = [1.0, 1.0, 1.0, 1.0]  # Adjust based on typical values

# Bounds for each parameter (adjust these bounds according to your system)
bounds = [(800, 1200),   # load bounds
          (1000, 4000),  # speed bounds
          (0, 50),   # bio_d bounds
          (26, 60)]   # bio_bt bounds

# Run the optimizer
result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')

# Optimal parameters
optimal_load, optimal_speed, optimal_bio_bt, optimal_bio_d = result.x
print(f"Optimal load: {optimal_load}")
print(f"Optimal speed: {optimal_speed}")
print(f"Optimal bio_bt: {optimal_bio_bt}")
print(f"Optimal bio_d: {optimal_bio_d}")
