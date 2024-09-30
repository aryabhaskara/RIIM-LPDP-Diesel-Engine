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
def run_optimization():
    pbounds = {
        'speed': (800, 1200),     
        'load': (1000, 4000),    
        'bio_d': (0, 50),         
        'bio_bt': (26, 60),       
    }

    optimizer = BayesianOptimization(
        f=objective,
        pbounds=pbounds,
        random_state=1,
        verbose=2
    )

    optimizer.maximize(init_points=10, n_iter=100)

    best_params = optimizer.max['params']
    best_value = optimizer.max['target']

    return best_params, best_value