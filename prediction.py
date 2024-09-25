import joblib

def predict(data):
    model1 = joblib.load("ANN_Torque_FINAL.h5")
    #model2 = joblib.load("XGB_SFC_MAE_final.model")
    #model3 = joblib.load("XGB_TE.model")
    return model1.predict(data)
    #return sfc = model2.predict(data)
    #return te = model3.predict(data)