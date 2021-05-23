from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from joblib import load, dump
import numpy as np
import json


class SDN_Model:

    def __init__(self, config_path):

        config = open(config_path, 'r')
        self.model_config = json.load(config)

        model_path = self.model_config['model_path']
        scaler_path = self.model_config['scaler]
        
        self.xgb_model = XGBClassifier()
        self.xgb_model.load_model(f'./config/{model_path}')
        self.scaler = load(scaler_path)

    def predict(self, test, mode='instant'):
        
        test = self.scaler.transform(test)
        train_pred_xgb = self.xgb_model.predict_proba(test)[:, 1]
        if mode == 'instant':
            return round(train_pred_xgb[0])
        else:
            return train_pred_xgb

    def getImportance(self):
    
        ft_cols = self.model_config['ft_cols']
        xgb_imp = self.xgb_model.feature_importances_
        ft_imp = [[ft_cols[i], xgb_imp[i]] for i in range(len(ft_cols))]
        ft_imp.sort(key=lambda x: -x[1])
        return ft_imp
