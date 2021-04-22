from model import SDN_Model
import numpy as np

md = SDN_Model('./config/md_config.json')
test = np.array(range(0,14)).reshape(1, 14)
print(f'test data : {test}')
pred = md.predict(test)
print(f'prediction : {pred}')