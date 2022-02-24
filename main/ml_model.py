import sys
import os
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.model_selection import train_test_split
import xgboost as xgb
import pickle


# El primer paso es obtener nuestro conjunto de datos de aprendizaje automático:
# 


def logic_layer(x):
	workpath = os.path.dirname(os.path.abspath(__file__)) #Devuelve la ruta en la que se encuentra su archivo .py


	# Carga el modelo entrenado en el entorno que se guarda en el mismo directorio llamado "xgboost_model.pkl"
	with open(os.path.join(workpath, 'xgboost_model.pkl'), 'rb') as f:
		clf = pickle.load(f)
	f.close()

	# Aquí preparamos nuestros datos en forma de dataframe, igual que en el modelo entrenado
	a = pd.DataFrame(columns = ['quarter', 'mode','purpose','year','dur_stay','market', 'Spend (£m)', '__target__'])
	a.loc[0]= x
	test = a


	# Aquí se realiza el reescalado y se toman scale_list, shift_list del conjunto de datos del modelo entrenado

# {'dur_stay': 'AVGSTD', 'mode': 'AVGSTD', 'Spend (£m)': 'AVGSTD', 'year': 'AVGSTD', 'quarter': 'AVGSTD', 'market': 'AVGSTD', 'purpose': 'AVGSTD'}
	scale_list = [5.379134276573772, 0.7736738167983019, 11.59681989634353, 1.1145033162651958, 1.0864534579734633, 15.398829026436987, 1.2086502909550274]
	shift_list = [8.212411193272438, 1.494562853414528, 5.049435762574525, 2015.4791938523997, 2.5104393214441063, 20.881542699724516, 2.411845730027548]
	# ambas listas tomadas del factor de reajuste del conjunto de datos de entrenamiento
	rescale_features = {u'dur_stay': u'AVGSTD', u'mode': u'AVGSTD', u'Spend (\xa3m)': u'AVGSTD', u'year': u'AVGSTD', u'quarter': u'AVGSTD', u'market': u'AVGSTD', u'purpose': u'AVGSTD'}
	for cnt, (feature_name, rescale_method) in enumerate(rescale_features.items()):
		test[feature_name] = (test[feature_name] - shift_list[cnt]).astype(np.float64) / scale_list[cnt]

	test_X = test.drop('__target__', axis=1)

	test_Y = np.array(test['__target__'])


	_predictions = clf.predict(test_X)
	predictions = pd.Series(data=_predictions, index=test_X.index, name='predicted_value')
	return int(round(predictions * 1000))



# Esto se utiliza para probar en el entorno local como x, y, z tres conjuntos de datos se dan como ejemplo
if __name__ == "__main__":
	x = [-1.380899, -0.638146, 1.315741, -1.325480, -0.231486, 0.081364, -0.319305, 0.380]
	y = [3, 3, 2, 2016, 7, 5, 0.897155, 3.556813]
	z = [3,	1,	1,	2016,	15,	11	,8.235992,	6.062523]
	print(logic_layer(y))
	input()

