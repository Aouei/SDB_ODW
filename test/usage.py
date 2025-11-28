import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sensingpy import reader, plot
from sensingpy.image import Image
from sklearn.base import BaseEstimator

def predict_2d(image: Image, model : BaseEstimator) -> np.ndarray:
    ODW = np.zeros(shape = (image.height, image.width)).ravel()
    ODW[:] = np.nan

    X = pd.DataFrame( { band : image.select(band).ravel() for band in model.feature_names_in_ }).dropna()

    ODW[X.index] = model.predict(X)
    ODW = ODW.reshape((image.height, image.width))

    return ODW

image = reader.open(r'../data/formosa_2018.tif') 

ml_model = joblib.load('../models/ML__MLP.pkl')
cs_ml_model = joblib.load('../models/CS_ML__MLP.pkl')

image['ODW_ML'] = predict_2d(image, ml_model)
image['ODW_CS_ML'] = predict_2d(image, cs_ml_model)

fig, axs = plot.get_geofigure(image.crs, 1, 3, figsize = (6 * 3, 6))

plot.plot_rgb(image, 'Rrs_B4', 'Rrs_B3', 'Rrs_B2', axs[0], brightness = 20)
plot.plot_band(image, 'ODW_ML', axs[1])
plot.plot_band(image, 'ODW_CS_ML', axs[2])

axs[0].set_title('True Color')
axs[1].set_title('ODW ML')
axs[2].set_title('ODW CS + ML')

plt.show()