import pandas as pd
import shap
from sklearn.datasets import make_regression
from keras.models import Sequential
from keras.layers import Dense

def get_dataset():
  # Create sample data with sklearn make_regression function
  X, y = make_regression(n_samples=1000, n_features=10, n_informative=7, n_targets=5, random_state=0)

  # Convert the data into Pandas Dataframes for easier maniplution and keeping stored column names
  # Create feature column names
  feature_cols = ['feature_01', 'feature_02', 'feature_03', 'feature_04',
                  'feature_05', 'feature_06', 'feature_07', 'feature_08',
                  'feature_09', 'feature_10']

  df_features = pd.DataFrame(data = X, columns = feature_cols)

  # Create lable column names and dataframe
  label_cols = ['labels_01', 'labels_02', 'labels_03', 'labels_04', 'labels_05']

  df_labels = pd.DataFrame(data = y, columns = label_cols)

  return df_features, df_labels

def get_model(n_inputs, n_outputs):
    model = Sequential()
    model.add(Dense(32, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(n_outputs, kernel_initializer='he_uniform'))
    model.compile(loss='mae', optimizer='adam')
    return model

X, y = get_dataset()

# Get the number of inputs and outputs from the dataset
n_inputs, n_outputs = X.shape[1], y.shape[1]
model = get_model(n_inputs, n_outputs)
model.fit(X, y, verbose=0, epochs=100)
model.predict(X.iloc[0:1,:])
explainer = shap.KernelExplainer(model = model.predict, data = X.head(50), link = "identity")
X_idx = 0

shap_value_single = explainer.shap_values(X = X.iloc[X_idx:X_idx+1,:], nsamples = 100)
print(shap_value_single)