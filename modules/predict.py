import dill
import os
import json
import pandas as pd

project_path = os.environ.get('PROJECT_PATH', '.')
TEST_DIR = os.path.join(project_path, 'data/test')
MODELS_DIR = os.path.join(project_path, 'data/models')
PREDICTIONS_DIR = os.path.join(project_path, 'data/predictions')

def load_model():
    file_path = os.path.join(MODELS_DIR, os.listdir(MODELS_DIR)[0])
    with open(file_path, 'rb') as file:
        return dill.load(file)

def get_test_data():
    rows = []
    for file_name in os.listdir(TEST_DIR):
        file_path = os.path.join(TEST_DIR, file_name)
        with open(file_path, 'r') as file:
            row = json.load(file)
            rows.append(row)
    return rows

def predict():
    model = load_model()
    df = pd.DataFrame.from_dict(get_test_data())
    price_category = model.predict(df)
    df.loc[:, 'price_category'] = price_category
    df_predict = df[['id', 'price_category']]
    df_predict.to_csv(os.path.join(PREDICTIONS_DIR, 'prediction.csv'))


if __name__ == '__main__':
    predict()
