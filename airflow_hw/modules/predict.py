import os
import pandas as pd
import dill
import json
import glob



# Функция для выполнения предсказаний для всех файлов в папке
def predict():
    # Функция для загрузки модели из файла
    def load_model(model_filename):
        with open(model_filename, 'rb') as file:
            model = dill.load(file)
        return model

    # Функция для загрузки данных из JSON файла
    def load_json_data(json_filename):
        with open(json_filename, 'r') as file:
            data = json.load(file)
        return data

    # Функция для предсказания с использованием загруженной модели
    def predict_with_model(model, data):
        # Предполагая, что данные в JSON имеют такую же структуру, как в примере
        X = pd.DataFrame([data])
        predictions = model.predict(X)
        return data['id'], predictions

    # Папка с моделями
    models_folder = os.path.join(os.environ.get('PROJECT_PATH', '/Users/ninaromanova/airflow_hw'), 'data', 'models')
    # Поиск всех файлов с расширением pkl
    model_files = glob.glob(os.path.join(models_folder, '*.pkl'))
    # Выбор последнего созданного файла
    latest_model_file = max(model_files, key=os.path.getctime)

    # Загрузка модели
    model = load_model(latest_model_file)

    # Папка с тестовыми данными
    test_data_folder = os.path.join(os.environ.get('PROJECT_PATH', '/Users/ninaromanova/airflow_hw'), 'data', 'test')
    predictions = []

    # Предсказание для каждого файла в папке
    for filename in os.listdir(test_data_folder):
        if filename.endswith('.json'):
            json_filepath = os.path.join(test_data_folder, filename)
            data = load_json_data(json_filepath)
            id_, prediction = predict_with_model(model, data)
            predictions.append({'id': id_, 'predicted_price_category': prediction})

    # Создание DataFrame с предсказаниями
    predictions_df = pd.DataFrame(predictions, columns=['id', 'predicted_price_category'])

    # Папка для сохранения предсказаний
    predictions_folder = os.path.join(os.environ.get('PROJECT_PATH', '/Users/ninaromanova/airflow_hw'), 'data', 'predictions')
    if not os.path.exists(predictions_folder):
        os.makedirs(predictions_folder)

    # Сохранение предсказаний в CSV формате
    predictions_csv_path = os.path.join(predictions_folder, 'predictions.csv')
    predictions_df.to_csv(predictions_csv_path, index=False)

    print(f"Predictions saved to: {predictions_csv_path}")

    predictions_df = pd.read_csv(predictions_csv_path)

    # Вывод содержимого файла на печать
    print(predictions_df)

if __name__ == '__main__':
    predict()