from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel, Field, ValidationError, validator
from typing import List, Union
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import r2_score, mean_squared_error as MSE
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from fastapi.responses import FileResponse, StreamingResponse
from io import BytesIO
import pandas as pd
import numpy as np
import re
import pickle
import csv


app = FastAPI()


class Item(BaseModel):
    name: str
    year: int = Field(..., gt=0)
    selling_price: int = Field(..., gt=0)
    km_driven: int = Field(..., gt=0)
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: Union[None, str]
    engine: Union[None, str]
    max_power: Union[None, str]
    torque: Union[None, str]
    seats:  Union[None, float, str]

    @validator('*')
    def empty_str_to_none(cls, v):
        if v == '':
            return None
        return v


class Items(BaseModel):
    object: List[Item]

    @validator('*')
    def empty_str_to_none(cls, v):
        if v == '':
            return None
        return v


# Загружаем информацию о модели 
with open('data.pickle', 'rb') as f:
        data = pickle.load(f)


def handle_torque(x: str):
    ''' На вход принимает строку параметра крутящий момент
    с единицами измерения. Переводит разные единицы измерения,
    отдает округленное значение параметра без единиц измерения, в формате float. '''

    if not pd.notnull(x):
        return None, None

    xm = re.sub(r'at', '@', x.replace(' ', ''))
    xm = re.sub(r'\d+-|~', '', xm)
    xm = re.sub(r',', '', xm)
    xm = xm.lower()

    torque_units = re.search(r'((?<=\(|\d)[a-z]+(?=@))', xm)
    torque_units = torque_units.group() if torque_units else 'nm'
    torque_to_nm = 1 if torque_units == 'nm' else 9.8067

    values = list(map(float, re.findall(r'[0-9]+\.*[0-9]*', xm)))

    if len(values) == 2:
        torque, rpm = values
        return (round(torque * torque_to_nm, 1), rpm)

    elif len(values) == 1:
        if values[0] < 1000:
            torque, rpm = values[0], None
            return (round(torque * torque_to_nm, 1), rpm)
        else:
            torque, rpm = None, values[0]
            return (torque, rpm)
    elif len(values) == 0:
        return None, None


def get_value(x: str):
    '''Фильрует все не-числовые значения'''
    if not pd.notnull(x):
        return None
    value = re.search(r'\d+.\d+|\d+', x)
    if not value:
        return None
    return float(value.group())


def get_car(x: str):
    '''Возвращает первое слово из столбца name'''
    value = re.findall(r'\w+', x)
    if not value:
        return None
    return value[0]


def preprocessing (df):
    """     
    Препроцессинг поступающих данных.
    Работа с пропусками, дополнительными признаками,
    скейлинг и подготовка к обработке моделью.

    input: pandas DataFrame с базовыми признаками
    return: pandas DataFrame с признаками, готовыми к предсказанию 
    """
    # Выбрасываем таргет
    df = df.drop('selling_price', axis=1)

    # Заполняем пустые строки NaN-ами
    df['mileage'] = df['mileage'].replace(r'^\s*$', np.nan, regex=True)
    df['engine'] = df['engine'].replace(r'^\s*$', np.nan, regex=True)
    df['max_power'] = df['max_power'].replace(r'^\s*$', np.nan, regex=True)
    df['torque'] = df['torque'].replace(r'^\s*$', np.nan, regex=True)
    df['seats'] = df['seats'].replace(r'^\s*$', np.nan, regex=True)

    # Уточняем формат для дальнейшей работы с csv
    df['year'] = df['year'].astype('Int64')
    df["km_driven"] = df["km_driven"].astype('Int64')
    df["seats"] = df["seats"].astype('float64')

    # Убираем еденицы измерения и разбираемся с параметром крутящий момент
    df["mileage"] = df["mileage"].apply(get_value).astype('float64')
    df["engine"] = df["engine"].apply(get_value).astype('float64')
    df["max_power"] = df["max_power"].apply(get_value).astype('float64')
    df[["torque", "max_torque_rpm"]] = pd.DataFrame(df["torque"].apply(handle_torque).to_list())

    # Добавляем индикацию пропусков в столбцах
    df['mileage_drops'] = df['mileage'].isna()
    df['engine_drops'] = df['engine'].isna()
    df['max_power_drops'] = df['max_power'].isna()
    df['torque_drops'] = df['torque'].isna()
    df['seats_drops'] = df['seats'].isna()
    df['max_torque_rpm_drops'] = df['max_torque_rpm'].isna()

    # Заполняем пропуски медианой 
    df['mileage'].fillna(data['mileage_median'], inplace=True)
    df['engine'].fillna(data['engine_median'], inplace=True)
    df['max_power'].fillna(data['max_power_median'], inplace=True)
    df['seats'].fillna(data['seats_median'], inplace=True)
    df['torque'].fillna(data['torque_median'], inplace=True)
    df['max_torque_rpm'].fillna(data['max_torque_rpm_median'], inplace=True)

    # Добавляем полиноминальные признаки
    poly = data['poly'].transform(df[['year', 'km_driven', 'mileage', 'engine', 
                                      'max_power', 'torque', 'max_torque_rpm']])
    poly = pd.DataFrame(poly, columns=data['poly_names'])

    # Добавляем производителя автомобиля
    df['brand'] = df['name'].apply(get_car)

    # Добавляем One-hot ecnoding
    encoded = data['ohe'].transform(df[["brand", "fuel", "seller_type", 
                                        "transmission", "owner", "seats"]]).toarray()
    encoded = pd.DataFrame(encoded, columns=data['ohe_names'])

    # Добавляем кастомные признаки

    # log(year)
    df['year_log'] = np.log(df['year'])
    # log(engine)
    df["engine_log"] = np.log(df["engine"])
    # log(km_driven)
    df["km_log"] = np.log(df["km_driven"])
    # log(torque)
    df["torque_log"] = np.log(df["torque"])
    # log(ma_torque_rpm)
    df["max_torque_rpm_log"] = np.log(df["max_torque_rpm"])
    # max_power / engine
    df["power_per_volume"] = df["max_power"] / df["engine"]
    # engine / torque
    df["volume_per_torque"] = df["engine"] / df["torque"]

    X_test = pd.concat([df, poly, encoded], axis=1).drop(columns=["name", "brand", 
                                                                  "fuel", "seller_type", 
                                                                  "transmission", "owner", "seats"])

    # Scale
    X_test_sc = pd.DataFrame(data['scaler'].transform(X_test), 
                             columns=X_test.columns)

    return X_test_sc

@app.post("/predict_item", summary='Get predicitions for one item')
def predict_item(item: Item) -> float:
    '''
    На вход получает json одного объекта.
    На выход отдает предсказание для данного объекта.

    input: json с описанием объекта
    return: файл с предсказаниями
    '''
    test_item = dict(item)
    df = pd.DataFrame.from_dict([test_item])

    return np.exp(data['best_model'].predict(preprocessing(df)))


@app.post("/predict_items", summary='Get predicitions for csv')
def predict_items(file: UploadFile):
    '''
    На вход получает файл csv, считывает его в датафрейм.
    Добавляет в датафрейм предсказания для каждого объекта
    отдельным столбцом.
    На выход отдает файл csv со столбцом предсказаний.

    input: загружаемый файл csv
    return: файл с предсказаниями
    '''

    content = file.file.read()
    buffer = BytesIO(content)
    df = pd.read_csv(buffer)
    buffer.close()
    file.close()

    output = df

    df['predict'] = pd.Series(np.exp(data['best_model'].predict(preprocessing(df))))
    output['predict'] = df['predict']
    output.to_csv('predictions.csv', index=False)
    response = FileResponse(path='predictions.csv', 
                            media_type='text/csv', filename='predictions.csv')

    return response
