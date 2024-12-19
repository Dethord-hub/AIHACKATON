import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

data = pd.read_csv('dataset.csv', delimiter=';')

original_columns = data.columns.tolist()

print("Доступные столбцы:", data.columns.tolist())

categorical_mappings = {
    'teamId': {100: 0, 200: 1},
    'win': {'Win': 1, 'Fail': 0},
    'firstBlood': {True: 1, False: 0},
    'firstTower': {True: 1, False: 0},
    'firstInhibitor': {True: 1, False: 0},
    'firstBaron': {True: 1, False: 0},
    'firstDragon': {True: 1, False: 0},
    'firstRiftHerald': {True: 1, False: 0}
}

if 'dominionVictoryScore' in data.columns:
    data = data.drop(columns=['dominionVictoryScore'])
elif 'DominionVictoryScore' in data.columns:
    data = data.drop(columns=['DominionVictoryScore'])

for column, mapping in categorical_mappings.items():
    if column in data.columns:
        data[column] = data[column].map(mapping)

numeric_features = ['towerKills', 'inhibitorKills', 'baronKills',
                   'dragonKills', 'vilemawKills', 'riftHeraldKills']

scaler = MinMaxScaler()
data[numeric_features] = scaler.fit_transform(data[numeric_features])

feature_columns = [col for col in original_columns if col.lower() not in ['dominionvictoryscore', 'win']]

X = data[feature_columns]
y = data['win']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential()
model.add(Input(shape=(X_train.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=200,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

y_pred_proba = model.predict(X_test).ravel()

roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f'ROC AUC: {roc_auc}')

model.save('keras_model.h5')
