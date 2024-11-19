import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential

class FinancialPredictor:
    def __init__(self, sequence_length=20):
        self.scaler_X = MinMaxScaler(feature_range=(0, 1))
        self.scaler_y = MinMaxScaler(feature_range=(0, 1))
        self.sequence_length = sequence_length
        self.model = self._build_model()

    def _build_model(self):
        """Construye el modelo LSTM."""
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(self.sequence_length, 1)))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def train(self, X, y):
        """Escala y prepara los datos para entrenamiento, luego entrena el modelo."""
        print("Contenido original de X:", X)
        print("Contenido original de y:", y)

        # Escala los datos
        X_scaled = self.scaler_X.fit_transform(X.reshape(-1, 1))
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1))

        # Genera secuencias para entrenamiento
        X_train = []
        y_train = []
        for i in range(self.sequence_length, len(X_scaled)):
            X_train.append(X_scaled[i-self.sequence_length:i, 0])
            y_train.append(y_scaled[i, 0])

        # Convierte a arrays y asegura la forma correcta
        X_train, y_train = np.array(X_train), np.array(y_train)
        print("Forma de X_train después de crear secuencias:", X_train.shape)

        # Ajusta la forma para que sea compatible con LSTM
        if X_train.shape[0] > 0:
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
            # Entrena el modelo
            self.model.fit(X_train, y_train, batch_size=1, epochs=5)
        else:
            print("No hay suficientes datos para crear secuencias de la longitud especificada.")

    def predict(self, X):
        """Realiza predicciones usando datos nuevos."""
        # Escala los datos de entrada para predicción
        X_scaled = self.scaler_X.transform(X.reshape(-1, 1))
        X_test = []

        for i in range(self.sequence_length, len(X_scaled)):
            X_test.append(X_scaled[i-self.sequence_length:i, 0])

        # Convierte a array y ajusta la forma para el modelo LSTM
        X_test = np.array(X_test)
        if X_test.shape[0] > 0:
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
            # Predice y desescala los resultados
            predictions = self.model.predict(X_test)
            return self.scaler_y.inverse_transform(predictions)
        else:
            print("No hay suficientes datos para realizar predicciones.")
            return []

def main():
    # Cargar el archivo CSV
    data = pd.read_csv("financial_data.csv")
    print("Datos cargados:")
    print(data.head())

    # Usar las columnas de 'income', 'expenses', 'investments' para las características (X) y 'target' para las predicciones (y)
    X = data[['income', 'expenses', 'investments']].values
    y = data['target'].values

    # Instancia del predictor y entrenamiento
    predictor = FinancialPredictor(sequence_length=20)
    predictor.train(X, y)

    # Predicciones
    predictions = predictor.predict(X)
    print("Predicciones realizadas:", predictions)

    # Puedes continuar con la visualización o análisis de las predicciones

if __name__ == "__main__":
    main()