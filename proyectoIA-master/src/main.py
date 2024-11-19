from data_loader import load_data
from predictor.predictor import FinancialPredictor
from visualizer.visualizer import plot_results

def main():
    # Cargar datos
    data = load_data('data/financial_data.csv')

    # Seleccionar variables
    X = data['income'].values  # Usaremos los ingresos como ejemplo de predictor
    y = data['target'].values  # Target es el balance financiero objetivo

    # Asegúrate de que X y y tengan la misma longitud
    print("Longitud de X:", len(X))
    print("Longitud de y:", len(y))

    # Inicializar y entrenar el modelo
    predictor = FinancialPredictor()
    predictor.train(X, y)

    # Realizar predicciones
    predictions = predictor.predict(X)

    # Visualizar resultados (tomando solo las últimas 28 observaciones de y)
    plot_results(y[-28:], predictions)  # Usar las últimas 28 predicciones

if __name__ == "__main__":
    main()
