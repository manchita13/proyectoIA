import numpy as np
import matplotlib.pyplot as plt

def plot_results(real_values, predicted_values):
    # Muestra la información a especificar
    print("Resultados de la Predicción:")
    print("-------------------------------------------------")
    print("Los valores reales corresponden a las observaciones originales que se usaron para entrenar el modelo. "
          "Son los valores que queremos predecir o estimar.")
    print("Los valores predichos son las estimaciones generadas por el modelo, basadas en los datos de entrada.")
    print("-------------------------------------------------")

    # Estadísticas simples para dar al usuario una idea del rendimiento
    diferencia_media = np.mean(np.abs(real_values - predicted_values))
    print(f"Promedio de la diferencia entre los valores reales y las predicciones: {diferencia_media:.2f}")

    # Si se desea, se puede agregar alguna medida de error, como el error cuadrático medio (RMSE)
    rmse = np.sqrt(np.mean((real_values - predicted_values) ** 2))
    print(f"Raíz del Error Cuadrático Medio (RMSE): {rmse:.2f}")

    # Comparación visual
    print("-------------------------------------------------")
    print("Gráfico de los valores reales frente a los predichos:")

    plt.figure(figsize=(10, 6))
    plt.plot(real_values, label='Valores Reales', marker='o', linestyle='-', color='blue')
    plt.plot(predicted_values, label='Valores Predichos', marker='x', linestyle='--', color='red')
    plt.title('Comparación de Valores Reales vs Predicciones')
    plt.xlabel('Índice')
    plt.ylabel('Valor')
    plt.legend()
    plt.grid(True)
    plt.show()

    print("-------------------------------------------------")
    print("El modelo ha completado la predicción con éxito. "
          "Asegúrese de revisar el gráfico para una comparación visual clara entre los valores reales y las predicciones.")