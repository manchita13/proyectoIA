## Predicción del Impacto Económico de Decisiones Financieras usando IA
# Descripción
Este proyecto utiliza inteligencia artificial para ayudar a personas y pequeñas empresas a predecir el impacto de sus decisiones financieras en el tiempo. A través de un modelo de aprendizaje profundo (LSTM), la herramienta analiza datos financieros históricos (como ingresos, gastos e inversiones) para generar predicciones sobre el balance financiero futuro.
# Objetivos
Facilitar la toma de decisiones financieras mediante la proyección de balances futuros.
Simplificar el análisis de tendencias financieras para personas sin experiencia en economía o finanzas.
Visualizar de manera clara y accesible el impacto de decisiones económicas en el tiempo.

1. Clonar el Repositorio
```bash
git clone <URL_del_repositorio>
```
2. Instalar Dependencias
```bash
pip install -r requirements.txt
```
3. Ejecución del Proyecto
   1. Colocar los Datos en data/financial_data.csv
Asegúrate de que el archivo financial_data.csv contiene columnas como income, expenses, investments, y target (balance financiero final).
    2. Ejecutar el Proyecto
```bash
python src/main.py
```
Esto entrenará el modelo en los datos históricos, realizará predicciones y generará una visualización de los resultados.

## Explicación del Modelo de IA Utilizado
Modelo LSTM (Long Short-Term Memory)
Este proyecto usa un modelo LSTM, un tipo de red neuronal recurrente (RNN) que se adapta especialmente bien al análisis de series temporales. Los datos financieros, como ingresos y gastos, tienen patrones que se extienden a lo largo del tiempo, y el modelo LSTM permite capturar estos patrones para hacer predicciones más precisas sobre el balance financiero futuro.

## Ventajas de LSTM en el Proyecto
Memoria de Largo Plazo: El modelo es capaz de recordar patrones pasados que afectan eventos futuros.
Detección de Tendencias: Detecta tendencias generales en los datos, lo cual es útil para proyectar el impacto de decisiones financieras a lo largo del tiempo.
Manejo de Series Temporales: La estructura de LSTM permite trabajar con datos financieros secuenciales.
Visualización de Resultados
El proyecto incluye una visualización comparativa de valores reales y predichos. Este gráfico facilita la interpretación de los resultados, permitiendo observar si el modelo LSTM captura correctamente la tendencia de los datos financieros.

## Explicación del Gráfico
1. **Valores Reales:** 
Representan el balance financiero real obtenido de los datos históricos.
2. **Valores Predichos:** Muestran las proyecciones del modelo LSTM basadas en datos históricos.
3. **Interpretación:** Una coincidencia entre los valores reales y predichos indica una predicción confiable. Las desviaciones, por otro lado, pueden mostrar áreas donde mejorar el modelo o factores externos que el modelo no captura.
## Ejemplo de Uso
1. **Ingreso de Datos:** La herramienta toma datos históricos de ingresos, gastos e inversiones.
2. **Entrenamiento del Modelo:** El modelo LSTM se entrena para reconocer patrones.
3. **Predicciones y Visualización:** Se genera un gráfico de comparación entre los valores reales y predichos, lo que permite visualizar el impacto de decisiones financieras.


