import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Cargar el archivo CSV desde la ruta proporcionada
file_path = 'C:/Users/eduar/Desktop/prediciendo/categorias.csv'
data = pd.read_csv(file_path)

# Limpiar los nombres de columna (eliminar espacios en blanco al principio o final si es necesario)
data.columns = data.columns.str.strip()

# Lista de columnas a eliminar
columns_to_drop = ['DEATH_EVENT', 'age', 'categoria_edad']

# Verificar si las columnas existen antes de intentar eliminarlas
existing_columns = [col for col in columns_to_drop if col in data.columns]

if existing_columns:
    X = data.drop(columns=existing_columns)
    y = data['age']  # Usamos 'age' como el vector objetivo

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crear una instancia del modelo de regresión lineal
    model = LinearRegression()

    # Ajustar el modelo con los datos de entrenamiento
    model.fit(X_train, y_train)

    # Predecir las edades sobre el conjunto de prueba
    y_pred = model.predict(X_test)

    # Calcular el error cuadrático medio (MSE) en el conjunto de prueba
    mse = mean_squared_error(y_test, y_pred)
    print(f"\nError cuadrático medio (MSE) en el conjunto de prueba: {mse}")

    # Comparar las edades predichas con las edades reales
    results = pd.DataFrame({'Edad Real': y_test, 'Edad Predicha': y_pred})
    print("\nComparación de Edades Reales vs. Edades Predichas:")
    print(results.head(10))  # Mostrar las primeras 10 filas para ver la comparación

    # Imprimir los coeficientes de las características
    print("\nCoeficientes de las características:")
    for feature, coef in zip(X.columns, model.coef_):
        print(f"{feature}: {coef}")

    # Imprimir el intercepto del modelo
    print("\nIntercepto del modelo:")
    print(model.intercept_)

else:
    print("\nUna o más columnas especificadas no existen en el DataFrame.")
