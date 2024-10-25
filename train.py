import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def calculer_cout(theta0, theta1, x, y):
    m = len(y)
    predictions = theta0 + theta1 * x
    cout = (1 / (2 * m)) * ((predictions - y) ** 2).sum()
    return cout

def descente_gradient(x, y, theta0, theta1, alpha, iterations):
    m = len(y)
    historique_cout = []
    for i in range(iterations):
        predictions = theta0 + theta1 * x
        erreur = predictions - y
        temp_theta0 = theta0 - alpha * (1/m) * erreur.sum()
        temp_theta1 = theta1 - alpha * (1/m) * (erreur * x).sum()
        
        if np.isnan(temp_theta0) or np.isnan(temp_theta1):
            print(f"NaN rencontré à l'itération {i}")
            break
        if np.isinf(temp_theta0) or np.isinf(temp_theta1):
            print(f"Infini rencontré à l'itération {i}")
            break
        
        theta0 = temp_theta0
        theta1 = temp_theta1
        cout = calculer_cout(theta0, theta1, x, y)
        historique_cout.append(cout)
        
    return theta0, theta1, historique_cout

data = pd.read_csv("data.csv")

if data[['km', 'price']].isnull().any().any():
    data = data.dropna(subset=['km', 'price'])

x = data['km']
y = data['price']

x_mean = x.mean()
x_std = x.std()
x_norm = (x - x_mean) / x_std

y_mean = y.mean()
y_std = y.std()
y_norm = (y - y_mean) / y_std

theta0 = 0
theta1 = 0
alpha = 0.01
iterations = 10000

theta0, theta1, historique_cout = descente_gradient(x_norm, y_norm, theta0, theta1, alpha, iterations)

plt.plot(range(len(historique_cout)), historique_cout)
plt.xlabel('Itérations')
plt.ylabel('Coût')
plt.title('Convergence de la descente de gradient')
plt.show()

theta1_denorm = theta1 * (y_std / x_std)
theta0_denorm = y_mean + y_std * theta0 - theta1_denorm * x_mean

print(f"Paramètres entraînés : theta0 = {theta0_denorm}, theta1 = {theta1_denorm}")

with open('theta.txt', 'w') as file:
    file.write(f'{theta0_denorm},{theta1_denorm}')


plt.scatter(data['km'], data['price'], label='Données réelles')

x_regression = data['km']
y_regression = theta0_denorm + theta1_denorm * x_regression
plt.plot(x_regression, y_regression, color='red', label='Modèle entraîné')

plt.xlabel('Kilométrage')
plt.ylabel('Prix')
plt.title('Régression linéaire - Prix vs Kilométrage')
plt.legend()
plt.show()
