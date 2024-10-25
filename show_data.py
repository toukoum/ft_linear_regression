import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("data.csv")


plt.scatter(data['price'], data['km'])
plt.xlabel('Price')
plt.ylabel('Km')
plt.title('Kilometrage en fonction du price')
plt.show()


