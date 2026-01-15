#%%
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from tabulate import tabulate 

# ====== VERIFICAREA SI PRELUCRAREA SETULUI DE DATE ======

df = pd.read_csv('apple_quality.csv')

print("\n=== Dimensiune ===")
print(df.shape) #numarul de randuri si coloane
print("\n=== Informații despre setul de date ===")
print(df.info())
print("\n=== Valori duplicate ===")
print(df.duplicated().sum()) 
print("\n=== Valori lipsă ===")
if df.isna().sum().sum() == 0:
    print("Nu există valori lipsă în setul de date.")
else:   
    print(df.isna().sum())
# Eliminare rand cu valoari lipsa
print("\n=== Rânduri cu valori lipsă ===")
print("\n" , df[df.isnull().any(axis=1)])
df = df.drop(index=4000)
print("\n Dimensiunea setului de date dupa prelucrare :" ,df.shape)

df['Acidity'] = pd.to_numeric(df['Acidity']) # convertim coloana 'Acidity' de la object la numeric
df['Quality_Binary'] = df['Quality'].map({'good': 1, 'bad': 0}) # adaugam o coloana binara pentru calitate
stat = df.drop(columns=['Quality']).describe(include='all').T 
print("\n" + " Statistici Descriptive ".center(104, "="))
print(tabulate(stat, headers='keys', tablefmt='fancy_grid', numalign="right"))
# %%
