#%%
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from tabulate import tabulate 

# ====== STATISTICI DESCRIPTIVE SI TESTE STATISTICE PENTRU CALITATEA MARULUI ======

df = pd.read_csv('D:\\proiect_examen\\apple_quality.csv')

print("\n=== Dimensiune ===")
print(df.shape) # afiseaza numarul de randuri si coloane
print("\n=== Informații despre setul de date ===")
print(df.info())
stat = df.describe().T # Transpus (.T) arată adesea mai bine pentru multe coloane
print("\n" + " Statistici Descriptive ".center(104, "="))
print(tabulate(stat, headers='keys', tablefmt='fancy_grid', numalign="right"))
print("\n=== Valori duplicate ===")
print(df.duplicated().sum()) 
print("\n=== Valori lipsă ===")
if df.isna().sum().sum() == 0:
    print("Nu există valori lipsă în setul de date.")
else:   
    print(df.isna().sum())


# %%
