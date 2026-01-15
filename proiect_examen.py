#%%
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from tabulate import tabulate 

# ====== STATISTICI DESCRIPTIVE ======

df = pd.read_csv('apple_quality.csv')

print("\n=== Dimensiune ===")
print(df.shape) #numarul de randuri si coloane
print("\n=== Informații despre setul de date ===")
print(df.info())
stat = df.describe().T 
print("\n" + " Statistici Descriptive ".center(104, "="))
print(tabulate(stat, headers='keys', tablefmt='fancy_grid', numalign="right"))
print("\n=== Valori duplicate ===")
print(df.duplicated().sum()) 
print("\n=== Valori lipsă ===")
if df.isna().sum().sum() == 0:
    print("Nu există valori lipsă în setul de date.")
else:   
    print(df.isna().sum())
# Eliminare rand cu valoari lipsa
print("\n" , df[df.isnull().any(axis=1)])
df = df.drop(index=4000)
print("\n" ,df.shape)
# %%
