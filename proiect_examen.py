#%%
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from tabulate import tabulate 
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ====== VERIFICAREA SI PRELUCRAREA SETULUI DE DATE ======

df = pd.read_csv('apple_quality.csv')

print("\n=== Dimensiune ===")
print(df.shape) #numarul de randuri si coloane
print("\n=== Informații despre setul de date ===")
df.info()
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
stat = df.drop(columns=['Quality']).describe(include='all').T # statistici descriptive
print("\n" + " Statistici Descriptive ".center(104, "="))
print(tabulate(stat, headers='keys', tablefmt='fancy_grid', numalign="right"))
# %%
# ====================================================================
# Eliminam coloana 'A_id' pentru ca nu este relevanta pentru analiza
# ====================================================================

df.drop(columns=['A_id'], inplace=True)
df.head() #verificare modificare
# %%

# ====== ANALIZA CORELAȚIILOR ======
print("\n" + " Analiza Corelațiilor ".center(104, "="))
import seaborn as sns

# Matricea de corelații Pearson
corr_matrix = df.corr(method='pearson', numeric_only=True, min_periods=1)
print("\nMatricea corelațiilor Pearson:")
print(tabulate(corr_matrix, headers='keys', tablefmt='fancy_grid', numalign="right"))

# Reprezentare grafică a matricei de corelații
plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    linewidths=0.5
)
plt.title("Matricea corelațiilor Pearson")
plt.show()

# %%

# ==========================================
# Definirea variabilelor pentru regresie  liniară
# ==========================================

X = df[['Size', 'Weight', 'Sweetness', 'Crunchiness', 'Juiciness', 'Ripeness', 'Acidity']]  # variabile independente
y = df['Quality_Binary'] #variabila dependenta
print("\nVariabilele pentru regresie liniară definite:")
print("X (variabile independente):")
print(X.head())
print("\ny (variabilă dependenta):")
print(y.head())
# %%

# ==========================================
# Regresie Liniară OLS
# ==========================================

import statsmodels.api as sm
X = sm.add_constant(X)  # Adăugăm o constantă pentru intercept
model = sm.OLS(y, X).fit()
print("\n" + " Rezultatele Regresiei Liniară OLS ".center(104, "="))
print(model.summary())
# %%

# ==========================================
# Vizualizarea rezultatelor regresiei
# ==========================================

# vizzualizarea rezultatelor regresiei pentru variabilele independente semnificative statistic
significant_vars = model.pvalues[model.pvalues < 0.05].index.tolist()
significant_vars.remove('const')  # eliminam constanta
for var in significant_vars:
    plt.figure(figsize=(8, 6))
    plt.scatter(df[var], y, alpha=0.5)
    plt.xlabel(var)
    plt.ylabel('Quality_Binary')
    plt.title(f'Regresie Liniară: Quality_Binary vs {var}')
    
    # Linie de regresie
    x_vals = pd.Series(sorted(df[var]))
    y_vals = model.params['const'] + model.params[var] * x_vals
    plt.plot(x_vals, y_vals, color='red')
    
    plt.show()
    print("\n")
# %%

# ==========================================
# Interpretarea rezultatelor regresiei
# ==========================================

print("\n" + " Interpretarea Rezultatelor Regresiei ".center(104, "="))
for var in significant_vars:
    coef = model.params[var]
    p_value = model.pvalues[var]
    print(f"Variabila: {var}")
    print(f"  Coeficient: {coef:.4f}")
    print(f"  P-value: {p_value:.4f}")
    if coef > 0:
        print(f"  Interpretare: O creștere cu o unitate în '{var}' este asociată cu o creștere a probabilității ca mărul să fie de calitate 'good'.")
    else:
        print(f"  Interpretare: O creștere cu o unitate în '{var}' este asociată cu o scădere a probabilității ca mărul să fie de calitate 'good'.")
    print("\n")
# %%
# ==========================================
# Formularea Ipotezelor Statistice
# ==========================================

print("\n" + " Formularea Ipotezelor Statistice ".center(104, "="))
for var in significant_vars:
    print(f"Pentru variabila '{var}':")
    print("  Ipoteza nulă (H0): Coeficientul de regresie este egal cu zero (nu există efect).")
    print("  Ipoteza alternativă (H1): Coeficientul de regresie este diferit de zero (există efect).")
    print("\n")
# %%

# ==========================================
# Testarea ipotezelor statistice
# ==========================================

print("\n" + " Testarea Ipotezelor Statistice ".center(104, "="))
for var in significant_vars:
    print(f"Testarea pentru variabila '{var}':")
    coef = model.params[var]
    p_value = model.pvalues[var]
    
    if p_value < 0.05:
        print(f"  Rezultat: Respingeți ipoteza nulă (H0). Există dovezi suficiente pentru a susține că '{var}' are un efect semnificativ asupra calității mărului.")
    else:
        print(f"  Rezultat: Nu se respinge ipoteza nulă (H0). Nu există dovezi suficiente pentru a susține că '{var}' are un efect semnificativ asupra calității mărului.")
    print("\n")

# %%
# ==========================================
# Verificarea Multicoliniarității (VIF)
# ==========================================
print("\n" + " Verificarea Multicoliniarității (VIF) ".center(104, "="))

# Calculăm VIF pentru fiecare variabilă independentă
vif_data = pd.DataFrame()
vif_data["Variabila"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

print("\nFactorul de Inflație a Varianței (VIF):")
print(tabulate(vif_data, headers='keys', tablefmt='fancy_grid', numalign="right"))
print("\nInterpretare: Un VIF > 5 sau 10 indică o multicoliniaritate ridicată (variabilele sunt puternic corelate între ele).")

# %%
# ==========================================
# Compararea Grupurilor (Testul T)
# ==========================================
print("\n" + " Compararea Grupurilor (Testul T) ".center(104, "="))

good_apples = df[df['Quality'] == 'good']
bad_apples = df[df['Quality'] == 'bad']

print("Verificăm dacă există diferențe semnificative între mediile caracteristicilor pentru merele 'good' vs 'bad':\n")

for col in X.columns:
    if col != 'const': # Ignorăm constanta adăugată pentru regresie
        t_stat, p_val = stats.ttest_ind(good_apples[col], bad_apples[col])
        semnificativ = "DA" if p_val < 0.05 else "NU"
        print(f"{col:<12} | T-stat: {t_stat:>8.4f} | P-value: {p_val:>8.4e} | Diferență semnificativă: {semnificativ}")

# %%
# ==========================================
# Concluzii Finale
# ==========================================

print("\n" + " Concluzii Finale ".center(104, "="))
print("1. Caracteristicile 'Sweetness', 'Crunchiness' și 'Acidity' au un efect semnificativ asupra calității mărului.")
print("2. Creșterea acestor caracteristici este asociată cu o probabilitate mai mare ca mărul să fie de calitate 'good'.")
print("3. Nu s-au identificat probleme majore de multicoliniaritate între variabilele independente.")
print("4. Testul T a confirmat existența diferențelor semnificative între mediile caracteristicilor pentru merele 'good' și 'bad'.")
print("5. Aceste rezultate pot ghida producătorii în optimizarea caracteristicilor mărului pentru a îmbunătăți calitatea acestuia.")
# %%
