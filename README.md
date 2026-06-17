# Proiect Examen AED: Analiza Calității Lotului de Mere

Acest repository conține proiectul realizat pentru examenul la disciplina **Analiza și Exploatarea Datelor (AED)** din cadrul **Universității Transilvania din Brașov (UNITBV)**. 

Proiectul are ca obiectiv principal analiza statistică, modelarea predictivă și evaluarea unui set de date care măsoară diverse caracteristici ale unui lot de mere, cu scopul de a determina calitatea acestora și de a clasifica fructele în categorii corespunzătoare.

## 📊 Obiectivele Proiectului

- **Analiza Exploratorie a Datelor (EDA):** Curățarea setului de date, vizualizarea distribuțiilor și analiza corelațiilor dintre proprietățile merelor (mărime, greutate, dulceață, crocant, aciditate etc.).
- **Modelare Predictivă:** Implementarea unor modele statistice și de Machine Learning (cum ar fi modele de regresie și clasificare) pentru evaluarea calității fructelor.
- **Evaluarea Performanței:** Generarea și interpretarea metricilor de clasificare, inclusiv utilizarea matricelor de confuzie pentru validarea acurateții modelelor.

## 📂 Structura Repository-ului

Conform structurii din proiect, fișierele sunt organizate astfel:

```text
├── apple_quality.csv         # Setul de date brut (baza de date) ce conține caracteristicile lotului de mere
├── proiect_examen.ipynb      # Notebook Jupyter care conține analiza pas cu pas, vizualizările și matricea de confuzie
├── proiect_examen.py         # Scriptul Python complet (versiunea finalizată a pipeline-ului de date)
├── analiza_rezultatelor.docx  # Document Word cu interpretarea detaliată a testelor statistice și a datelor
└── README.md                 # Documentația curentă a proiectului
```
🛠️ Tehnologii și Librării Utilizate

Proiectul a fost dezvoltat în Python, utilizând următoarele pachete standard pentru analiza datelor:

    Pandas & NumPy: Pentru manipularea seturilor de date și operații matematice.

    Matplotlib & Seaborn: Pentru generarea graficelor statistice și vizualizarea matricelor.

    Scikit-Learn (Sklearn): Pentru construirea modelelor de regresie/clasificare, calculul matricelor de confuzie și al metricilor de evaluare.

🚀 Cum se rulează proiectul
1. Clonarea repository-ului
Bash

git clone [https://github.com/ciufuu/examen_AED.git](https://github.com/ciufuu/examen_AED.git)
cd examen_AED

2. Instalarea dependențelor necesare

Asigură-te că ai librăriile necesare instalate executând:
Bash

pip install pandas numpy matplotlib seaborn scikit-learn jupyter

3. Rularea analizei

Poți rula scriptul Python direct din terminal:
Bash

python proiect_examen.py

Sau poți deschide notebook-ul interactiv pentru a vedea graficele și matricea de confuzie pas cu pas:
Bash

jupyter notebook proiect_examen.ipynb

📈 Rezultate și Concluzii

Analiza detaliată a testelor statistice efectuate, coeficienții modelelor de regresie și interpretarea matricelor de confuzie se regăsesc argumentate pe larg în fișierul analiza_rezultatelor.docx.

📝 Autori

    Ciufu Andrei (@ciufuu)

    Canuta Andrei (@andrwz1)
