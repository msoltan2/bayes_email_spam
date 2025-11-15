# Clasificator Bayes Multinomial pentru Email Spam

## 1. Descriere

Acest proiect implementeaza un clasificator Naive Bayes Multinomial pentru detectarea email-urilor spam. Codul prelucreaza un set de date textual, calculeaza probabilitati a priori si conditionate pentru fiecare cuvant si clasifica textele necunoscute pe baza probabilitatilor calculate.

Aceasta tema a clasificatorului a fost aleasa pentru a facilita procesul de gasire a datelor de antrenare si testare. Am pus accentul pe implementarea propriu-zisa a modelului Bayes.

---

## 2. Modelul matematic

Naive Bayes Multinomial se bazeaza pe teorema lui Bayes:

```
P(C | X) = P(C) * P(X | C) / P(X)
```

unde:
- C = clasa (spam sau ok)
- X = vector de caracteristici (cuvinte din email)

Se presupune independenta conditionata a cuvintelor:

```
P(X | C) = Product(P(x_i | C)) pentru i = 1..n
```

Cu smoothing Laplace pentru a evita probabilitatile zero:

```
P(w | C) = (count(w, C) + alpha) / (total_cuvinte_C + alpha * |V|)
```

unde:
- alpha = parametru de regularizare (1.0 implicit)
- |V| = dimensiunea vocabularului

Predicția clasei pentru un text nou se face alegand clasa cu probabilitatea maxima:

```
C_hat = argmax_C P(C) * Product(P(x_i | C))
```

---

## 3. Date de antrenare si testare

Pentru a antrena si ulterior testa acest model, a fost folosit un set public de date de pe Kaggle, care contine doua clase:

- spam - email-uri nedorite
- ok - email-uri normale (original ham)

Pentru acest set de date s-a obtinut accuratete de:
- `99.5737%`, n = 4457 (train)
- `98.3857%`, n = 1115 (test)


## 4. Structura codului

- clean(text, scoate_stopuri=True) - curata textul si returneaza lista de cuvinte, eliminand semnele de punctuatie si stop-words.
- truncate(text, max_len=100) - scurteaza textul pentru afisarea in tabel.
- NaivBayesMulti - clasa principala:
  - antreneaza(texts, etichete) - antreneaza modelul pe setul de date
  - scor_log_unic(text) - calculeaza scorul logaritmic pentru fiecare clasa
  - predict(texts) - returneaza clasa cu probabilitatea maxima
  - predict_prob(texts) - returneaza probabilitatile normalizate pentru fiecare clasa
  - score(texts, etichete) - calculeaza acuratetea modelului
  - cuvinte_indicative(clasa, topn) - returneaza top cuvinte caracteristice pentru o clasa
- imparte_date(texts, etichete, test=0.2, seed=42) - imparte datele in set de antrenament si test
- main() - fluxul principal:
  1. Incarca si curata datele
  2. Imparte datele in train/test
  3. Antreneaza modelul
  4. Evalueaza performanta si afiseaza acuratetea si matricea de confuzie
  5. Afiseaza exemple de predictie si cuvinte caracteristice

---

## 5. Instrucțiuni de utilizare

1. Instaleaza dependențele:

```
pip install pandas tabulate
```

2. Pune fisierul spam.csv in acelasi folder cu scriptul.

3. Ruleaza scriptul:

```
python bayer.py
```

4. Output-ul va include:
   - acuratetea modelului pe setul de antrenament si test
   - matricea de confuzie
   - exemple de texte cu predictiile si probabilitatile
   - top 15 cuvinte caracteristice pentru spam si ok

---

## 6. Exemple de utilizare

Exemplu de output:

```
=== Performanță ===
  set      acc    n
train 0.995737 4457
 test 0.983857 1115

=== Matrice confuzie ===
       ok  spam
ok    950    11
spam    7   147

=== Exemple ===
...

=== Spam top 15 ===
claim: 218.330
prize: 164.335
...

=== OK top 15 ===
lt: 107.768
gt: 107.768
...
```

---

## 7. Referinte bibliografice

1. Kaggle - Email Spam Detection Dataset: https://www.kaggle.com/datasets/shantanudhakadd/email-spam-detection-dataset-classification
2. Wikipedia - Naive Bayes classifier: https://en.wikipedia.org/wiki/Naive_Bayes_classifier
