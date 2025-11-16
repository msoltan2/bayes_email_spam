import re
import random
import math
from collections import defaultdict, Counter
from typing import List
import pandas as pd
from tabulate import tabulate

seed = 48921481

stopuri = {"the", "a", "an", "and", "or", "but", "to", "from", "for", "of", "in", "on", "at", "with", "without", "is",
           "are", "was", "were", "be", "been", "being", "i", "you", "he", "she", "it", "we", "they", "them", "me", "my",
           "your", "our", "their", "this", "that", "these", "those", "not", "no", "yes", "do", "does", "did", "as",
           "by", "if", "so", "than", "then", "too", "very"}

def truncate(text, max_len=50):
    if len(text) > max_len:
        return text[:max_len] + "…"
    return text

def clean(text: str, scoate_stopuri: bool = True) -> List[str]:
    t = text.lower()
    t = re.sub("[^\w\s]", " ", t)
    tok = t.split()
    if scoate_stopuri:
        tok = [x for x in tok if x not in stopuri]
    return tok

class NaivBayesMulti:
    def __init__(self, alfa: float = 1.0):
        self.alfa = alfa
        self.vocab = set()
        self.nr_cuv_clasa = defaultdict(Counter)
        self.tot_cuv_clasa = defaultdict(int)
        self.nr_doc_clasa = defaultdict(int)
        self.priori = {}
        self.prob_cond = {}
        self.antrenat = False

    def antreneaza(self, texts: List[str], etichete: List[str]):
        bow = [Counter(clean(t)) for t in texts]
        n = len(etichete)
        self.vocab = set()
        self.nr_cuv_clasa = defaultdict(Counter)
        self.tot_cuv_clasa = defaultdict(int)
        self.nr_doc_clasa = defaultdict(int)

        for b, e in zip(bow, etichete):
            self.nr_doc_clasa[e] += 1
            for cuv, cnt in b.items():
                self.vocab.add(cuv)
                self.nr_cuv_clasa[e][cuv] += cnt
                self.tot_cuv_clasa[e] += cnt

        self.priori = {c: self.nr_doc_clasa[c] / n for c in self.nr_doc_clasa}
        V = len(self.vocab)
        self.prob_cond = {}

        for c in self.nr_cuv_clasa:
            total = self.tot_cuv_clasa[c]
            den = total + self.alfa * V
            tmp = {}
            for cuv in self.vocab:
                cnt = self.nr_cuv_clasa[c].get(cuv, 0)
                tmp[cuv] = (cnt + self.alfa) / den
            self.prob_cond[c] = tmp

        self.antrenat = True

    def scor_log_unic(self, text: str):
        if not self.antrenat:
            raise ValueError("Nu e antrenat")
        bow = Counter(clean(text))
        scoruri = {}
        V = len(self.vocab)

        for c in self.priori:
            s = math.log(self.priori[c]) if self.priori[c] > 0 else float("-inf")
            for cuv, cnt in bow.items():
                if cuv in self.vocab:
                    p = self.prob_cond[c][cuv]
                else:
                    den = self.tot_cuv_clasa[c] + self.alfa * V
                    p = self.alfa / den
                s += cnt * math.log(p)
            scoruri[c] = s
        return scoruri

    def predict(self, texts: List[str]) -> List[str]:
        return [max(self.scor_log_unic(t).items(), key=lambda x: x[1])[0] for t in texts]

    def predict_prob(self, texts: List[str]):
        rez = []
        for t in texts:
            scoruri = self.scor_log_unic(t)
            m = max(scoruri.values())
            exps = {c: math.exp(v - m) for c, v in scoruri.items()}
            s = sum(exps.values())
            rez.append({c: exps[c] / s for c in exps})
        return rez

    def score(self, texts: List[str], etichete: List[str]):
        p = self.predict(texts)
        return sum(1 for a, b in zip(p, etichete) if a == b) / len(etichete)

    def cuvinte_indicative(self, clasa: str, topn: int = 10):
        other = [c for c in self.priori if c != clasa]
        rez = []
        for cuv in self.vocab:
            p1 = self.prob_cond[clasa].get(cuv, 0)
            p2 = 0
            w = 0
            for c in other:
                p2 += self.priori[c] * self.prob_cond[c].get(cuv, 0)
                w += self.priori[c]
            if w > 0:
                p2 /= w
            else:
                p2 = 1e-12
            scor = (p1 + 1e-12) / (p2 + 1e-12)
            rez.append((cuv, scor))
        rez.sort(key=lambda x: x[1], reverse=True)
        return rez[:topn]

def imparte_date(texts, etichete, test=0.2, seed=42):
    ind = list(range(len(texts)))
    random.Random(seed).shuffle(ind)
    s = int(len(texts) * (1 - test))
    tr = ind[:s]
    ts = ind[s:]
    Xtr = [texts[i] for i in tr]
    ytr = [etichete[i] for i in tr]
    Xts = [texts[i] for i in ts]
    yts = [etichete[i] for i in ts]
    return Xtr, Xts, ytr, yts


def main():
    df = pd.read_csv("spam.csv", encoding="latin1")

    df = df.rename(columns={"v1": "label", "v2": "text"})

    df["label"] = df["label"].replace({"ham": "ok"})

    texts = df["text"].tolist()
    et = df["label"].tolist()

    Xtr, Xts, ytr, yts = imparte_date(texts, et, test=0.2, seed=seed)

    model = NaivBayesMulti(alfa=1.0)
    model.antreneaza(Xtr, ytr)

    acc_tr = model.score(Xtr, ytr)
    acc_ts = model.score(Xts, yts)

    pred = model.predict(Xts)
    clase = sorted(list(model.priori.keys()))
    cm = pd.DataFrame(0, index=clase, columns=clase)
    for p, y in zip(pred, yts):
        cm.loc[y, p] += 1

    indices = random.sample(range(len(Xts)), 10)
    exemple = [Xts[i] for i in indices]
    adevar = [yts[i] for i in indices]
    prob_ex = model.predict_prob(exemple)
    pred_ex = model.predict(exemple)

    df = pd.DataFrame({
        "text": [truncate(t, 100) for t in exemple],
        "adevar": adevar,
        "predictie": pred_ex,
        "prob": [f"spam: {x['spam']:.3f}, ok: {x['ok']:.3f}" for x in prob_ex]
    })

    top_spam = model.cuvinte_indicative("spam", topn=15)
    top_ok = model.cuvinte_indicative("ok", topn=15)

    print("\n=== Performanță ===")
    print(pd.DataFrame([
        {"set": "train", "acc": acc_tr, "n": len(ytr)},
        {"set": "test", "acc": acc_ts, "n": len(yts)}
    ]).to_string(index=False))

    print("\n=== Matrice confuzie ===")
    print(cm)

    print("\n=== Exemple ===")
    print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))

    print("\n=== Spam top 15 ===")
    for w, s in top_spam:
        print(f"{w}: {s:.3f}")

    print("\n=== OK top 15 ===")
    for w, s in top_ok:
        print(f"{w}: {s:.3f}")


if __name__ == "__main__":
    main()
