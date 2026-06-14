import pandas as pd
import jiwer

df = pd.read_csv("results/baseline.csv")
cers = []
for _, row in df.iterrows():
    cer = jiwer.cer(row["reference"], row["prediction"])
    cers.append(round(cer, 4))

df["cer"] = cers
df.to_csv("results/baseline.csv", index=False)
print("✅ CER ajouté à baseline.csv")