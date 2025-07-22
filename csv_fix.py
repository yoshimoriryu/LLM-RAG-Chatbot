import pandas as pd

# Load CSV with semicolon delimiter and comma decimal
df = pd.read_csv("transactions_10k.csv", delimiter=";")

# Replace comma with dot in 'balance', then convert to float
df['amount'] = df['amount'].str.replace(",", ".", regex=False).astype(float)

# Save back to a new CSV (PostgreSQL-friendly)
df.to_csv("transactions_10k_fixed.csv", sep=";", index=False)
