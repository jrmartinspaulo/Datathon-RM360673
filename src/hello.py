import pandas as pd
import os

def main():
    csv_path = os.path.join("data", "entrevistas.csv")
    df = pd.read_csv(csv_path)

    print("\n=== Informações do dataset ===")
    print(df.info())

    print("\n=== Primeiras linhas ===")
    print(df.head())

if __name__ == "__main__":
    main()
