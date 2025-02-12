import pandas as pd
from sklearn.datasets import load_breast_cancer

def run():
    print("✅ Running Data Preprocessing...")
    
        # Load dataset
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target

    # Save dataset
    df.to_csv("breast_cancer_data.csv", index=False)
    print("✅ Data saved to 'breast_cancer_data.csv'")

if __name__ == "__main__":
    run()





