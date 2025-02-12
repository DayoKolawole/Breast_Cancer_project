from sklearn.feature_selection import SelectKBest, f_classif
import pandas as pd

def run():
    print("✅ Running Feature Selection...")

    # Load dataset
    df = pd.read_csv("breast_cancer_data.csv")
    X = df.drop(columns=['target'])
    y = df['target']

    # Apply SelectKBest
    selector = SelectKBest(score_func=f_classif, k=10)
    X_selected = selector.fit_transform(X, y)

    # Get selected features
    selected_features = X.columns[selector.get_support()]
    print("✅ Selected Features:", selected_features)

if __name__ == "__main__":
    run()