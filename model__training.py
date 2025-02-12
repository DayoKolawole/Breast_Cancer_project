import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import joblib

def run():
    print("✅ Running Model Training...")

    # Load dataset
    df = pd.read_csv("breast_cancer_data.csv")
    X = df.drop(columns=['target'])
    y = df['target']

    # Split data into training (80%) and testing (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train ANN model
    model = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', max_iter=500, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred = model.predict(X_test_scaled)

    # Evaluate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"🎯 Model Accuracy: {accuracy:.4f}")

    # Save the trained model and scaler
    joblib.dump((scaler, model), "model.pkl")
    print("✅ Model and Scaler saved as 'model.pkl'")

if __name__ == "__main__":
    run()
