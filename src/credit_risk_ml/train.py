from credit_risk_ml.data import load_data
from credit_risk_ml.features import preprocess
from credit_risk_ml.model import train_model

def main():
    print("Starting trainig pipeline")
    df = load_data('data/raw/credit_default_data.csv')
    df = preprocess(df)
    X = df.drop('default', axis=1)
    y = df['default']
    model = train_model(X,y)
    print("Model trained successfully!")

if __name__ == "__main__":
    main()