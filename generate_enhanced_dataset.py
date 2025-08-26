import numpy as np
import pandas as pd

def generate_large_synthetic_fusion_data(n_samples=5000, random_state=42):
    np.random.seed(random_state)

    # Keystroke features (in milliseconds)
    mean_hold_time = np.random.normal(150, 25, n_samples)
    std_hold_time = np.random.normal(15, 5, n_samples)
    mean_flight_time = np.random.normal(80, 15, n_samples)
    std_flight_time = np.random.normal(10, 3, n_samples)

    # Typing speed (chars per second), typical typing speed around 3-7 cps
    typing_speed = np.random.normal(5, 1, n_samples)

    # Error rate (typos per 100 chars), typical range 0-10%
    error_rate = np.clip(np.random.normal(3, 2, n_samples), 0, 15)

    # Sentiment features
    sentiment_polarity = np.random.uniform(-1, 1, n_samples)      # -1 negative, +1 positive
    sentiment_subjectivity = np.random.uniform(0, 1, n_samples)   # 0 objective, 1 subjective

    # Fraud probability model (logistic regression-like)
    logit = (
        0.02 * (mean_hold_time - 150) +
        0.03 * (mean_flight_time - 80) +
        0.05 * (error_rate - 3) -
        0.4 * sentiment_polarity +
        0.2 * sentiment_subjectivity -
        0.01 * (typing_speed - 5)
    )

    fraud_prob = 1 / (1 + np.exp(-logit))

    labels = np.random.binomial(1, fraud_prob)

    df = pd.DataFrame({
        'mean_hold_time': mean_hold_time,
        'std_hold_time': std_hold_time,
        'mean_flight_time': mean_flight_time,
        'std_flight_time': std_flight_time,
        'typing_speed': typing_speed,
        'error_rate': error_rate,
        'sentiment_polarity': sentiment_polarity,
        'sentiment_subjectivity': sentiment_subjectivity,
        'label': labels
    })

    return df

# Generate the dataset
df = generate_large_synthetic_fusion_data()

# Save to CSV
df.to_csv('enhanced_synthetic_fusion_fraud_data.csv', index=False)

print("Enhanced synthetic dataset created with shape:", df.shape)
print(df.head())

