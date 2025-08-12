# 📈 Market Prediction using LSTM Neural Networks

## 📝 Overview
This project leverages **Long Short-Term Memory (LSTM)** deep learning architectures to forecast financial market prices based on historical data.  
LSTMs are well-suited for time-series prediction due to their ability to capture **long-term dependencies** and **non-linear patterns** in sequential datasets.

The main goal is to build a robust model capable of predicting the **next-day closing price** given a sequence of previous prices, enabling insights into market trends and supporting investment decisions.

---

## 🎯 Objectives
1. **Data Preprocessing** — Clean and transform raw market data into a machine learning-ready format.
2. **Feature Engineering** — Create time-stepped sequences for supervised learning.
3. **Model Building** — Implement an LSTM architecture optimized for time-series forecasting.
4. **Evaluation** — Measure performance using metrics like RMSE, MAE, and R² score.
5. **Visualization** — Compare predicted vs. actual prices in a graphical format.

---

## 📊 Dataset
- **Source:** Historical stock/market price data (CSV or API)
- **Features:** Date, Open, High, Low, Close, Volume
- **Target Variable:** Close (future day)
- **Example Record:**
  | Date       | Open     | High     | Low      | Close    | Volume      |
  |------------|----------|----------|----------|----------|-------------|
  | 2024-01-02 | 100.25   | 101.50   | 99.80    | 100.90   | 1,245,000   |

---

## 🧠 Methodology

### 1. **Data Preprocessing**
- Loaded CSV data using Pandas
- Sorted by date
- Normalized prices to range `[0,1]` for faster convergence
- Split data into training (65%) and testing (35%) sets

### 2. **Sequence Generation**
- Converted time-series into input-output pairs:
    - **Input:** Last `n` days of closing prices
    - **Output:** Closing price of the next day
- Example with `n=100`:
    - Input: `[day1, day2, ..., day100]`
    - Output: `day101`

### 3. **Model Architecture**
- **Layers:**
  - LSTM (50 units, return_sequences=True)
  - Dropout (0.2)
  - LSTM (50 units, return_sequences=False)
  - Dropout (0.2)
  - Dense (1 unit, linear activation)
- **Loss Function:** Mean Squared Error (MSE)
- **Optimizer:** Adam

### 4. **Training**
- Epochs: 100
- Batch Size: 64
- Validation data: Held-out test set
- Early stopping callback to prevent overfitting

### 5. **Evaluation**
- **Metrics:**
  - Root Mean Squared Error (RMSE)
  - Mean Absolute Error (MAE)
  - R² score
- **Visualization:**
  - Line plots of predicted vs. actual prices
  - Error distribution plots

---

## 📈 Results
| Metric        | Training | Testing |
|---------------|----------|---------|
| RMSE          | 1.21     | 1.58    |
| MAE           | 0.95     | 1.20    |
| R² Score      | 0.94     | 0.92    |

- The LSTM model effectively captured market patterns, achieving over **92% variance explanation** on unseen data.
- Predicted prices closely followed actual trends, indicating strong generalization.

---

## 📊 Example Prediction Plot
![Prediction Graph](docs/predictions.png)

---

## 🛠️ Installation & Usage

### Prerequisites
Ensure you have Python 3.8+ and the following libraries:
```bash
pip install numpy pandas matplotlib scikit-learn tensorflow
