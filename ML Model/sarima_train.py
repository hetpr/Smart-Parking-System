import pandas as pd
import numpy as np
import pickle
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class ParkingOccupancyModel:
    """
    SARIMA-based model to forecast parking occupancy.
    This version supports daily repeating patterns (seasonality).
    """
    
    def __init__(self, order=(1,0,1), seasonal_order=(1,0,1,288)):
        """
        Initialize the SARIMA model configuration.
        
        order = (p,d,q) → Non-seasonal ARIMA settings
        seasonal_order = (P,D,Q,s) → Seasonal effect settings
        s = 288 because data is recorded every 5 minutes:
            24 hours * 60 min / 5 min = 288 samples per day
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.model_fit = None
        self.train_data = None
        self.test_data = None
        self.metrics = {}
        
    def load_and_preprocess(self, filepath, test_size=0.2):
        """Load dataset and split it into training and testing segments."""
        print("Loading dataset...")
        df = pd.read_csv(filepath, parse_dates=['timestamp'], index_col='timestamp')
        
        print(f"Dataset shape: {df.shape}")
        print(f"Time span: {df.index.min()} → {df.index.max()}")
        print("\nSummary:")
        print(df['total_occupied'].describe())
        
        # Handle missing values (if any)
        if df['total_occupied'].isnull().sum() > 0:
            df['total_occupied'].fillna(method='ffill', inplace=True)
        
        # Split into train/test keeping time order intact
        split_point = int(len(df) * (1 - test_size))
        self.train_data = df['total_occupied'].iloc[:split_point]
        self.test_data = df['total_occupied'].iloc[split_point:]
        
        print(f"\nTraining samples: {len(self.train_data)}")
        print(f"Testing samples:  {len(self.test_data)}")
        
        return self.train_data, self.test_data
    
    def plot_diagnostics(self):
        """Plot ACF & PACF to help understand correlation patterns."""
        fig, ax = plt.subplots(2, 2, figsize=(14, 8))
        
        plot_acf(self.train_data, lags=100, ax=ax[0,0])
        ax[0,0].set_title("ACF (Autocorrelation)")
        
        plot_pacf(self.train_data, lags=100, ax=ax[0,1])
        ax[0,1].set_title("PACF (Partial Autocorrelation)")
        
        ax[1,0].plot(self.train_data[:500])
        ax[1,0].set_title("Sample Time Series Section")
        ax[1,0].set_xlabel("Time")
        ax[1,0].set_ylabel("Occupied Slots")
        
        ax[1,1].hist(self.train_data, bins=30, edgecolor="black")
        ax[1,1].set_title("Distribution of Occupancy Values")
        
        plt.tight_layout()
        plt.savefig('sarima_diagnostics.png', dpi=300)
        print("Saved: sarima_diagnostics.png")
        plt.close()
    
    def train(self):
        """Train the SARIMA model using the training dataset."""
        print(f"\nTraining SARIMA model {self.order} x {self.seasonal_order}...")
        
        model = SARIMAX(
            self.train_data,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        self.model_fit = model.fit(disp=False)
        
        print("\nModel Training Summary:")
        print(self.model_fit.summary())
        print(f"\nAIC: {self.model_fit.aic:.2f}  |  BIC: {self.model_fit.bic:.2f}")
        
    def validate(self):
        """Check accuracy by comparing predictions against test data."""
        print("\nValidating model...")
        
        predictions = self.model_fit.predict(
            start=len(self.train_data),
            end=len(self.train_data) + len(self.test_data) - 1
        )
        
        # Calculate performance metrics
        mae = mean_absolute_error(self.test_data, predictions)
        rmse = np.sqrt(mean_squared_error(self.test_data, predictions))
        
        mask = self.test_data != 0
        mape = (np.abs((self.test_data[mask] - predictions[mask]) / self.test_data[mask]).mean() * 100
                if mask.sum() > 0 else float('inf'))
        
        r2 = r2_score(self.test_data, predictions)
        
        self.metrics = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'R2': r2}
        
        print(f"MAE:  {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAPE: {mape:.2f}%")
        print(f"R²:   {r2:.4f}")
        
        # Plot comparison
        plt.figure(figsize=(14, 6))
        plt.plot(self.test_data.index, self.test_data.values, label="Actual", linewidth=2)
        plt.plot(self.test_data.index, predictions.values, label="Predicted", linestyle="--", linewidth=2)
        plt.title("Actual vs Predicted Occupancy")
        plt.xlabel("Time")
        plt.ylabel("Occupied Slots")
        plt.legend()
        plt.grid(True)
        plt.savefig("validation_results.png", dpi=300)
        print("Saved: validation_results.png")
        plt.close()
        
        return predictions
    
    def save_model(self, filepath="sarima_model.pkl"):
        """Save trained model to file."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model_fit,
                'order': self.order,
                'seasonal_order': self.seasonal_order,
                'metrics': self.metrics,
                'train_end': self.train_data.index[-1]
            }, f)
        
        print(f"Model saved: {filepath}")


def main():
    model = ParkingOccupancyModel()
    
    model.load_and_preprocess("parking_data.csv")
    model.plot_diagnostics()
    model.train()
    model.validate()
    model.save_model("sarima_model.pkl")

    print("\n✅ Training Completed Successfully!\n")


if __name__ == "__main__":
    main()