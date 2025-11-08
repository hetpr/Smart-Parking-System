import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class ParkingOccupancyPredictor:
    """
    This class loads the trained SARIMA model and uses it to
    generate future parking occupancy predictions. It also
    provides insights, visualizations, and recommendations.
    """
    
    def __init__(self, model_path="sarima_model.pkl"):
        self.model_path = model_path
        self.model_fit = None
        self.model_info = None
        self.historical_data = None
        
    def load_model(self):
        """Load the trained SARIMA model and stored metadata."""
        print("Loading trained SARIMA model...")
        with open(self.model_path, 'rb') as f:
            model_package = pickle.load(f)
        
        self.model_fit = model_package['model']
        self.model_info = {
            'order': model_package.get('order', 'Unknown'),
            'seasonal_order': model_package.get('seasonal_order', 'Unknown'),
            'metrics': model_package.get('metrics', {}),
            'train_end': model_package.get('train_end', 'Unknown')
        }
        
        print("Model loaded successfully.")
        print(f"Model used: SARIMA{self.model_info['order']} x {self.model_info['seasonal_order']}")
        
        # Display stored validation accuracy (useful for report/viva)
        if self.model_info['metrics']:
            print("\nModel Validation Metrics:")
            for metric, value in self.model_info['metrics'].items():
                if metric == 'MAPE':
                    print(f"  {metric}: {value:.2f}%")
                elif metric == 'R2':
                    print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value:.2f}")
    
    def load_historical_data(self, filepath="parking_data.csv"):
        """Load past occupancy data (required to generate timestamps for forecast)."""
        print(f"\nLoading historical data from {filepath}...")
        df = pd.read_csv(filepath, parse_dates=['timestamp'], index_col='timestamp')
        self.historical_data = df['total_occupied']
        print(f"Loaded {len(self.historical_data)} past data points.")
        return self.historical_data
    
    def predict(self, steps=24, confidence_level=0.95):
        """
        Predict future occupancy levels.

        steps = how many future time periods to forecast
        confidence_level = controls the width of the prediction interval
        """
        print(f"\nGenerating forecast for the next {steps} time steps...")

        # Generate forecast + uncertainty bounds
        forecast_result = self.model_fit.get_forecast(steps=steps)
        forecast = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int(alpha=1-confidence_level)

        # Remove negative predictions (occupancy cannot be negative)
        forecast = forecast.clip(lower=0)
        conf_int = conf_int.clip(lower=0)

        # Build time index for predictions
        last_timestamp = self.historical_data.index[-1]
        freq = pd.infer_freq(self.historical_data.index) or '5T'
        forecast_index = pd.date_range(start=last_timestamp + pd.Timedelta(freq), periods=steps, freq=freq)

        # Combine everything into one dataframe
        forecast_df = pd.DataFrame({
            'forecast': forecast.values,
            'lower_bound': conf_int.iloc[:, 0].values,
            'upper_bound': conf_int.iloc[:, 1].values
        }, index=forecast_index)

        return forecast_df
    
    def generate_insights(self, forecast_df):
        """Analyze forecast to identify trends like peak congestion time."""
        print("\n" + "="*70)
        print("FORECAST INSIGHTS")
        print("="*70)

        peak_time = forecast_df['forecast'].idxmax()
        peak_value = forecast_df['forecast'].max()

        min_time = forecast_df['forecast'].idxmin()
        min_value = forecast_df['forecast'].min()

        avg_value = forecast_df['forecast'].mean()

        trend_change = forecast_df['forecast'].iloc[-1] - forecast_df['forecast'].iloc[0]
        if abs(trend_change) < 1:
            trend = "STABLE"
        elif trend_change > 0:
            trend = "INCREASING"
        else:
            trend = "DECREASING"

        print(f"\nPeak Expected Occupancy: {peak_value:.1f} slots at {peak_time}")
        print(f"Lowest Expected Occupancy: {min_value:.1f} slots at {min_time}")
        print(f"Average Forecasted Occupancy: {avg_value:.1f} slots")
        print(f"Overall Trend: {trend}")

        return {
            'peak_time': peak_time,
            'peak_value': peak_value,
            'min_time': min_time,
            'min_value': min_value,
            'trend': trend
        }
    
    def plot_forecast(self, forecast_df, lookback_hours=12, save_path='sarima_forecast_plot.png'):
        """Plot historical data + forecast + confidence interval."""
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Upper plot: historical + forecast
        ax1 = axes[0]
        recent_data = self.historical_data[self.historical_data.index >= 
                                           self.historical_data.index[-1] - pd.Timedelta(hours=lookback_hours)]
        
        ax1.plot(recent_data.index, recent_data.values, label="Recent Historical Data", linewidth=2)
        ax1.plot(forecast_df.index, forecast_df['forecast'], label="Forecast", linestyle="--", linewidth=2.5)
        ax1.fill_between(forecast_df.index, forecast_df['lower_bound'], forecast_df['upper_bound'], alpha=0.3)
        ax1.set_title("Parking Forecast (Last 12 Hours + Future)")
        ax1.legend()
        ax1.grid(True)

        # Lower plot: forecast zoomed in
        ax2 = axes[1]
        ax2.plot(forecast_df.index, forecast_df['forecast'], linewidth=3)
        ax2.set_title("Forecast Detail View")
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"\nForecast graph saved as {save_path}")
        plt.close()
    
    def export_forecast(self, forecast_df, filepath='sarima_forecast_output.csv'):
        """Save forecast results for dashboard use."""
        forecast_df.to_csv(filepath)
        print(f"Forecast saved to {filepath}")
    
    def get_recommendation(self, forecast_df, capacity=20):
        """
        Suggest operational strategy based on forecasted demand.
        capacity = total available parking slots.
        """
        print("\n" + "="*70)
        print("PARKING MANAGEMENT RECOMMENDATION")
        print("="*70)

        avg_usage = forecast_df['forecast'].mean()
        peak_usage = forecast_df['forecast'].max()

        avg_percent = (avg_usage / capacity) * 100
        peak_percent = (peak_usage / capacity) * 100

        print(f"\nAverage Expected Usage: {avg_percent:.1f}%")
        print(f"Peak Expected Usage: {peak_percent:.1f}%")

        if peak_percent > 90:
            print("⚠ HIGH CONGESTION LIKELY → Prepare overflow parking.")
        elif peak_percent > 70:
            print("⚡ Moderate-High Usage → Monitor frequently.")
        elif peak_percent > 50:
            print("✓ Normal Usage → Standard operations are fine.")
        else:
            print("✓ Low Usage → Parking availability will be high.")


def main():
    predictor = ParkingOccupancyPredictor()
    
    predictor.load_model()
    predictor.load_historical_data("parking_data.csv")
    
    forecast_hours = 2
    steps = int((forecast_hours * 60) / 5)   # Convert hours → 5-min intervals
    
    forecast_df = predictor.predict(steps=steps)
    
    predictor.generate_insights(forecast_df)
    predictor.get_recommendation(forecast_df, capacity=20)
    predictor.plot_forecast(forecast_df)
    predictor.export_forecast(forecast_df)

    print("\n✅ Forecasting Completed.\n")


if __name__ == "__main__":
    main()