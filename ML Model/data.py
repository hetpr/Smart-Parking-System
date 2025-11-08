import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_realistic_parking_data(
    total_slots=20,
    days=14,
    interval_minutes=5,
    seed=42
):
    """
    Generate realistic parking occupancy data with daily patterns
    
    Parameters:
    -----------
    total_slots : int
        Total parking capacity
    days : int
        Number of days to generate
    interval_minutes : int
        Time interval between observations
    seed : int
        Random seed for reproducibility
    """
    np.random.seed(seed)
    
    # Calculate total number of records
    records_per_day = (24 * 60) // interval_minutes
    total_records = records_per_day * days
    
    # Generate timestamps
    start_date = datetime(2025, 1, 1, 0, 0, 0)
    timestamps = [start_date + timedelta(minutes=i*interval_minutes) 
                  for i in range(total_records)]
    
    occupancy_data = []
    
    for ts in timestamps:
        hour = ts.hour
        day_of_week = ts.weekday()  # 0=Monday, 6=Sunday
        
        # Base occupancy patterns by hour (percentage of capacity)
        if 0 <= hour < 6:  # Late night/early morning
            base_occupancy = 0.15
        elif 6 <= hour < 8:  # Early morning arrival
            base_occupancy = 0.35 + (hour - 6) * 0.15
        elif 8 <= hour < 10:  # Morning rush
            base_occupancy = 0.75 + (hour - 8) * 0.10
        elif 10 <= hour < 12:  # Late morning (peak)
            base_occupancy = 0.90
        elif 12 <= hour < 14:  # Lunch time (slight dip)
            base_occupancy = 0.75
        elif 14 <= hour < 17:  # Afternoon (high)
            base_occupancy = 0.85
        elif 17 <= hour < 19:  # Evening departure
            base_occupancy = 0.70 - (hour - 17) * 0.15
        elif 19 <= hour < 22:  # Evening
            base_occupancy = 0.35
        else:  # Late evening
            base_occupancy = 0.20
        
        # Weekend adjustment (lower occupancy)
        if day_of_week >= 5:  # Saturday or Sunday
            base_occupancy *= 0.6
        
        # Add some randomness
        noise = np.random.normal(0, 0.08)  # 8% standard deviation
        occupancy_pct = np.clip(base_occupancy + noise, 0, 1)
        
        # Convert to actual occupied slots
        occupied = int(round(occupancy_pct * total_slots))
        
        # Add occasional spikes (special events)
        if np.random.random() < 0.02:  # 2% chance of spike
            occupied = min(total_slots, occupied + np.random.randint(3, 8))
        
        occupancy_data.append(occupied)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'total_occupied': occupancy_data,
        'total_slots': total_slots
    })
    
    # Calculate availability
    df['available_slots'] = df['total_slots'] - df['total_occupied']
    
    return df


def add_statistics(df):
    """Add summary statistics"""
    print("="*70)
    print("GENERATED PARKING DATASET SUMMARY")
    print("="*70)
    print(f"\nDataset Shape: {df.shape}")
    print(f"Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Total Duration: {(df['timestamp'].max() - df['timestamp'].min()).days} days")
    print(f"\nTotal Capacity: {df['total_slots'].iloc[0]} slots")
    
    print("\n" + "-"*70)
    print("OCCUPANCY STATISTICS")
    print("-"*70)
    print(df['total_occupied'].describe())
    
    print("\n" + "-"*70)
    print("OCCUPANCY BY HOUR OF DAY (Average)")
    print("-"*70)
    df['hour'] = df['timestamp'].dt.hour
    hourly_avg = df.groupby('hour')['total_occupied'].mean()
    for hour, avg_occ in hourly_avg.items():
        bar = '█' * int(avg_occ)
        print(f"{hour:02d}:00 - {avg_occ:5.1f} slots {bar}")
    
    print("\n" + "-"*70)
    print("OCCUPANCY BY DAY OF WEEK (Average)")
    print("-"*70)
    df['day_name'] = df['timestamp'].dt.day_name()
    daily_avg = df.groupby('day_name')['total_occupied'].mean()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    for day in day_order:
        if day in daily_avg.index:
            avg_occ = daily_avg[day]
            bar = '█' * int(avg_occ)
            print(f"{day:12s} - {avg_occ:5.1f} slots {bar}")
    
    # Clean up temporary columns
    df.drop(['hour', 'day_name'], axis=1, inplace=True)
    
    return df


def plot_sample_data(df, save_path='parking_data_visualization.png'):
    """Create visualization of generated data"""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        # Plot 1: Full time series
        ax1 = axes[0]
        ax1.plot(df['timestamp'], df['total_occupied'], linewidth=1, alpha=0.7)
        ax1.axhline(y=df['total_slots'].iloc[0], color='red', 
                   linestyle='--', label='Total Capacity', linewidth=2)
        ax1.fill_between(df['timestamp'], 0, df['total_occupied'], alpha=0.3)
        ax1.set_title('Parking Occupancy Over Time (Full Dataset)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date', fontsize=11)
        ax1.set_ylabel('Occupied Slots', fontsize=11)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Single day sample
        ax2 = axes[1]
        sample_day = df[df['timestamp'].dt.date == df['timestamp'].dt.date.iloc[len(df)//2]]
        ax2.plot(sample_day['timestamp'], sample_day['total_occupied'], 
                marker='o', linewidth=2, markersize=4)
        ax2.axhline(y=df['total_slots'].iloc[0], color='red', 
                   linestyle='--', label='Total Capacity', linewidth=2)
        ax2.fill_between(sample_day['timestamp'], 0, sample_day['total_occupied'], alpha=0.3)
        ax2.set_title('Sample Day - Typical Daily Pattern', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Time', fontsize=11)
        ax2.set_ylabel('Occupied Slots', fontsize=11)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Hourly average pattern
        ax3 = axes[2]
        df['hour'] = df['timestamp'].dt.hour
        hourly_stats = df.groupby('hour')['total_occupied'].agg(['mean', 'std'])
        ax3.plot(hourly_stats.index, hourly_stats['mean'], 
                marker='o', linewidth=2.5, markersize=6, label='Average')
        ax3.fill_between(hourly_stats.index, 
                        hourly_stats['mean'] - hourly_stats['std'],
                        hourly_stats['mean'] + hourly_stats['std'],
                        alpha=0.3, label='±1 Std Dev')
        ax3.axhline(y=df['total_slots'].iloc[0], color='red', 
                   linestyle='--', label='Total Capacity', linewidth=2)
        ax3.set_title('Average Hourly Occupancy Pattern', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Hour of Day', fontsize=11)
        ax3.set_ylabel('Occupied Slots', fontsize=11)
        ax3.set_xticks(range(0, 24, 2))
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        df.drop(['hour'], axis=1, inplace=True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Visualization saved as '{save_path}'")
        plt.close()
        
    except ImportError:
        print("\n⚠ Matplotlib not available. Skipping visualization.")


def main():
    print("\n" + "="*70)
    print(" REALISTIC PARKING DATA GENERATOR ")
    print("="*70)
    
    # Generate data
    print("\nGenerating realistic parking occupancy data...")
    df = generate_realistic_parking_data(
        total_slots=20,      # 20 parking slots
        days=14,             # 2 weeks of data
        interval_minutes=5,  # 5-minute intervals
        seed=42              # Reproducible results
    )
    
    # Add statistics
    df = add_statistics(df)
    
    # Save to CSV
    output_file = 'parking_data.csv'
    # Save only required columns for model
    df[['timestamp', 'total_occupied', 'total_slots']].to_csv(output_file, index=False)
    print(f"\n✓ Data saved as '{output_file}'")
    print(f"  Records: {len(df):,}")
    print(f"  Columns: {list(df.columns)}")
    
    # Create visualization
    plot_sample_data(df)
    
    # Show sample records
    print("\n" + "="*70)
    print("SAMPLE RECORDS")
    print("="*70)
    print("\nFirst 10 records:")
    print(df[['timestamp', 'total_occupied', 'available_slots']].head(10).to_string(index=False))
    print("\n...")
    print("\nLast 5 records:")
    print(df[['timestamp', 'total_occupied', 'available_slots']].tail(5).to_string(index=False))
    
    print("\n" + "="*70)
    print("✓ Data generation completed successfully!")
    print("="*70)
    print("\nYou can now run:")
    print("  1. python arima_t.py  (for training)")
    print("  2. python arima_i.py  (for inference)")
    print("="*70)


if __name__ == "__main__":
    main()