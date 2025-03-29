import folium
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os

def create_risk_map():
    """Create an example risk map"""
    # Create base map centered on Chandigarh
    chandigarh_coords = [30.7333, 76.7794]
    m = folium.Map(location=chandigarh_coords, zoom_start=10)
    
    # Add example markers
    locations = [
        {'name': 'Chandigarh', 'coords': [30.7333, 76.7794], 'risk': 'High'},
        {'name': 'Mohali', 'coords': [30.7046, 76.7179], 'risk': 'Medium'},
        {'name': 'Panchkula', 'coords': [30.6942, 76.8606], 'risk': 'Low'}
    ]
    
    colors = {'High': 'red', 'Medium': 'yellow', 'Low': 'green'}
    
    for loc in locations:
        folium.CircleMarker(
            location=loc['coords'],
            radius=10,
            color=colors[loc['risk']],
            fill=True,
            popup=f"<b>{loc['name']}</b><br>Risk Level: {loc['risk']}"
        ).add_to(m)
    
    m.save('screenshots/risk_map.html')
    
    # Convert HTML to PNG using a headless browser
    import webbrowser
    webbrowser.open('screenshots/risk_map.html')

def create_risk_distribution():
    """Create an example risk distribution plot"""
    plt.figure(figsize=(10, 6))
    
    # Generate example data
    risk_levels = ['Low', 'Medium', 'High']
    counts = [45, 30, 25]
    
    # Create bar plot
    sns.barplot(x=risk_levels, y=counts)
    
    # Customize plot
    plt.title('Distribution of Flood Risk Levels')
    plt.xlabel('Risk Level')
    plt.ylabel('Number of Areas')
    
    # Save plot
    plt.savefig('screenshots/risk_distribution.png')
    plt.close()

def create_rainfall_pattern():
    """Create an example rainfall pattern plot"""
    plt.figure(figsize=(12, 6))
    
    # Generate example data
    dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
    rainfall = np.random.normal(10, 5, 365)
    
    # Create line plot
    plt.plot(dates, rainfall)
    
    # Customize plot
    plt.title('Historical Rainfall Patterns')
    plt.xlabel('Date')
    plt.ylabel('Rainfall (mm)')
    plt.xticks(rotation=45)
    
    # Save plot
    plt.savefig('screenshots/rainfall_pattern.png')
    plt.close()

def create_risk_dashboard():
    """Create an example risk dashboard"""
    plt.figure(figsize=(15, 10))
    
    # Create subplots
    gs = plt.GridSpec(2, 2)
    
    # Risk Distribution
    ax1 = plt.subplot(gs[0, 0])
    risk_levels = ['Low', 'Medium', 'High']
    counts = [45, 30, 25]
    sns.barplot(x=risk_levels, y=counts, ax=ax1)
    ax1.set_title('Risk Distribution')
    
    # Rainfall Trend
    ax2 = plt.subplot(gs[0, 1])
    dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
    rainfall = np.random.normal(10, 5, 30)
    ax2.plot(dates, rainfall)
    ax2.set_title('30-Day Rainfall Trend')
    
    # Risk Metrics
    ax3 = plt.subplot(gs[1, 0])
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [0.85, 0.82, 0.88, 0.85]
    sns.barplot(x=metrics, y=values, ax=ax3)
    ax3.set_title('Model Performance Metrics')
    
    # Risk Timeline
    ax4 = plt.subplot(gs[1, 1])
    dates = pd.date_range(start='2023-01-01', periods=12, freq='M')
    risk_scores = np.random.normal(0.5, 0.2, 12)
    ax4.plot(dates, risk_scores)
    ax4.set_title('Monthly Risk Score Trend')
    
    plt.tight_layout()
    plt.savefig('screenshots/risk_dashboard.png')
    plt.close()

def create_mobile_interface():
    """Create an example mobile interface screenshot"""
    plt.figure(figsize=(9, 16))
    
    # Create a simple mobile interface layout
    plt.subplot(311)
    plt.text(0.5, 0.5, 'Flood Risk Assessment', 
             ha='center', va='center', fontsize=20)
    plt.axis('off')
    
    plt.subplot(312)
    plt.text(0.5, 0.5, 'Current Risk Level: MEDIUM', 
             ha='center', va='center', fontsize=16)
    plt.axis('off')
    
    plt.subplot(313)
    plt.text(0.5, 0.5, 'Tap to view details', 
             ha='center', va='center', fontsize=14)
    plt.axis('off')
    
    plt.savefig('screenshots/mobile_interface.png')
    plt.close()

def main():
    """Generate all screenshots"""
    # Create screenshots directory if it doesn't exist
    os.makedirs('screenshots', exist_ok=True)
    
    # Generate all screenshots
    print("Generating screenshots...")
    create_risk_map()
    create_risk_distribution()
    create_rainfall_pattern()
    create_risk_dashboard()
    create_mobile_interface()
    print("Screenshots generated successfully!")

if __name__ == "__main__":
    main() 