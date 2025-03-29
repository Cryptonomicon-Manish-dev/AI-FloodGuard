import folium
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from geopy.geocoders import Nominatim
import logging
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FloodRiskVisualizer:
    def __init__(self):
        self.geolocator = Nominatim(user_agent="flood_risk_visualizer")
        
    def create_risk_map(self, data_path, output_path='visualizations/risk_map.html'):
        """
        Create an interactive map showing flood risk levels
        """
        try:
            # Load prediction data
            df = pd.read_csv(data_path)
            
            # Create base map centered on Chandigarh
            chandigarh_coords = [30.7333, 76.7794]
            m = folium.Map(location=chandigarh_coords, zoom_start=10)
            
            # Define color scheme for risk levels
            risk_colors = {
                'Low': 'green',
                'Medium': 'yellow',
                'High': 'red'
            }
            
            # Add markers for each location
            for idx, row in df.iterrows():
                # Get coordinates for the location
                location = self.geolocator.geocode(row['Location'])
                if location:
                    # Create popup content
                    popup_content = f"""
                    <b>Location:</b> {row['Location']}<br>
                    <b>Risk Level:</b> {row['prediction']}<br>
                    <b>Probability:</b> {row['probability']:.2f}
                    """
                    
                    # Add marker to map
                    folium.CircleMarker(
                        location=[location.latitude, location.longitude],
                        radius=10,
                        color=risk_colors.get(row['prediction'], 'gray'),
                        fill=True,
                        popup=popup_content
                    ).add_to(m)
            
            # Save map
            m.save(output_path)
            logger.info(f"Risk map saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error creating risk map: {str(e)}")
            raise

    def create_risk_distribution_plot(self, data_path, output_path='visualizations/risk_distribution.png'):
        """
        Create a bar plot showing the distribution of risk levels
        """
        try:
            # Load prediction data
            df = pd.read_csv(data_path)
            
            # Create figure
            plt.figure(figsize=(10, 6))
            
            # Create bar plot
            sns.countplot(data=df, x='prediction')
            
            # Customize plot
            plt.title('Distribution of Flood Risk Levels')
            plt.xlabel('Risk Level')
            plt.ylabel('Count')
            
            # Save plot
            plt.savefig(output_path)
            plt.close()
            
            logger.info(f"Risk distribution plot saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error creating risk distribution plot: {str(e)}")
            raise

    def create_probability_heatmap(self, data_path, output_path='visualizations/probability_heatmap.png'):
        """
        Create a heatmap showing probability distribution
        """
        try:
            # Load prediction data
            df = pd.read_csv(data_path)
            
            # Create figure
            plt.figure(figsize=(10, 6))
            
            # Create heatmap
            sns.heatmap(
                pd.crosstab(df['prediction'], pd.qcut(df['probability'], q=5)),
                annot=True,
                fmt='d',
                cmap='YlOrRd'
            )
            
            # Customize plot
            plt.title('Flood Risk Probability Heatmap')
            plt.xlabel('Probability Quintiles')
            plt.ylabel('Risk Level')
            
            # Save plot
            plt.savefig(output_path)
            plt.close()
            
            logger.info(f"Probability heatmap saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error creating probability heatmap: {str(e)}")
            raise

def main():
    parser = argparse.ArgumentParser(description='Flood Risk Visualization')
    parser.add_argument('--data', type=str, required=True,
                      help='Path to prediction data CSV file')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                      help='Directory to save visualization outputs')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    
    visualizer = FloodRiskVisualizer()
    
    try:
        # Create all visualizations
        visualizer.create_risk_map(
            args.data,
            os.path.join(args.output_dir, 'risk_map.html')
        )
        
        visualizer.create_risk_distribution_plot(
            args.data,
            os.path.join(args.output_dir, 'risk_distribution.png')
        )
        
        visualizer.create_probability_heatmap(
            args.data,
            os.path.join(args.output_dir, 'probability_heatmap.png')
        )
        
        logger.info("All visualizations created successfully")
        
    except Exception as e:
        logger.error(f"Error in visualization process: {str(e)}")
        raise

if __name__ == "__main__":
    main() 