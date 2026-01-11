"""
Data Download Module for SkyPredict
====================================

Downloads flight data from BTS and weather data from NOAA/OpenWeather.
Handles caching to avoid re-downloading.
"""

import os
import logging
import requests
import zipfile
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List
from io import BytesIO

import pandas as pd
from tqdm import tqdm

from ..config import config, RAW_DATA_DIR

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FlightDataDownloader:
    """
    Downloads flight data from Bureau of Transportation Statistics (BTS).

    Data source: https://www.transtats.bts.gov/
    Alternative: Kaggle datasets for historical data
    """

    # BTS API endpoint (On-Time Performance)
    BTS_BASE_URL = "https://www.transtats.bts.gov/PREZIP/"

    # Kaggle dataset URLs (pre-downloaded alternatives)
    KAGGLE_DATASETS = {
        "2019_with_weather": "https://www.kaggle.com/datasets/threnjen/2019-airline-delays-and-cancellations",
        "2018_2022": "https://www.kaggle.com/datasets/robikscube/flight-delay-dataset-20182022",
        "airlines_delay": "https://www.kaggle.com/datasets/jimschacko/airlines-dataset-to-predict-a-delay"
    }

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or RAW_DATA_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_bts_data(self, year: int, month: int) -> Optional[Path]:
        """
        Download BTS On-Time Performance data for a specific month.

        Args:
            year: Year (e.g., 2023)
            month: Month (1-12)

        Returns:
            Path to downloaded file or None if failed
        """
        filename = f"On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
        url = f"{self.BTS_BASE_URL}{filename}"
        output_path = self.output_dir / f"flights_{year}_{month:02d}.csv"

        # Check if already downloaded
        if output_path.exists():
            logger.info(f"Data already exists: {output_path}")
            return output_path

        logger.info(f"Downloading BTS data for {year}-{month:02d}...")

        try:
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()

            # Extract CSV from ZIP
            with zipfile.ZipFile(BytesIO(response.content)) as zf:
                csv_name = [n for n in zf.namelist() if n.endswith('.csv')][0]
                with zf.open(csv_name) as csv_file:
                    df = pd.read_csv(csv_file, low_memory=False)
                    df.to_csv(output_path, index=False)

            logger.info(f"Downloaded and saved: {output_path}")
            return output_path

        except requests.RequestException as e:
            logger.error(f"Failed to download BTS data: {e}")
            return None
        except Exception as e:
            logger.error(f"Error processing BTS data: {e}")
            return None

    def download_multiple_months(
        self,
        start_year: int,
        start_month: int,
        end_year: int,
        end_month: int
    ) -> List[Path]:
        """
        Download BTS data for a range of months.

        Args:
            start_year: Starting year
            start_month: Starting month
            end_year: Ending year
            end_month: Ending month

        Returns:
            List of paths to downloaded files
        """
        downloaded_files = []
        current = datetime(start_year, start_month, 1)
        end = datetime(end_year, end_month, 1)

        while current <= end:
            file_path = self.download_bts_data(current.year, current.month)
            if file_path:
                downloaded_files.append(file_path)
            current += timedelta(days=32)
            current = current.replace(day=1)

        return downloaded_files

    def load_kaggle_sample(self, dataset_name: str = "sample") -> pd.DataFrame:
        """
        Load a sample dataset for development/testing.
        Creates a synthetic sample if no real data available.

        For production, download from Kaggle manually:
        - 2019 with Weather: kaggle.com/datasets/threnjen/2019-airline-delays-and-cancellations
        - 2018-2022: kaggle.com/datasets/robikscube/flight-delay-dataset-20182022
        """
        sample_path = self.output_dir / "sample_flights.csv"

        if sample_path.exists():
            logger.info(f"Loading existing sample: {sample_path}")
            return pd.read_csv(sample_path)

        logger.info("Creating synthetic sample dataset for development...")
        df = self._create_sample_dataset()
        df.to_csv(sample_path, index=False)
        logger.info(f"Sample dataset saved: {sample_path}")

        return df

    def _create_sample_dataset(self, n_samples: int = 100000) -> pd.DataFrame:
        """
        Create a realistic synthetic dataset for development.
        Structure matches BTS On-Time Performance data.
        """
        import numpy as np
        np.random.seed(42)

        # Airlines and airports
        airlines = ["AA", "DL", "UA", "WN", "AS", "B6", "NK", "F9"]
        airports = [
            "ATL", "DFW", "DEN", "ORD", "LAX", "CLT", "MCO", "LAS",
            "PHX", "MIA", "SEA", "IAH", "JFK", "EWR", "SFO"
        ]

        # Generate dates for 2023
        dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="H")
        sampled_dates = np.random.choice(dates, n_samples)

        # Create DataFrame
        df = pd.DataFrame({
            "FL_DATE": pd.to_datetime(sampled_dates).date,
            "OP_UNIQUE_CARRIER": np.random.choice(airlines, n_samples),
            "OP_CARRIER_FL_NUM": np.random.randint(1, 9999, n_samples),
            "ORIGIN": np.random.choice(airports, n_samples),
            "DEST": np.random.choice(airports, n_samples),
            "CRS_DEP_TIME": np.random.randint(0, 2359, n_samples),
            "DEP_TIME": np.random.randint(0, 2359, n_samples),
            "DEP_DELAY": np.random.normal(5, 30, n_samples),
            "CRS_ARR_TIME": np.random.randint(0, 2359, n_samples),
            "ARR_TIME": np.random.randint(0, 2359, n_samples),
            "ARR_DELAY": np.random.normal(5, 35, n_samples),
            "CANCELLED": np.random.choice([0, 1], n_samples, p=[0.98, 0.02]),
            "DIVERTED": np.random.choice([0, 1], n_samples, p=[0.995, 0.005]),
            "CRS_ELAPSED_TIME": np.random.randint(60, 400, n_samples),
            "ACTUAL_ELAPSED_TIME": np.random.randint(60, 420, n_samples),
            "AIR_TIME": np.random.randint(50, 380, n_samples),
            "DISTANCE": np.random.randint(100, 3000, n_samples),
            "CARRIER_DELAY": np.abs(np.random.normal(0, 20, n_samples)),
            "WEATHER_DELAY": np.abs(np.random.normal(0, 15, n_samples)),
            "NAS_DELAY": np.abs(np.random.normal(0, 10, n_samples)),
            "SECURITY_DELAY": np.abs(np.random.normal(0, 2, n_samples)),
            "LATE_AIRCRAFT_DELAY": np.abs(np.random.normal(0, 25, n_samples)),
        })

        # Remove same origin-destination
        df = df[df["ORIGIN"] != df["DEST"]]

        # Clip delays to realistic values
        df["DEP_DELAY"] = df["DEP_DELAY"].clip(-30, 300)
        df["ARR_DELAY"] = df["ARR_DELAY"].clip(-30, 300)

        # Set delays to 0 for cancelled flights
        df.loc[df["CANCELLED"] == 1, ["DEP_DELAY", "ARR_DELAY"]] = None

        return df.reset_index(drop=True)


class WeatherDataDownloader:
    """
    Downloads weather data from OpenWeatherMap or NOAA.

    For historical data: NOAA ISD (Integrated Surface Database)
    For forecasts: OpenWeatherMap API
    """

    NOAA_BASE_URL = "https://www.ncei.noaa.gov/data/global-hourly/access/"
    OPENWEATHER_URL = "https://api.openweathermap.org/data/2.5/"

    # Airport to weather station mapping (ICAO -> USAF-WBAN)
    AIRPORT_STATIONS = {
        "ATL": "722190-13874",
        "DFW": "722590-03927",
        "DEN": "725650-03017",
        "ORD": "725300-94846",
        "LAX": "722950-23174",
        "JFK": "744860-94789",
        "SFO": "724940-23234",
        "SEA": "727930-24233",
        "MIA": "722020-12839",
        "LAS": "723860-23169",
    }

    # Airport coordinates for API calls
    AIRPORT_COORDS = {
        "ATL": (33.6407, -84.4277),
        "DFW": (32.8998, -97.0403),
        "DEN": (39.8561, -104.6737),
        "ORD": (41.9742, -87.9073),
        "LAX": (33.9416, -118.4085),
        "JFK": (40.6413, -73.7781),
        "SFO": (37.6213, -122.3790),
        "SEA": (47.4502, -122.3088),
        "MIA": (25.7959, -80.2870),
        "LAS": (36.0840, -115.1537),
        "CLT": (35.2140, -80.9431),
        "MCO": (28.4312, -81.3081),
        "PHX": (33.4373, -112.0078),
        "IAH": (29.9902, -95.3368),
        "EWR": (40.6895, -74.1745),
        "MSP": (44.8848, -93.2223),
        "BOS": (42.3656, -71.0096),
        "DTW": (42.2162, -83.3554),
        "FLL": (26.0742, -80.1506),
        "PHL": (39.8729, -75.2437),
    }

    def __init__(self, api_key: Optional[str] = None, output_dir: Optional[Path] = None):
        self.api_key = api_key or config.OPENWEATHER_API_KEY
        self.output_dir = output_dir or RAW_DATA_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_current_weather(self, airport_code: str) -> Optional[dict]:
        """
        Get current weather for an airport using OpenWeatherMap.

        Args:
            airport_code: IATA airport code (e.g., "JFK")

        Returns:
            Weather data dict or None if failed
        """
        if not self.api_key:
            logger.warning("OpenWeatherMap API key not set")
            return None

        coords = self.AIRPORT_COORDS.get(airport_code)
        if not coords:
            logger.warning(f"Unknown airport: {airport_code}")
            return None

        url = f"{self.OPENWEATHER_URL}weather"
        params = {
            "lat": coords[0],
            "lon": coords[1],
            "appid": self.api_key,
            "units": "metric"
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            return {
                "airport": airport_code,
                "timestamp": datetime.utcnow().isoformat(),
                "temp_c": data["main"]["temp"],
                "humidity": data["main"]["humidity"],
                "pressure_hpa": data["main"]["pressure"],
                "wind_speed_ms": data["wind"]["speed"],
                "wind_deg": data["wind"].get("deg", 0),
                "visibility_m": data.get("visibility", 10000),
                "clouds_pct": data["clouds"]["all"],
                "weather_main": data["weather"][0]["main"],
                "weather_desc": data["weather"][0]["description"],
            }

        except requests.RequestException as e:
            logger.error(f"Failed to get weather for {airport_code}: {e}")
            return None

    def get_weather_forecast(self, airport_code: str, hours: int = 24) -> Optional[List[dict]]:
        """
        Get weather forecast for an airport.

        Args:
            airport_code: IATA airport code
            hours: Number of hours to forecast

        Returns:
            List of weather forecasts or None if failed
        """
        if not self.api_key:
            logger.warning("OpenWeatherMap API key not set")
            return None

        coords = self.AIRPORT_COORDS.get(airport_code)
        if not coords:
            logger.warning(f"Unknown airport: {airport_code}")
            return None

        url = f"{self.OPENWEATHER_URL}forecast"
        params = {
            "lat": coords[0],
            "lon": coords[1],
            "appid": self.api_key,
            "units": "metric",
            "cnt": hours // 3  # API returns 3-hour intervals
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            forecasts = []
            for item in data["list"]:
                forecasts.append({
                    "airport": airport_code,
                    "timestamp": item["dt_txt"],
                    "temp_c": item["main"]["temp"],
                    "humidity": item["main"]["humidity"],
                    "pressure_hpa": item["main"]["pressure"],
                    "wind_speed_ms": item["wind"]["speed"],
                    "wind_deg": item["wind"].get("deg", 0),
                    "visibility_m": item.get("visibility", 10000),
                    "clouds_pct": item["clouds"]["all"],
                    "weather_main": item["weather"][0]["main"],
                    "weather_desc": item["weather"][0]["description"],
                    "pop": item.get("pop", 0),  # Probability of precipitation
                })

            return forecasts

        except requests.RequestException as e:
            logger.error(f"Failed to get forecast for {airport_code}: {e}")
            return None

    def create_sample_weather_data(self, flight_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create synthetic weather data matching flight dates/airports.
        Used for development when API is not available.

        Args:
            flight_df: DataFrame with flight data (needs ORIGIN, DEST, FL_DATE)

        Returns:
            DataFrame with weather data for each airport-date combination
        """
        import numpy as np
        np.random.seed(42)

        # Get unique airport-date combinations
        origins = flight_df[["FL_DATE", "ORIGIN"]].rename(columns={"ORIGIN": "airport"})
        dests = flight_df[["FL_DATE", "DEST"]].rename(columns={"DEST": "airport"})
        airport_dates = pd.concat([origins, dests]).drop_duplicates()

        n = len(airport_dates)

        # Generate realistic weather data
        weather_df = pd.DataFrame({
            "date": airport_dates["FL_DATE"],
            "airport": airport_dates["airport"],
            "temp_c": np.random.normal(15, 15, n).clip(-20, 45),
            "humidity": np.random.uniform(30, 100, n),
            "pressure_hpa": np.random.normal(1013, 20, n),
            "wind_speed_ms": np.abs(np.random.normal(5, 5, n)),
            "wind_deg": np.random.uniform(0, 360, n),
            "visibility_m": np.random.choice(
                [10000, 8000, 5000, 2000, 1000, 500],
                n,
                p=[0.6, 0.2, 0.1, 0.05, 0.03, 0.02]
            ),
            "clouds_pct": np.random.uniform(0, 100, n),
            "precipitation_mm": np.random.exponential(2, n).clip(0, 50),
        })

        # Add weather conditions based on other features
        conditions = []
        for _, row in weather_df.iterrows():
            if row["visibility_m"] < 1000:
                conditions.append("Fog")
            elif row["precipitation_mm"] > 10:
                conditions.append("Rain")
            elif row["wind_speed_ms"] > 15:
                conditions.append("Windy")
            elif row["clouds_pct"] > 80:
                conditions.append("Cloudy")
            else:
                conditions.append("Clear")

        weather_df["weather_condition"] = conditions

        return weather_df


if __name__ == "__main__":
    # Test the downloaders
    print("Testing FlightDataDownloader...")
    flight_downloader = FlightDataDownloader()
    sample_df = flight_downloader.load_kaggle_sample()
    print(f"Sample flights shape: {sample_df.shape}")
    print(f"Columns: {list(sample_df.columns)}")

    print("\nTesting WeatherDataDownloader...")
    weather_downloader = WeatherDataDownloader()
    weather_df = weather_downloader.create_sample_weather_data(sample_df)
    print(f"Sample weather shape: {weather_df.shape}")
    print(f"Columns: {list(weather_df.columns)}")
