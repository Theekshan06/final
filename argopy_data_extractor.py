#!/usr/bin/env python3
"""
ARGO Data Extractor using argopy library
Downloads delayed-mode (quality-controlled) ARGO data for Indian Ocean regions
Converts to parquet format maintaining schema compatibility
"""

import argopy
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from pathlib import Path
import warnings
import json
import hashlib
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('argopy_extractor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ArgopyDataExtractor:
    """Extract ARGO data using argopy for Indian Ocean regions"""

    def __init__(self, base_dir=None):
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).parent.parent
        self.parquet_dir = self.base_dir / "parquet_data"
        self.metadata_file = self.parquet_dir / "extraction_metadata.json"

        # Setup argopy for delayed-mode data
        argopy.set_options(mode='expert')  # Expert mode for advanced features
        argopy.set_options(src='erddap')   # Use ERDDAP data source

        # Indian Ocean region boundaries (entire Indian Ocean)
        # Latitude: ~50°S to ~30°N, Longitude: ~20°E to ~120°E
        self.indian_ocean_regions = {
            "Indian Ocean": {"lat_min": -50.0, "lat_max": 30.0, "lon_min": 20.0, "lon_max": 120.0}
        }

        # Cache for existing float IDs to avoid repeated database reads
        self._existing_float_ids_cache = None
        self._existing_profile_signatures_cache = None

        # Time periods to extract (delayed-mode data available ~6 months delayed)
        current_date = datetime.now()
        delayed_cutoff = current_date - timedelta(days=180)  # 6 months delay

        self.time_periods = []
        start_year = 2000  # ARGO program started around 2000

        for year in range(start_year, delayed_cutoff.year + 1):
            if year == delayed_cutoff.year:
                # For current year, only go up to delayed cutoff
                end_date = delayed_cutoff.strftime('%Y-%m-%d')
            else:
                end_date = f"{year}-12-31"

            self.time_periods.append({
                'start_date': f"{year}-01-01",
                'end_date': end_date,
                'year': year
            })

        logger.info(f"Will extract data from {start_year} to {delayed_cutoff.year}")
        logger.info(f"Total time periods: {len(self.time_periods)}")

    def setup_database(self, clear_existing=False):
        """Setup database structure"""
        logger.info("Setting up database structure...")

        if clear_existing and self.parquet_dir.exists():
            logger.info("Clearing existing data...")
            import shutil
            shutil.rmtree(self.parquet_dir)
            # Reset caches when clearing data
            self._existing_float_ids_cache = None
            self._existing_profile_signatures_cache = None

        self.parquet_dir.mkdir(exist_ok=True)

        # Initialize empty parquet files with proper schema (only if they don't exist)
        if clear_existing or not all((self.parquet_dir / f"{table}.parquet").exists() for table in ['floats', 'profiles', 'measurements']):
            self.initialize_empty_tables()

        # Initialize extraction metadata tracking
        self.initialize_metadata_tracking()
        logger.info("Database structure ready")

    def initialize_empty_tables(self):
        """Initialize empty parquet tables with correct schema"""

        # Floats table schema
        floats_schema = {
            'float_id': 'object',
            'wmo_number': 'int64',
            'program_name': 'object',
            'platform_type': 'object',
            'data_assembly_center': 'object',
            'deployment_date': 'object',
            'deployment_latitude': 'float64',
            'deployment_longitude': 'float64',
            'deployment_depth': 'object',
            'current_status': 'object',
            'last_latitude': 'float64',
            'last_longitude': 'float64',
            'last_update': 'object',
            'cycle_time_days': 'int64',
            'park_pressure_dbar': 'object',
            'profile_pressure_dbar': 'object',
            'total_profiles': 'int64',
            'quality_profiles': 'int64',
            'metadata_text': 'object',
            'created_at': 'object',
            'updated_at': 'object'
        }

        # Profiles table schema
        profiles_schema = {
            'profile_id': 'int64',
            'float_id': 'object',
            'cycle_number': 'int64',
            'profile_direction': 'object',
            'profile_date': 'object',
            'latitude': 'float64',
            'longitude': 'float64',
            'max_pressure': 'float64',
            'num_levels': 'int64',
            'vertical_sampling_scheme': 'object',
            'data_mode': 'object',
            'data_quality_flag': 'int64',
            'processing_date': 'object',
            'netcdf_filename': 'object',
            'file_checksum': 'object',
            'profile_summary': 'object',
            'created_at': 'object',
            'updated_at': 'object'
        }

        # Measurements table schema
        measurements_schema = {
            'measurement_id': 'int64',
            'profile_id': 'int64',
            'pressure': 'float64',
            'depth': 'float64',
            'pressure_qc': 'int64',
            'temperature': 'float64',
            'temperature_qc': 'int64',
            'salinity': 'float64',
            'salinity_qc': 'int64',
            'dissolved_oxygen': 'object',
            'dissolved_oxygen_qc': 'int64',
            'ph_in_situ': 'object',
            'ph_in_situ_qc': 'int64',
            'chlorophyll_a': 'object',
            'chlorophyll_a_qc': 'int64',
            'particle_backscattering': 'object',
            'particle_backscattering_qc': 'int64',
            'downward_irradiance': 'object',
            'downward_irradiance_qc': 'int64',
            'potential_temperature': 'object',
            'potential_density': 'object',
            'buoyancy_frequency': 'object',
            'mixed_layer_depth': 'object',
            'processing_level': 'object',
            'interpolated': 'int64',
            'spike_test_flag': 'int64',
            'gradient_test_flag': 'int64',
            'parameter_summary': 'object',
            'created_at': 'object'
        }

        # Create empty DataFrames and save as parquet
        for table_name, schema in [('floats', floats_schema), ('profiles', profiles_schema), ('measurements', measurements_schema)]:
            df = pd.DataFrame({col: pd.Series(dtype=dtype) for col, dtype in schema.items()})
            df.to_parquet(self.parquet_dir / f"{table_name}.parquet", index=False)
            logger.info(f"Initialized empty {table_name} table")

    def initialize_metadata_tracking(self):
        """Initialize metadata file to track what has been extracted"""
        if not self.metadata_file.exists():
            metadata = {
                "extracted_combinations": {},  # region_year combinations
                "failed_combinations": {},     # combinations that failed
                "last_extraction_date": None,
                "extraction_stats": {
                    "total_floats": 0,
                    "total_profiles": 0,
                    "total_measurements": 0
                }
            }
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info("Initialized extraction metadata tracking")
        else:
            logger.info("Found existing extraction metadata")

    def load_metadata(self):
        """Load extraction metadata"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {"extracted_combinations": {}, "failed_combinations": {}, "last_extraction_date": None, "extraction_stats": {}}

    def save_metadata(self, metadata):
        """Save extraction metadata"""
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def get_combination_key(self, region_name, year):
        """Generate unique key for region-year combination"""
        return f"{region_name}_{year}"

    def is_combination_extracted(self, region_name, year):
        """Check if a region-year combination has already been extracted successfully"""
        metadata = self.load_metadata()
        combination_key = self.get_combination_key(region_name, year)
        return combination_key in metadata.get("extracted_combinations", {})

    def is_combination_failed_recently(self, region_name, year, max_age_days=7):
        """Check if a combination failed recently (to avoid immediate retries)"""
        metadata = self.load_metadata()
        combination_key = self.get_combination_key(region_name, year)

        if combination_key in metadata.get("failed_combinations", {}):
            fail_date_str = metadata["failed_combinations"][combination_key]["failure_date"]
            fail_date = datetime.fromisoformat(fail_date_str)
            days_since_failure = (datetime.now() - fail_date).days
            return days_since_failure < max_age_days
        return False

    def mark_combination_extracted(self, region_name, year, stats=None):
        """Mark a region-year combination as successfully extracted"""
        metadata = self.load_metadata()
        combination_key = self.get_combination_key(region_name, year)

        metadata["extracted_combinations"][combination_key] = {
            "region": region_name,
            "year": year,
            "extraction_date": datetime.now().isoformat(),
            "stats": stats or {"floats": 0, "profiles": 0, "measurements": 0}
        }

        # Remove from failed combinations if it was there
        if combination_key in metadata.get("failed_combinations", {}):
            del metadata["failed_combinations"][combination_key]

        metadata["last_extraction_date"] = datetime.now().isoformat()
        self.save_metadata(metadata)
        logger.info(f"Marked {combination_key} as successfully extracted")

    def mark_combination_failed(self, region_name, year, error_msg):
        """Mark a region-year combination as failed"""
        metadata = self.load_metadata()
        combination_key = self.get_combination_key(region_name, year)

        metadata["failed_combinations"][combination_key] = {
            "region": region_name,
            "year": year,
            "failure_date": datetime.now().isoformat(),
            "error": error_msg[:500]  # Limit error message length
        }

        self.save_metadata(metadata)
        logger.info(f"Marked {combination_key} as failed")

    def load_existing_float_ids(self):
        """Load and cache existing float IDs for efficient duplicate checking"""
        if self._existing_float_ids_cache is None:
            try:
                existing_floats = pd.read_parquet(self.parquet_dir / "floats.parquet")
                self._existing_float_ids_cache = set(existing_floats['float_id'].values)
                logger.info(f"Loaded {len(self._existing_float_ids_cache)} existing float IDs into cache")
            except Exception as e:
                logger.warning(f"Error loading existing float IDs: {e}")
                self._existing_float_ids_cache = set()
        return self._existing_float_ids_cache

    def load_existing_profile_signatures(self):
        """Load and cache existing profile signatures for efficient duplicate checking"""
        if self._existing_profile_signatures_cache is None:
            try:
                existing_profiles = pd.read_parquet(self.parquet_dir / "profiles.parquet")
                self._existing_profile_signatures_cache = set()
                for _, row in existing_profiles.iterrows():
                    signature = f"{row['float_id']}_{row['cycle_number']}_{row['profile_date'][:10]}"
                    self._existing_profile_signatures_cache.add(signature)
                logger.info(f"Loaded {len(self._existing_profile_signatures_cache)} existing profile signatures into cache")
            except Exception as e:
                logger.warning(f"Error loading existing profile signatures: {e}")
                self._existing_profile_signatures_cache = set()
        return self._existing_profile_signatures_cache

    def check_for_duplicate_floats(self, new_floats_data):
        """Check for duplicate floats and filter them out using cached data"""
        if not new_floats_data:
            return []

        existing_float_ids = self.load_existing_float_ids()

        filtered_floats = []
        for float_data in new_floats_data:
            float_id = str(float_data['float_id']).strip()
            if float_id not in existing_float_ids:
                filtered_floats.append(float_data)
                # Add to cache to avoid duplicates within this batch
                existing_float_ids.add(float_id)
            else:
                logger.debug(f"Skipping duplicate float {float_id}")

        if len(new_floats_data) - len(filtered_floats) > 0:
            logger.info(f"Filtered {len(new_floats_data) - len(filtered_floats)} duplicate floats")
        return filtered_floats

    def check_for_duplicate_profiles(self, new_profiles_data):
        """Check for duplicate profiles and filter them out using cached data"""
        if not new_profiles_data:
            return []

        existing_signatures = self.load_existing_profile_signatures()

        filtered_profiles = []
        for profile_data in new_profiles_data:
            signature = f"{profile_data['float_id']}_{profile_data['cycle_number']}_{profile_data['profile_date'][:10]}"
            if signature not in existing_signatures:
                filtered_profiles.append(profile_data)
                # Add to cache to avoid duplicates within this batch
                existing_signatures.add(signature)
            else:
                logger.debug(f"Skipping duplicate profile {signature}")

        if len(new_profiles_data) - len(filtered_profiles) > 0:
            logger.info(f"Filtered {len(new_profiles_data) - len(filtered_profiles)} duplicate profiles")
        return filtered_profiles

    def quick_check_for_new_data(self, region_name, region_bounds, year_period):
        """Quick check to see if region/year would likely contain new data without full download"""
        try:
            # Get a small sample of data to check float IDs
            fetcher = argopy.DataFetcher()
            box = [
                region_bounds['lon_min'], region_bounds['lon_max'],
                region_bounds['lat_min'], region_bounds['lat_max'],
                0, 100,  # Only surface data for quick check
                year_period['start_date'], year_period['end_date']
            ]

            # Set a small limit for the quick check
            fetcher = fetcher.region(box)
            ds = fetcher.load().data

            if 'N_POINTS' in ds.dims and len(ds.N_POINTS) > 0:
                # Get unique platform numbers from the sample
                sample_platforms = np.unique(ds.PLATFORM_NUMBER.values)
                existing_float_ids = self.load_existing_float_ids()

                # Check if any of these floats are new
                new_floats = []
                for platform in sample_platforms:
                    float_id = str(int(platform))
                    if float_id not in existing_float_ids:
                        new_floats.append(float_id)

                if new_floats:
                    logger.info(f"Quick check found {len(new_floats)} potentially new floats in {region_name} {year_period['year']}")
                    return True, len(new_floats)
                else:
                    logger.info(f"Quick check: all floats in {region_name} {year_period['year']} already exist, skipping full download")
                    return False, 0

            return True, 0  # No data found, but don't skip - might be data at deeper levels

        except Exception as e:
            # If quick check fails, proceed with full download
            logger.warning(f"Quick check failed for {region_name} {year_period['year']}: {e}")
            return True, 0

    def extract_region_data(self, region_name, region_bounds, year_period, skip_quick_check=False):
        """Extract data for a specific region and time period using argopy"""

        year = year_period['year']

        # Check if this combination has already been extracted
        if self.is_combination_extracted(region_name, year):
            logger.info(f"Skipping {region_name} {year} - already extracted")
            return None

        # Check if this combination failed recently
        if self.is_combination_failed_recently(region_name, year, max_age_days=1):  # Retry after 1 day
            logger.info(f"Skipping {region_name} {year} - failed recently, will retry later")
            return None

        # Quick check to see if this region/year would contain new data
        if not skip_quick_check:
            should_download, estimated_new_floats = self.quick_check_for_new_data(region_name, region_bounds, year_period)
            if not should_download:
                # Mark as extracted since we confirmed no new data
                self.mark_combination_extracted(region_name, year, {"floats": 0, "profiles": 0, "measurements": 0})
                return None

        logger.info(f"Extracting {region_name} data for {year}")

        try:
            # Create region fetcher with timeout and error handling
            fetcher = argopy.DataFetcher()

            # Set region boundaries
            box = [
                region_bounds['lon_min'], region_bounds['lon_max'],  # longitude range
                region_bounds['lat_min'], region_bounds['lat_max'],  # latitude range
                0, 2000,  # pressure range (surface to 2000 dbar)
                year_period['start_date'], year_period['end_date']  # time range
            ]

            logger.info(f"  Region box: {box}")

            # Fetch data with timeout
            fetcher = fetcher.region(box)

            # Load the data with memory management
            ds = fetcher.load().data

            # Check what dimensions are available
            if 'N_POINTS' in ds.dims:
                point_count = len(ds.N_POINTS)
                logger.info(f"  Found {point_count} data points")

                # Check for memory issues with large datasets
                if point_count > 500000:  # Limit to 500K points to avoid memory issues
                    logger.warning(f"  Large dataset ({point_count} points), subsampling to avoid memory issues")
                    # Subsample the data to avoid memory allocation errors
                    step = max(1, point_count // 500000)
                    ds = ds.isel(N_POINTS=slice(None, None, step))
                    point_count = len(ds.N_POINTS)
                    logger.info(f"  Subsampled to {point_count} data points")

            elif 'N_PROF' in ds.dims:
                point_count = len(ds.N_PROF)
                logger.info(f"  Found {point_count} profiles")
            else:
                logger.info(f"  Dataset dimensions: {list(ds.dims.keys())}")
                logger.info(f"  Dataset variables: {list(ds.data_vars.keys())}")
                point_count = len(list(ds.dims.values())[0]) if ds.dims else 0
                logger.info(f"  Found {point_count} data points")

            if point_count == 0:
                logger.info(f"  No data found for {region_name} in {year_period['year']}")
                return None

            return ds

        except MemoryError as e:
            error_msg = f"Memory error: {e}"
            logger.error(f"Memory error extracting {region_name} data for {year}: {error_msg}")
            self.mark_combination_failed(region_name, year, error_msg)
            return None
        except Exception as e:
            error_msg = str(e) if str(e) else f"Unknown error type: {type(e).__name__}"
            logger.error(f"Error extracting {region_name} data for {year}: {error_msg}")
            self.mark_combination_failed(region_name, year, error_msg)
            return None

    def process_argo_dataset(self, ds, region_name, year):
        """Process argopy point-based dataset and convert to our schema format"""

        logger.info(f"Processing point-based dataset: {len(ds.N_POINTS)} data points")

        floats_data = []
        profiles_data = []
        measurements_data = []

        measurement_id_counter = 1
        profile_id_counter = 1

        # Get unique floats and cycles
        unique_platforms = np.unique(ds.PLATFORM_NUMBER.values)

        for platform_number in unique_platforms:
            try:
                float_str = str(int(platform_number))

                # Get data for this float with memory-safe approach
                float_mask = ds.PLATFORM_NUMBER == platform_number
                float_data = ds.where(float_mask, drop=True)

                # Check for memory issues
                if hasattr(float_data, 'N_POINTS') and len(float_data.N_POINTS) > 100000:
                    logger.warning(f"Large float dataset for {float_str} ({len(float_data.N_POINTS)} points), subsampling")
                    step = max(1, len(float_data.N_POINTS) // 50000)  # Limit to 50K points per float
                    float_data = float_data.isel(N_POINTS=slice(None, None, step))

                # Get unique cycles for this float
                unique_cycles = np.unique(float_data.CYCLE_NUMBER.values)

                # Create float record
                float_record = {
                    'float_id': float_str,
                    'wmo_number': int(platform_number),
                    'program_name': 'ARGO-GLOBAL',
                    'platform_type': 'ARGO_FLOAT',
                    'data_assembly_center': 'delayed_mode',
                    'deployment_date': None,
                    'deployment_latitude': None,
                    'deployment_longitude': None,
                    'deployment_depth': None,
                    'current_status': 'DELAYED_MODE',
                    'last_latitude': None,
                    'last_longitude': None,
                    'last_update': datetime.now().isoformat(),
                    'cycle_time_days': 10,
                    'park_pressure_dbar': None,
                    'profile_pressure_dbar': None,
                    'total_profiles': 0,
                    'quality_profiles': 0,
                    'metadata_text': f"ARGO Float {float_str} - Delayed mode data from {region_name} {year}",
                    'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }

                # Process each cycle as a profile
                for cycle_number in unique_cycles:
                    try:
                        # Get data for this cycle
                        cycle_mask = (float_data.PLATFORM_NUMBER == platform_number) & (float_data.CYCLE_NUMBER == cycle_number)
                        cycle_data = float_data.where(cycle_mask, drop=True)

                        if len(cycle_data.N_POINTS) == 0:
                            continue

                        # Get profile metadata from first valid point
                        valid_points = ~pd.isna(cycle_data.LONGITUDE.values) & ~pd.isna(cycle_data.LATITUDE.values)
                        if not valid_points.any():
                            continue

                        first_valid_idx = np.where(valid_points)[0][0]

                        lat = float(cycle_data.LATITUDE.values[first_valid_idx])
                        lon = float(cycle_data.LONGITUDE.values[first_valid_idx])

                        # Create a synthetic date if TIME is available, otherwise use current time
                        try:
                            date = pd.to_datetime(cycle_data.TIME.values[first_valid_idx])
                        except:
                            date = datetime.now()

                        profile_record = {
                            'profile_id': profile_id_counter,
                            'float_id': float_str,
                            'cycle_number': int(cycle_number),
                            'profile_direction': 'A',  # Ascending
                            'profile_date': date.strftime('%Y-%m-%dT%H:%M:%S'),
                            'latitude': lat,
                            'longitude': lon,
                            'max_pressure': 0.0,
                            'num_levels': 0,
                            'vertical_sampling_scheme': None,
                            'data_mode': 'D',  # Delayed mode
                            'data_quality_flag': 1,
                            'processing_date': None,
                            'netcdf_filename': None,
                            'file_checksum': None,
                            'profile_summary': f"Profile {profile_id_counter} from Float {float_str} - {region_name}",
                            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        }

                        # Process measurements for this cycle
                        valid_measurements = 0
                        max_pres = 0.0

                        for point_idx in range(len(cycle_data.N_POINTS)):
                            try:
                                pres = cycle_data.PRES.values[point_idx]
                                temp = cycle_data.TEMP.values[point_idx]
                                psal = cycle_data.PSAL.values[point_idx]

                                # Skip invalid measurements
                                if pd.isna(pres) or pres < 0:
                                    continue

                                if pd.isna(temp) and pd.isna(psal):
                                    continue

                                # Calculate depth from pressure (approximation)
                                depth = float(pres) * 1.019716  # rough conversion

                                measurement_record = {
                                    'measurement_id': measurement_id_counter,
                                    'profile_id': profile_id_counter,
                                    'pressure': float(pres),
                                    'depth': depth,
                                    'pressure_qc': 1,
                                    'temperature': float(temp) if not pd.isna(temp) else None,
                                    'temperature_qc': 1,
                                    'salinity': float(psal) if not pd.isna(psal) else None,
                                    'salinity_qc': 1,
                                    'dissolved_oxygen': None,
                                    'dissolved_oxygen_qc': 1,
                                    'ph_in_situ': None,
                                    'ph_in_situ_qc': 1,
                                    'chlorophyll_a': None,
                                    'chlorophyll_a_qc': 1,
                                    'particle_backscattering': None,
                                    'particle_backscattering_qc': 1,
                                    'downward_irradiance': None,
                                    'downward_irradiance_qc': 1,
                                    'potential_temperature': None,
                                    'potential_density': None,
                                    'buoyancy_frequency': None,
                                    'mixed_layer_depth': None,
                                    'processing_level': 'D',  # Delayed mode
                                    'interpolated': 0,
                                    'spike_test_flag': 1,
                                    'gradient_test_flag': 1,
                                    'parameter_summary': f"Depth {depth:.0f}m ({pres:.1f}dbar): T:{temp:.2f}°C, S:{psal:.2f}PSU" if not pd.isna(temp) and not pd.isna(psal) else "",
                                    'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                }

                                measurements_data.append(measurement_record)
                                measurement_id_counter += 1
                                valid_measurements += 1
                                max_pres = max(max_pres, float(pres))

                            except Exception as e:
                                error_msg = str(e) if str(e) else f"Unknown error type: {type(e).__name__}"
                                logger.warning(f"Error processing point {point_idx}: {error_msg}")
                                continue

                        # Update profile with measurement stats
                        profile_record['max_pressure'] = max_pres
                        profile_record['num_levels'] = valid_measurements

                        if valid_measurements > 0:
                            profiles_data.append(profile_record)
                            profile_id_counter += 1

                    except MemoryError as e:
                        logger.warning(f"Memory error processing cycle {cycle_number}: {e}")
                        continue
                    except Exception as e:
                        error_msg = str(e) if str(e) else f"Unknown error type: {type(e).__name__}"
                        logger.warning(f"Error processing cycle {cycle_number}: {error_msg}")
                        continue

                # Update float record
                float_record['total_profiles'] = len([p for p in profiles_data if p['float_id'] == float_str])

                # Set deployment and last positions
                if profiles_data:
                    float_profiles_list = [p for p in profiles_data if p['float_id'] == float_str]
                    if float_profiles_list:
                        # Sort by date
                        float_profiles_list.sort(key=lambda x: x['profile_date'])

                        first_profile = float_profiles_list[0]
                        last_profile = float_profiles_list[-1]

                        float_record['deployment_date'] = first_profile['profile_date']
                        float_record['deployment_latitude'] = first_profile['latitude']
                        float_record['deployment_longitude'] = first_profile['longitude']
                        float_record['last_latitude'] = last_profile['latitude']
                        float_record['last_longitude'] = last_profile['longitude']

                floats_data.append(float_record)

            except MemoryError as e:
                logger.error(f"Memory error processing platform {platform_number}: {e}")
                continue
            except Exception as e:
                error_msg = str(e) if str(e) else f"Unknown error type: {type(e).__name__}"
                logger.error(f"Error processing platform {platform_number}: {error_msg}")
                continue

        logger.info(f"Processed: {len(floats_data)} floats, {len(profiles_data)} profiles, {len(measurements_data)} measurements")

        return floats_data, profiles_data, measurements_data

    def append_to_database(self, floats_data, profiles_data, measurements_data, region_name=None, year=None):
        """Append new data to existing parquet files with deduplication"""

        # Filter out duplicates before appending
        floats_data = self.check_for_duplicate_floats(floats_data)
        profiles_data = self.check_for_duplicate_profiles(profiles_data)

        if not floats_data and not profiles_data and not measurements_data:
            logger.info("No new data to append after deduplication")
            return

        try:
            # Load existing data
            existing_floats = pd.read_parquet(self.parquet_dir / "floats.parquet")
            existing_profiles = pd.read_parquet(self.parquet_dir / "profiles.parquet")
            existing_measurements = pd.read_parquet(self.parquet_dir / "measurements.parquet")

            # Convert new data to DataFrames
            if floats_data:
                new_floats_df = pd.DataFrame(floats_data)
                # Remove duplicates and append
                combined_floats = pd.concat([existing_floats, new_floats_df], ignore_index=True)
                combined_floats = combined_floats.drop_duplicates(subset=['float_id'], keep='last')
                combined_floats.to_parquet(self.parquet_dir / "floats.parquet", index=False)
                logger.info(f"Added {len(new_floats_df)} float records (total: {len(combined_floats)})")

            if profiles_data:
                new_profiles_df = pd.DataFrame(profiles_data)
                # Adjust profile IDs to be unique
                if len(existing_profiles) > 0:
                    max_profile_id = existing_profiles['profile_id'].max()
                    new_profiles_df['profile_id'] += max_profile_id

                    # Update measurement profile_ids accordingly
                    if measurements_data:
                        profile_id_mapping = {old_id: new_id for old_id, new_id in zip(
                            [p['profile_id'] - max_profile_id for p in profiles_data],
                            new_profiles_df['profile_id']
                        )}

                        for measurement in measurements_data:
                            old_prof_id = measurement['profile_id']
                            if old_prof_id in profile_id_mapping:
                                measurement['profile_id'] = profile_id_mapping[old_prof_id]

                combined_profiles = pd.concat([existing_profiles, new_profiles_df], ignore_index=True)
                combined_profiles.to_parquet(self.parquet_dir / "profiles.parquet", index=False)
                logger.info(f"Added {len(new_profiles_df)} profile records (total: {len(combined_profiles)})")

            if measurements_data:
                new_measurements_df = pd.DataFrame(measurements_data)
                # Adjust measurement IDs to be unique
                if len(existing_measurements) > 0:
                    max_measurement_id = existing_measurements['measurement_id'].max()
                    new_measurements_df['measurement_id'] += max_measurement_id

                combined_measurements = pd.concat([existing_measurements, new_measurements_df], ignore_index=True)
                combined_measurements.to_parquet(self.parquet_dir / "measurements.parquet", index=False)
                logger.info(f"Added {len(new_measurements_df)} measurement records (total: {len(combined_measurements)})")

            # Mark this combination as successfully extracted if region and year provided
            if region_name and year:
                stats = {
                    "floats": len(floats_data) if floats_data else 0,
                    "profiles": len(profiles_data) if profiles_data else 0,
                    "measurements": len(measurements_data) if measurements_data else 0
                }
                self.mark_combination_extracted(region_name, year, stats)

        except MemoryError as e:
            logger.error(f"Memory error appending to database: {e}")
        except Exception as e:
            error_msg = str(e) if str(e) else f"Unknown error type: {type(e).__name__}"
            logger.error(f"Error appending to database: {error_msg}")

    def run_incremental_update(self, months_back=6, max_regions=None):
        """Run incremental update for recent data only"""
        logger.info(f"Starting INCREMENTAL ARGO data update (last {months_back} months)")
        logger.info("=" * 60)

        # Setup database without clearing existing data
        self.setup_database(clear_existing=False)

        # Create time periods for recent months only
        current_date = datetime.now()
        # For ARGO delayed mode, data is available ~6 months behind
        delayed_cutoff = current_date - timedelta(days=180)
        # Start from the requested months back from delayed cutoff
        start_cutoff = delayed_cutoff - timedelta(days=30 * months_back)

        recent_periods = []
        for year in range(start_cutoff.year, delayed_cutoff.year + 1):
            if year == start_cutoff.year:
                start_date = start_cutoff.strftime('%Y-%m-%d')
            else:
                start_date = f"{year}-01-01"

            if year == delayed_cutoff.year:
                end_date = delayed_cutoff.strftime('%Y-%m-%d')
            else:
                end_date = f"{year}-12-31"

            # Only add if start_date <= end_date
            if start_date <= end_date:
                recent_periods.append({
                    'start_date': start_date,
                    'end_date': end_date,
                    'year': year
                })
                logger.info(f"Will check period: {start_date} to {end_date} (year {year})")
            else:
                logger.warning(f"Skipping {year} - start date {start_date} is after end date {end_date}")

        logger.info(f"Will update data for {len(recent_periods)} recent time periods")

        regions_to_process = list(self.indian_ocean_regions.items())
        if max_regions:
            regions_to_process = regions_to_process[:max_regions]

        successful_updates = 0
        skipped_updates = 0
        failed_updates = 0

        for region_name, region_bounds in regions_to_process:
            logger.info(f"\n=== INCREMENTAL UPDATE: {region_name} ===")

            for year_period in recent_periods:
                try:
                    # Use quick check to avoid unnecessary downloads
                    dataset = self.extract_region_data(region_name, region_bounds, year_period, skip_quick_check=False)

                    if dataset is None:
                        skipped_updates += 1
                        continue

                    # Process and convert to our schema
                    floats_data, profiles_data, measurements_data = self.process_argo_dataset(
                        dataset, region_name, year_period['year']
                    )

                    if floats_data or profiles_data or measurements_data:
                        # Append to database
                        self.append_to_database(floats_data, profiles_data, measurements_data, region_name, year_period['year'])
                        successful_updates += 1
                        logger.info(f"✓ Updated {region_name} {year_period['year']} with {len(floats_data)} floats, {len(profiles_data)} profiles")
                    else:
                        logger.info(f"No new data found for {region_name} {year_period['year']}")
                        skipped_updates += 1

                    # Rate limiting
                    time.sleep(2)

                    # Memory cleanup
                    import gc
                    gc.collect()

                except Exception as e:
                    error_msg = str(e) if str(e) else f"Unknown error type: {type(e).__name__}"
                    logger.error(f"Error during incremental update {region_name} {year_period['year']}: {error_msg}")
                    failed_updates += 1
                    continue

        logger.info(f"\n=== INCREMENTAL UPDATE COMPLETE ===")
        logger.info(f"Successful updates: {successful_updates}")
        logger.info(f"Skipped (no new data): {skipped_updates}")
        logger.info(f"Failed updates: {failed_updates}")

        self.print_final_stats()

    def run_extraction(self, clear_existing=True, max_regions=None, max_years=None, skip_existing=True):
        """Main extraction process"""

        logger.info("Starting ARGO data extraction using argopy (delayed-mode data)")
        logger.info("=" * 60)

        # Setup database
        self.setup_database(clear_existing=clear_existing)

        regions_to_process = list(self.indian_ocean_regions.items())
        if max_regions:
            regions_to_process = regions_to_process[:max_regions]

        years_to_process = self.time_periods
        if max_years:
            years_to_process = years_to_process[-max_years:]  # Get recent years

        total_combinations = len(regions_to_process) * len(years_to_process)

        if skip_existing:
            # Count how many combinations are already extracted
            metadata = self.load_metadata()
            existing_count = 0
            failed_count = 0

            for region_name, _ in regions_to_process:
                for year_period in years_to_process:
                    if self.is_combination_extracted(region_name, year_period['year']):
                        existing_count += 1
                    elif self.is_combination_failed_recently(region_name, year_period['year']):
                        failed_count += 1

            remaining_combinations = total_combinations - existing_count - failed_count
            logger.info(f"Total combinations: {total_combinations}")
            logger.info(f"Already extracted: {existing_count}")
            logger.info(f"Recently failed (will skip): {failed_count}")
            logger.info(f"Remaining to process: {remaining_combinations}")
        else:
            logger.info(f"Processing {len(regions_to_process)} regions x {len(years_to_process)} years = {total_combinations} combinations")

        successful_extractions = 0
        failed_extractions = 0

        for region_name, region_bounds in regions_to_process:
            logger.info(f"\n=== PROCESSING REGION: {region_name} ===")

            for year_period in years_to_process:
                try:
                    # Extract data for this region/year combination
                    # For full extraction, we can skip quick check if we're doing a comprehensive rebuild
                    skip_quick_check = clear_existing and not skip_existing
                    dataset = self.extract_region_data(region_name, region_bounds, year_period, skip_quick_check=skip_quick_check)

                    if dataset is None:
                        continue

                    # Process and convert to our schema
                    floats_data, profiles_data, measurements_data = self.process_argo_dataset(
                        dataset, region_name, year_period['year']
                    )

                    if floats_data or profiles_data or measurements_data:
                        # Append to database
                        self.append_to_database(floats_data, profiles_data, measurements_data, region_name, year_period['year'])
                        successful_extractions += 1
                    else:
                        logger.info(f"No data found for {region_name} {year_period['year']}")

                    # Rate limiting to be respectful to data servers and allow memory cleanup
                    time.sleep(3)  # Increased delay for better rate limiting

                    # Force garbage collection to free memory
                    import gc
                    gc.collect()

                except MemoryError as e:
                    error_msg = f"Memory error: {e}"
                    logger.error(f"Memory error processing {region_name} {year_period['year']}: {error_msg}")
                    self.mark_combination_failed(region_name, year_period['year'], error_msg)
                    failed_extractions += 1
                    continue
                except Exception as e:
                    error_msg = str(e) if str(e) else f"Unknown error type: {type(e).__name__}"
                    logger.error(f"Error processing {region_name} {year_period['year']}: {error_msg}")
                    self.mark_combination_failed(region_name, year_period['year'], error_msg)
                    failed_extractions += 1
                    continue

                # Progress update every 10 extractions
                total_processed = successful_extractions + failed_extractions
                if total_processed % 10 == 0:
                    logger.info(f"Progress: {total_processed} extractions processed ({successful_extractions} successful)")

        logger.info(f"\n=== EXTRACTION COMPLETE ===")
        logger.info(f"Successful: {successful_extractions}")
        logger.info(f"Failed: {failed_extractions}")

        self.print_final_stats()

    def print_final_stats(self):
        """Print final database statistics and extraction metadata"""
        try:
            floats_df = pd.read_parquet(self.parquet_dir / "floats.parquet")
            profiles_df = pd.read_parquet(self.parquet_dir / "profiles.parquet")
            measurements_df = pd.read_parquet(self.parquet_dir / "measurements.parquet")

            logger.info("=== FINAL DATABASE STATISTICS ===")
            logger.info(f"Floats: {len(floats_df):,} records")
            logger.info(f"Profiles: {len(profiles_df):,} records")
            logger.info(f"Measurements: {len(measurements_df):,} records")

            if len(profiles_df) > 0:
                date_range = f"{profiles_df['profile_date'].min()} to {profiles_df['profile_date'].max()}"
                logger.info(f"Date range: {date_range}")

                lat_range = f"{profiles_df['latitude'].min():.1f}° to {profiles_df['latitude'].max():.1f}°N"
                lon_range = f"{profiles_df['longitude'].min():.1f}° to {profiles_df['longitude'].max():.1f}°E"
                logger.info(f"Geographic coverage: {lat_range}, {lon_range}")

            # Print extraction metadata stats
            metadata = self.load_metadata()
            extracted_count = len(metadata.get("extracted_combinations", {}))
            failed_count = len(metadata.get("failed_combinations", {}))
            logger.info("\n=== EXTRACTION METADATA ====")
            logger.info(f"Successfully extracted combinations: {extracted_count}")
            logger.info(f"Failed combinations: {failed_count}")

            if failed_count > 0:
                logger.info("\nRecent failures:")
                for key, fail_info in list(metadata.get("failed_combinations", {}).items())[-5:]:  # Show last 5
                    logger.info(f"  {fail_info['region']} {fail_info['year']}: {fail_info['error'][:100]}...")

        except Exception as e:
            error_msg = str(e) if str(e) else f"Unknown error type: {type(e).__name__}"
            logger.error(f"Error generating final stats: {error_msg}")

def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='Extract ARGO data using argopy')
    parser.add_argument('--clear', action='store_true', help='Clear existing data')
    parser.add_argument('--incremental', action='store_true', help='Run incremental update for recent months only')
    parser.add_argument('--months-back', type=int, default=6, help='Months back for incremental update (default: 6)')
    parser.add_argument('--max-regions', type=int, help='Limit number of regions (for testing)')
    parser.add_argument('--max-years', type=int, help='Limit number of years (for testing)')
    parser.add_argument('--basedir', type=str, help='Base directory')
    parser.add_argument('--no-skip-existing', action='store_true', help='Do not skip existing data (re-download everything)')

    args = parser.parse_args()

    try:
        extractor = ArgopyDataExtractor(base_dir=args.basedir)

        if args.incremental:
            logger.info("Running in INCREMENTAL UPDATE mode")
            extractor.run_incremental_update(
                months_back=args.months_back,
                max_regions=args.max_regions
            )
        else:
            logger.info("Running in FULL EXTRACTION mode")
            extractor.run_extraction(
                clear_existing=args.clear,
                max_regions=args.max_regions,
                max_years=args.max_years,
                skip_existing=not args.no_skip_existing
            )
    except MemoryError as e:
        logger.error(f"Memory error during extraction: {e}")
        raise
    except Exception as e:
        error_msg = str(e) if str(e) else f"Unknown error type: {type(e).__name__}"
        logger.error(f"Extraction failed: {error_msg}")
        raise

if __name__ == "__main__":
    main()