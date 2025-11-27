#!/usr/bin/env python3
"""
Parquet Exporter for Roca Real Estate Data
Transforms XML property data into Parquet format for data analysis and ML.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from parse_roca import RocaXMLParser


class RocaParquetExporter:
    """Export Roca property data to Parquet format for analysis and ML"""
    
    def __init__(self, parser: RocaXMLParser):
        self.parser = parser
        self.properties = parser.properties
    
    def create_main_dataframe(self) -> pd.DataFrame:
        """Create main properties DataFrame with core information"""
        data = []
        
        for prop in self.properties:
            # Extract basic info
            row = {
                'property_code': prop.code,
                'property_reference': prop.reference,
                'title': prop.title,
                'description': prop.description,
                'property_type': prop.property_type,
                'property_subtype': prop.property_subtype,
            }
            
            # Extract location
            row['postal_code'] = prop.location.postal_code
            row['address'] = prop.location.address
            row['locality'] = prop.location.locality
            row['latitude'] = self._safe_float(prop.location.latitude)
            row['longitude'] = self._safe_float(prop.location.longitude)
            row['show_map'] = prop.location.show_map
            
            # Extract pricing
            sale_price = None
            rent_price = None
            for price in prop.prices:
                if price.operation == 'VENTA':
                    sale_price = float(price.amount)
                elif price.operation == 'ALQUILER':
                    rent_price = float(price.amount)
            
            row['sale_price'] = sale_price
            row['rent_price'] = rent_price
            row['has_sale_price'] = sale_price is not None
            row['has_rent_price'] = rent_price is not None
            
            # Extract key characteristics
            characteristics_dict = {c.name: c.value or c.value_id for c in prop.characteristics}
            
            row['bedrooms'] = self._safe_int(characteristics_dict.get('QUARTO'))
            row['bathrooms'] = self._safe_int(characteristics_dict.get('BANHEIRO'))
            row['suites'] = self._safe_int(characteristics_dict.get('SUITE'))
            row['parking_spaces'] = self._safe_int(characteristics_dict.get('VAGA'))
            row['area_util'] = self._safe_float(characteristics_dict.get('AREA_UTIL'))
            row['area_total'] = self._safe_float(characteristics_dict.get('AREA_TOTAL'))
            row['condominium_fee'] = self._safe_float(characteristics_dict.get('CONDOMINIO'))
            row['property_tax'] = self._safe_float(characteristics_dict.get('IPTU'))
            
            # Boolean amenities
            row['has_pool'] = characteristics_dict.get('PISCINA') == '1'
            row['has_bbq'] = characteristics_dict.get('CHURRASQUEIRA') == '1'
            row['has_gym'] = characteristics_dict.get('ACADEMIA') == '1'
            row['has_playground'] = characteristics_dict.get('PLAYGROUND') == '1'
            row['has_sauna'] = characteristics_dict.get('SAUNA') == '1'
            row['has_party_room'] = characteristics_dict.get('SALÃO_DE_FESTAS') == '1'
            row['has_sports_court'] = characteristics_dict.get('QUADRA_POLIESPORTIVA') == '1'
            row['has_24h_security'] = characteristics_dict.get('PORTARIA_24_HORAS') == '1'
            row['has_laundry'] = characteristics_dict.get('LAVANDERIA') == '1'
            row['has_closet'] = characteristics_dict.get('CLOSET') == '1'
            row['has_office'] = characteristics_dict.get('ESCRITORIO') == '1'
            row['has_pantry'] = characteristics_dict.get('DESPENSA') == '1'
            
            # Image count
            row['image_count'] = len(prop.images)
            
            # Publisher info
            row['publisher_code'] = prop.publisher_code
            row['publisher_name'] = prop.publisher_name
            row['publisher_phone'] = prop.publisher_phone
            
            # Derived features for ML
            row['price_per_sqm_sale'] = (sale_price / row['area_util'] 
                                        if sale_price and row['area_util'] else None)
            row['price_per_sqm_rent'] = (rent_price / row['area_util'] 
                                        if rent_price and row['area_util'] else None)
            
            # Total monthly cost (rent + condominium)
            row['total_monthly_cost'] = None
            if rent_price:
                condo = row['condominium_fee'] or 0
                row['total_monthly_cost'] = rent_price + condo
            
            # Property size category
            if row['area_util']:
                if row['area_util'] < 50:
                    row['size_category'] = 'small'
                elif row['area_util'] < 100:
                    row['size_category'] = 'medium'
                elif row['area_util'] < 200:
                    row['size_category'] = 'large'
                else:
                    row['size_category'] = 'extra_large'
            else:
                row['size_category'] = None
            
            # Amenity score (count of amenities)
            amenities = [
                row['has_pool'], row['has_bbq'], row['has_gym'], 
                row['has_playground'], row['has_sauna'], row['has_party_room'],
                row['has_sports_court'], row['has_24h_security']
            ]
            row['amenity_score'] = sum(amenities)
            
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Set proper data types
        df = self._optimize_dtypes(df)
        
        return df
    
    def create_characteristics_dataframe(self) -> pd.DataFrame:
        """Create detailed characteristics DataFrame (normalized)"""
        data = []
        
        for prop in self.properties:
            for char in prop.characteristics:
                row = {
                    'property_code': prop.code,
                    'characteristic_id': char.id,
                    'characteristic_name': char.name,
                    'value': char.value,
                    'value_id': char.value_id
                }
                data.append(row)
        
        return pd.DataFrame(data)
    
    def create_images_dataframe(self) -> pd.DataFrame:
        """Create images DataFrame"""
        data = []
        
        for prop in self.properties:
            for idx, img in enumerate(prop.images):
                row = {
                    'property_code': prop.code,
                    'image_sequence': idx + 1,
                    'image_url': img.url
                }
                data.append(row)
        
        return pd.DataFrame(data)
    
    def create_prices_dataframe(self) -> pd.DataFrame:
        """Create prices DataFrame"""
        data = []
        
        for prop in self.properties:
            for price in prop.prices:
                row = {
                    'property_code': prop.code,
                    'amount': float(price.amount),
                    'currency': price.currency,
                    'operation': price.operation,
                    'operation_type': 'sale' if price.operation == 'VENTA' else 'rent'
                }
                data.append(row)
        
        return pd.DataFrame(data)
    
    def create_ml_features_dataframe(self) -> pd.DataFrame:
        """Create DataFrame optimized for ML with encoded categorical variables"""
        df = self.create_main_dataframe()
        
        # One-hot encode property type
        property_type_dummies = pd.get_dummies(df['property_type'], prefix='type')
        
        # One-hot encode property subtype
        property_subtype_dummies = pd.get_dummies(df['property_subtype'], prefix='subtype')
        
        # One-hot encode size category
        size_category_dummies = pd.get_dummies(df['size_category'], prefix='size')
        
        # Combine with original features
        ml_df = pd.concat([df, property_type_dummies, property_subtype_dummies, 
                          size_category_dummies], axis=1)
        
        # Create location-based features
        ml_df['has_coordinates'] = (~ml_df['latitude'].isna() & 
                                    ~ml_df['longitude'].isna())
        
        # Parse locality for city extraction
        ml_df['city'] = ml_df['locality'].str.split(',').str[1].str.strip()
        
        # City encoding (top cities)
        city_counts = ml_df['city'].value_counts()
        top_cities = city_counts.head(10).index.tolist()
        ml_df['city_encoded'] = ml_df['city'].apply(
            lambda x: x if x in top_cities else 'other'
        )
        city_dummies = pd.get_dummies(ml_df['city_encoded'], prefix='city')
        ml_df = pd.concat([ml_df, city_dummies], axis=1)
        
        return ml_df
    
    def export_to_parquet(self, output_dir: str = '.', 
                         include_normalized: bool = True,
                         include_ml: bool = True) -> Dict[str, str]:
        """
        Export all DataFrames to Parquet files
        
        Args:
            output_dir: Directory to save Parquet files
            include_normalized: Include normalized tables (characteristics, images, prices)
            include_ml: Include ML-optimized DataFrame
            
        Returns:
            Dictionary mapping table name to file path
        """
        import os
        
        output_files = {}
        
        # Main properties table
        print("Creating main properties DataFrame...")
        main_df = self.create_main_dataframe()
        main_path = os.path.join(output_dir, 'properties_main.parquet')
        main_df.to_parquet(main_path, engine='pyarrow', compression='snappy', index=False)
        output_files['main'] = main_path
        print(f"✓ Saved {len(main_df)} rows to {main_path}")
        
        if include_normalized:
            # Characteristics table
            print("Creating characteristics DataFrame...")
            char_df = self.create_characteristics_dataframe()
            char_path = os.path.join(output_dir, 'properties_characteristics.parquet')
            char_df.to_parquet(char_path, engine='pyarrow', compression='snappy', index=False)
            output_files['characteristics'] = char_path
            print(f"✓ Saved {len(char_df)} rows to {char_path}")
            
            # Images table
            print("Creating images DataFrame...")
            img_df = self.create_images_dataframe()
            img_path = os.path.join(output_dir, 'properties_images.parquet')
            img_df.to_parquet(img_path, engine='pyarrow', compression='snappy', index=False)
            output_files['images'] = img_path
            print(f"✓ Saved {len(img_df)} rows to {img_path}")
            
            # Prices table
            print("Creating prices DataFrame...")
            price_df = self.create_prices_dataframe()
            price_path = os.path.join(output_dir, 'properties_prices.parquet')
            price_df.to_parquet(price_path, engine='pyarrow', compression='snappy', index=False)
            output_files['prices'] = price_path
            print(f"✓ Saved {len(price_df)} rows to {price_path}")
        
        if include_ml:
            # ML-optimized table
            print("Creating ML-optimized DataFrame...")
            ml_df = self.create_ml_features_dataframe()
            ml_path = os.path.join(output_dir, 'properties_ml_features.parquet')
            ml_df.to_parquet(ml_path, engine='pyarrow', compression='snappy', index=False)
            output_files['ml_features'] = ml_path
            print(f"✓ Saved {len(ml_df)} rows with {len(ml_df.columns)} features to {ml_path}")
        
        return output_files
    
    def _safe_int(self, value: Any) -> int | None:
        """Safely convert value to int"""
        if value is None or value == '':
            return None
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return None
    
    def _safe_float(self, value: Any) -> float | None:
        """Safely convert value to float"""
        if value is None or value == '':
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame data types for efficiency"""
        # Convert boolean columns
        bool_cols = [col for col in df.columns if col.startswith('has_')]
        for col in bool_cols:
            df[col] = df[col].astype('boolean')
        
        # Convert int columns with NaN support
        int_cols = ['bedrooms', 'bathrooms', 'suites', 'parking_spaces', 
                   'image_count', 'amenity_score']
        for col in int_cols:
            if col in df.columns:
                df[col] = df[col].astype('Int64')  # Nullable integer
        
        # Convert float columns
        float_cols = ['latitude', 'longitude', 'sale_price', 'rent_price',
                     'area_util', 'area_total', 'condominium_fee', 'property_tax',
                     'price_per_sqm_sale', 'price_per_sqm_rent', 'total_monthly_cost']
        for col in float_cols:
            if col in df.columns:
                df[col] = df[col].astype('float64')
        
        # Convert category columns
        cat_cols = ['property_type', 'property_subtype', 'size_category', 
                   'show_map', 'publisher_code']
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].astype('category')
        
        return df
    
    def get_dataframe_info(self) -> Dict[str, Any]:
        """Get information about the created DataFrames"""
        main_df = self.create_main_dataframe()
        
        return {
            'total_properties': len(main_df),
            'total_columns': len(main_df.columns),
            'memory_usage_mb': main_df.memory_usage(deep=True).sum() / 1024 / 1024,
            'null_percentages': (main_df.isnull().sum() / len(main_df) * 100).to_dict(),
            'column_types': main_df.dtypes.astype(str).to_dict(),
            'numeric_columns': main_df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': main_df.select_dtypes(include=['category']).columns.tolist(),
            'boolean_columns': main_df.select_dtypes(include=['boolean']).columns.tolist(),
        }


def main():
    """Main function to demonstrate Parquet export"""
    print("=" * 70)
    print("Roca XML to Parquet Converter")
    print("=" * 70)
    
    # Parse XML
    print("\n[1] Parsing XML file...")
    parser = RocaXMLParser('roca.xml')
    properties = parser.parse()
    print(f"✓ Loaded {len(properties)} properties")
    
    # Create exporter
    print("\n[2] Creating Parquet exporter...")
    exporter = RocaParquetExporter(parser)
    print("✓ Exporter initialized")
    
    # Get DataFrame info
    print("\n[3] Analyzing DataFrame structure...")
    info = exporter.get_dataframe_info()
    print(f"✓ Total properties: {info['total_properties']}")
    print(f"✓ Total columns: {info['total_columns']}")
    print(f"✓ Memory usage: {info['memory_usage_mb']:.2f} MB")
    print(f"✓ Numeric columns: {len(info['numeric_columns'])}")
    print(f"✓ Categorical columns: {len(info['categorical_columns'])}")
    print(f"✓ Boolean columns: {len(info['boolean_columns'])}")
    
    # Export to Parquet
    print("\n[4] Exporting to Parquet files...")
    output_files = exporter.export_to_parquet(
        output_dir='.',
        include_normalized=True,
        include_ml=True
    )
    
    print("\n" + "=" * 70)
    print("Export Summary")
    print("=" * 70)
    for table_name, file_path in output_files.items():
        import os
        size_mb = os.path.getsize(file_path) / 1024 / 1024
        print(f"✓ {table_name}: {file_path} ({size_mb:.2f} MB)")
    
    print("\n" + "=" * 70)
    print("Export completed successfully!")
    print("=" * 70)
    
    # Show sample data
    print("\nSample of main DataFrame (first 5 rows, key columns):")
    main_df = exporter.create_main_dataframe()
    sample_cols = ['property_code', 'property_type', 'bedrooms', 'bathrooms', 
                   'area_util', 'sale_price', 'rent_price', 'amenity_score']
    print(main_df[sample_cols].head())


if __name__ == '__main__':
    main()
