# Data cleaning pipeline available as a script
import pandas as pd
from unidecode import unidecode


def clean():
    # Load data
    df_main_raw = pd.read_parquet("data/properties_main.parquet", engine="pyarrow", dtype_backend="numpy_nullable")

    # Split locality data
    df_main = df_main_raw.copy()
    df_main["city"] = df_main_raw["locality"].apply(lambda row: row.split(",")[1])
    df_main["neighborhood"] = df_main_raw["locality"].apply(lambda row: unidecode(row.split(",")[0]).upper().strip())
    df_main = df_main.drop(columns=["locality"])

    # Filter sao carlos properties
    sao_carlos = df_main[df_main["city"] == "São Carlos"]

    # Column selection
    selected = (sao_carlos
    .drop(columns=[
        "property_reference",
        "title",
        'description',
        'postal_code',
        'address',
        'latitude',
        'longitude',  # A more complex model may use it to calculate distance to uptown, will not be used at first
        'city',
        'neighborhood',  # Using neighborhoods may introduce too much sparsity in the model (over 200 neighborhoods)
        'show_map',
        'has_sale_price',
        'has_rent_price',  # Already visible by NaNs
        'image_count',
        'publisher_code',
        'publisher_name',
        'publisher_phone',
        'price_per_sqm_rent',
        'price_per_sqm_sale'
    ]))

    # Select residencial properties
    residencial_df = selected[selected['property_type'].isin(["Casa", "Apartamento"])]

    # Based on the EDA notebook, we shall remove nonsense outliers from the area_util column
    residencial_df = residencial_df[residencial_df['area_util'] > 10]


    # Cleaning null values
    clean_data = residencial_df.copy()
    clean_data = clean_data.dropna(
        subset=['bathrooms', 'bedrooms', 'area_total', 'area_util', 'size_category', 'parking_spaces'])
    clean_data["condominium_fee"] = clean_data["condominium_fee"].fillna(0)
    clean_data = clean_data.drop(columns=['total_monthly_cost', 'suites', 'property_tax'])
    clean_data_sell = clean_data[~clean_data["sale_price"].isnull()].drop(columns=["rent_price"])
    clean_data_rent = clean_data[~clean_data["rent_price"].isnull()].drop(columns=["sale_price"])


    # save to csv
    clean_data_rent.to_csv("data/clean_data_rent.csv", index=False)
    clean_data_sell.to_csv("data/clean_data_sell.csv", index=False)

if __name__ == "__main__":
    clean()