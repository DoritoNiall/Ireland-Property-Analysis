"""
Merge PPR residential sales and AirBnb listings by County for DkIT analysis.
"""

from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
PPR_PATH = BASE / "PPR_2025_Cleaned.csv"
# Cleaned AirBnb export in the Major Project Python Analysis folder
AIRBNB_PATH = BASE / "Airbnb_transformed_dataset final.csv"

RESIDENTIAL_DESC = "Second-Hand Dwelling house /Apartment"

# PPR uses traditional county names; AirBnb often stores Local Authority (LA) labels.
# Map normalized LA strings to the county name used in PPR so the join keys align.
IRISH_COUNTIES = frozenset(
    {
        "Carlow",
        "Cavan",
        "Clare",
        "Cork",
        "Donegal",
        "Dublin",
        "Galway",
        "Kerry",
        "Kildare",
        "Kilkenny",
        "Laois",
        "Leitrim",
        "Limerick",
        "Longford",
        "Louth",
        "Mayo",
        "Meath",
        "Monaghan",
        "Offaly",
        "Roscommon",
        "Sligo",
        "Tipperary",
        "Waterford",
        "Westmeath",
        "Wexford",
        "Wicklow",
    }
)

LOCAL_AUTHORITY_TO_COUNTY: dict[str, str] = {
    "Mayo County Council": "Mayo",
    "Clare County Council": "Clare",
    "Galway County Council": "Galway",
    "Kilkenny County Council": "Kilkenny",
    "Dun Laoghaire-Rathdown County Council": "Dublin",
    "Sligo County Council": "Sligo",
    "Cavan County Council": "Cavan",
    "Cork County Council": "Cork",
    "Roscommon County Council": "Roscommon",
    "Wicklow County Council": "Wicklow",
    "Kerry County Council": "Kerry",
    "Dublin City Council": "Dublin",
    "Donegal County Council": "Donegal",
    "Limerick City And County Council": "Limerick",
    "Carlow County Council": "Carlow",
    "South Dublin County Council": "Dublin",
    "Kildare County Council": "Kildare",
    "Tipperary County Council": "Tipperary",
    "Galway City Council": "Galway",
    "Meath County Council": "Meath",
    "Longford County Council": "Longford",
    "Westmeath County Council": "Westmeath",
    "Offaly County Council": "Offaly",
    "Wexford County Council": "Wexford",
    "Monaghan County Council": "Monaghan",
    "Waterford City And County Council": "Waterford",
    "Fingal County Council": "Dublin",
    "Cork City Council": "Cork",
    "Leitrim County Council": "Leitrim",
    "Louth County Council": "Louth",
    "Laois County Council": "Laois",
}


def standardize_county(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.title()


def align_airbnb_county_to_ppr(series: pd.Series) -> pd.Series:
    """Map LA labels to PPR county names; pass through values already in IRISH_COUNTIES."""

    def one_cell(raw: object):
        if pd.isna(raw) or str(raw).strip() == "" or str(raw).lower() == "nan":
            return float("nan")
        s = str(raw).strip()
        t = " ".join(s.split()).title()
        if t in IRISH_COUNTIES:
            return t
        if t in LOCAL_AUTHORITY_TO_COUNTY:
            return LOCAL_AUTHORITY_TO_COUNTY[t]
        return t

    return series.map(one_cell)


def build_county_stats(ppr_res: pd.DataFrame, airbnb: pd.DataFrame) -> pd.DataFrame:
    """Aggregate PPR and AirBnb by County and add Gross_STR_Yield_Pct."""
    ppr_agg = (
        ppr_res.groupby("County", as_index=False)
        .agg(
            median_final_price=("Final_Price", "median"),
            mean_final_price=("Final_Price", "mean"),
        )
    )

    airbnb_agg = (
        airbnb.groupby("County", as_index=False)
        .agg(
            mean_estimated_revenue_l365d=("estimated_revenue_l365d", "mean"),
            airbnb_listing_count=("id", "count"),
        )
    )

    county_stats = ppr_agg.merge(airbnb_agg, on="County", how="inner")

    county_stats["Gross_STR_Yield_Pct"] = (
        county_stats["mean_estimated_revenue_l365d"]
        / county_stats["median_final_price"].replace(0, float("nan"))
        * 100
    )

    return county_stats


def prepare_analysis_data():
    """
    Load CSVs, apply filters, align counties, and build county_stats (incl. Gross_STR_Yield_Pct).
    Returns (ppr_res, airbnb, county_stats) for reuse in visualizations and reporting.
    """
    ppr = pd.read_csv(PPR_PATH)
    airbnb = pd.read_csv(AIRBNB_PATH, low_memory=False)

    ppr["County"] = standardize_county(ppr["County"])
    airbnb["County"] = align_airbnb_county_to_ppr(airbnb["County"])
    airbnb = airbnb.dropna(subset=["County"])

    mask_residential = ppr["Description of Property"] == RESIDENTIAL_DESC
    ppr_res = ppr.loc[mask_residential].copy()

    county_stats = build_county_stats(ppr_res, airbnb)

    return ppr_res, airbnb, county_stats


def main() -> None:
    ppr = pd.read_csv(PPR_PATH)
    airbnb = pd.read_csv(AIRBNB_PATH, low_memory=False)

    print("=== Initial load (data cleaning rigor) ===")
    print(f"PPR shape (rows, columns): {ppr.shape}")
    print(f"AirBnb shape (rows, columns): {airbnb.shape}\n")

    ppr["County"] = standardize_county(ppr["County"])
    airbnb_before_align = len(airbnb)
    airbnb["County"] = align_airbnb_county_to_ppr(airbnb["County"])
    airbnb = airbnb.dropna(subset=["County"])
    print("=== AirBnb: after county alignment (rows with a mappable County) ===")
    print(f"AirBnb shape: {airbnb.shape} (dropped {airbnb_before_align - len(airbnb)} rows with missing County)\n")

    print("=== PPR: before residential filter ===")
    print(f"PPR shape: {ppr.shape}\n")

    mask_residential = ppr["Description of Property"] == RESIDENTIAL_DESC
    ppr_res = ppr.loc[mask_residential].copy()

    print("=== PPR: after filter (Second-Hand Dwelling house /Apartment only) ===")
    print(f"PPR shape: {ppr_res.shape}")
    print(f"Rows removed: {len(ppr) - len(ppr_res)}\n")

    county_stats = build_county_stats(ppr_res, airbnb)

    print("=== county_stats (merged on County) ===")
    print(f"county_stats shape: {county_stats.shape}")
    print(county_stats.to_string(index=False))

    print("\n=== Gross STR yield — preliminary statistics (Preliminary Analysis) ===")
    yield_desc = county_stats[["Gross_STR_Yield_Pct"]].describe()
    print(yield_desc.to_string())

    national_median_price = ppr_res["Final_Price"].median()
    national_mean_price = ppr_res["Final_Price"].mean()

    ranked = county_stats.sort_values("Gross_STR_Yield_Pct", ascending=False).reset_index(
        drop=True
    )
    top3 = ranked.head(3)

    print("\n=== Yield hotspots (ranked by Gross_STR_Yield_Pct) ===")
    print(ranked[["County", "Gross_STR_Yield_Pct", "median_final_price"]].to_string(index=True))

    print("\n=== Top 3 counties by gross yield vs national second-hand benchmarks ===")
    print(
        f"National median Final_Price (all second-hand sales): €{national_median_price:,.2f}"
    )
    print(
        f"National mean Final_Price (all second-hand sales):   €{national_mean_price:,.2f}"
    )
    for _, row in top3.iterrows():
        diff = row["median_final_price"] - national_median_price
        print(
            f"  {row['County']}: median €{row['median_final_price']:,.0f} "
            f"({diff:+,.0f} vs national median); "
            f"Gross_STR_Yield_Pct = {row['Gross_STR_Yield_Pct']:.4f}%"
        )

    room_mask = airbnb["room_type"].isin(["Entire home/apt", "Private room"])
    room_by_county = (
        airbnb.loc[room_mask]
        .groupby(["County", "room_type"], as_index=False)
        .agg(
            mean_price=("price", "mean"),
            mean_occupancy_rate_365d=("Occupancy_Rate_365d", "mean"),
            listings=("id", "count"),
        )
        .sort_values(["County", "room_type"])
    )

    print("\n=== Room types: mean price & Occupancy_Rate_365d (Entire home/apt vs Private room) ===")
    print(room_by_county.to_string(index=False))

    # Revenue_Density is listing-level; county land area is not in this dataset.
    rev_den = (
        airbnb.dropna(subset=["Revenue_Density"])
        .groupby("County", as_index=False)
        .agg(
            mean_revenue_density=("Revenue_Density", "mean"),
            listings_with_revenue_density=("id", "count"),
        )
    )
    rev_den = rev_den.merge(
        county_stats[
            ["County", "airbnb_listing_count", "mean_estimated_revenue_l365d"]
        ],
        on="County",
        how="left",
    )
    rev_den = rev_den.sort_values("mean_revenue_density", ascending=False)

    print("\n=== Revenue density by county (mean Revenue_Density per listing; listing count shown) ===")
    print(
        "Note: No county area (km²) column in the AirBnb file; rank uses mean Revenue_Density "
        "(listing-level metric in the scrape) and total listings in the county. "
        "Mean Revenue_Density uses only rows where Revenue_Density is non-null; "
        "mean_estimated_revenue_l365d is the county mean from all listings (county_stats).\n"
    )
    print(rev_den.to_string(index=False))


if __name__ == "__main__":
    main()
