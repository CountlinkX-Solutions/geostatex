# King County house sales, 2014–2015 (sample)

2,000 house sales drawn at random (Python `random.seed(2014)`) from the 21,613 sales in King County,
Washington (Seattle and surroundings) between May 2014 and May 2015, as published on OpenML
(dataset 42092, `house_sales`; originally the Kaggle "House Sales in King County, USA" data by
harlfoxem). **CC0 1.0** (public domain dedication).

`row` (0–1999) numbers the sample and names the image chip of each sale in `../king_county_naip`;
the other columns are the original ones. Coordinates are WGS84 latitude and longitude. Used by
`notebooks/king_county_images_case.livemd`.

| Column | Meaning |
|---|---|
| `id`, `date` | sale identifier and date |
| `price` | sale price, dollars |
| `bedrooms`, `bathrooms`, `floors` | bedrooms, bathrooms, floors |
| `sqft_living`, `sqft_lot`, `sqft_above`, `sqft_basement` | living area, lot, above ground and basement, square feet |
| `waterfront`, `view` | waterfront indicator, view rating (0–4) |
| `condition`, `grade` | condition (1–5) and construction grade (1–13) |
| `yr_built`, `yr_renovated` | year built, year renovated (0 if never) |
| `zipcode`, `lat`, `long` | ZIP code and location |
| `sqft_living15`, `sqft_lot15` | mean living area and lot of the 15 nearest neighbours |
