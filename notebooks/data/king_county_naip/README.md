# Aerial image chips of the King County sales

2,000 JPEG chips, `sale_<row>.jpg` (row = the `row` column of `../king_county/sales_sample.csv`,
zero-padded to 4 digits), 224 x 224 pixels covering about 120 x 120 m around each sale, fetched on
2026-10-25 by `fetch_chips.py` from the USGS The National Map `USGSNAIPImagery` image service (USDA
National Agriculture Imagery Program orthophotos, recent years). **Public domain** (works of the US
government).

The images are some years newer than the 2014–2015 sales, so a few houses may have been rebuilt.
Used by `notebooks/king_county_images_case.livemd`.
