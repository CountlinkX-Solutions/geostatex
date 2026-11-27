# Aerial image chips of the Baltimore sales

211 JPEG chips, `sale_<row>.jpg` (row = the row of the sale in `../baltimore/baltim.shp`), 224 x
224 pixels covering about 150 x 150 m around each sale, fetched on 2026-10-25 by `fetch_chips.py`
from the USGS The National Map `USGSNAIPImagery` image service (USDA National Agriculture Imagery
Program orthophotos, recent years). **Public domain** (works of the US government).

The sale coordinates are taken as NAD27 / Maryland (EPSG:26785) in thousands of US feet, which
places them in Baltimore City and County. The images are decades newer than the 1978 sales. Used
by `notebooks/baltimore_satellite_case.livemd` and `notebooks/es/caso_satelite_baltimore.livemd`.

`sale_114.jpg` is a uniform patch with nothing recognisable (water, or a gap in the imagery at
the given location); it is kept so that every sale has a chip.
