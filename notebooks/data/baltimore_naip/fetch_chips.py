"""Fetch a USGS NAIP aerial image chip around every Baltimore sale (dev-time only).

Run from the repository root, with geopandas and pyproj available:

    python notebooks/data/baltimore_naip/fetch_chips.py

Each chip is 224 x 224 pixels covering about 150 x 150 m on the ground, centred on the sale. The
sale coordinates (X, Y of ../baltimore/baltim.shp) are taken as NAD27 / Maryland (EPSG:26785) in
thousands of US feet, which places them in Baltimore City and County. Chips are written as
sale_<row>.jpg (row = the row of the sale in the shapefile). Existing files are kept.

Source: USGS The National Map, USGSNAIPImagery image service (USDA National Agriculture Imagery
Program), public domain.
"""

import math
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import geopandas as gpd
import pyproj

HERE = Path(__file__).resolve().parent
SERVICE = "https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPImagery/ImageServer/exportImage"
HALF_SIDE_M = 75.0
SIZE = 224


def fetch(request, attempts=5):
    # The service answers an occasional 502/503 under load: retry with a growing pause.
    for attempt in range(attempts):
        try:
            return urllib.request.urlopen(request, timeout=60).read()
        except urllib.error.HTTPError as error:
            if error.code < 500 or attempt == attempts - 1:
                raise
            time.sleep(2 * (attempt + 1))


def main():
    sales = gpd.read_file(HERE.parent / "baltimore" / "baltim.shp")
    to_mercator = pyproj.Transformer.from_crs("EPSG:26785", "EPSG:3857", always_xy=True)
    to_lonlat = pyproj.Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    for row, (x, y) in enumerate(zip(sales["X"], sales["Y"])):
        out = HERE / f"sale_{row:03d}.jpg"
        if out.exists():
            continue
        mx, my = to_mercator.transform(x * 1000, y * 1000)
        lat = to_lonlat.transform(mx, my)[1]
        # Web Mercator stretches distances by 1 / cos(latitude).
        half = HALF_SIDE_M / math.cos(math.radians(lat))
        query = {
            "bbox": f"{mx - half},{my - half},{mx + half},{my + half}",
            "bboxSR": 3857,
            "imageSR": 3857,
            "size": f"{SIZE},{SIZE}",
            "format": "jpg",
            "f": "image",
        }
        request = urllib.request.Request(
            SERVICE + "?" + urllib.parse.urlencode(query),
            headers={"User-Agent": "geostatex-notebook/0.1 (CountlinkX Solutions)"},
        )
        data = fetch(request)
        if not data.startswith(b"\xff\xd8"):
            raise RuntimeError(f"sale {row}: not a JPEG: {data[:120]!r}")
        out.write_bytes(data)
        time.sleep(0.3)
    print(len(list(HERE.glob("sale_*.jpg"))), "chips")


if __name__ == "__main__":
    main()
