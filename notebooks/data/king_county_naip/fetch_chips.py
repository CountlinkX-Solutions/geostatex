"""Fetch a USGS NAIP aerial image chip around every sampled King County sale (dev-time only).

Run from the repository root, with pyproj available:

    python notebooks/data/king_county_naip/fetch_chips.py

Each chip is 224 x 224 pixels covering about 120 x 120 m on the ground, centred on the sale's
latitude and longitude (../king_county/sales_sample.csv). Chips are written as sale_<row>.jpg
(row = the `row` column of the sample, zero-padded to 4 digits). Existing files are kept; at most
4 requests run at a time.

Source: USGS The National Map, USGSNAIPImagery image service (USDA National Agriculture Imagery
Program), public domain.
"""

import csv
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyproj

HERE = Path(__file__).resolve().parent
SERVICE = "https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPImagery/ImageServer/exportImage"
HALF_SIDE_M = 60.0
SIZE = 224
TO_MERCATOR = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)


def fetch(url, attempts=6):
    # The service answers an occasional 502/503 under load: retry with a growing pause.
    request = urllib.request.Request(url, headers={"User-Agent": "geostatex-notebook/0.1 (CountlinkX Solutions)"})
    for attempt in range(attempts):
        try:
            return urllib.request.urlopen(request, timeout=60).read()
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
            if isinstance(error, urllib.error.HTTPError) and error.code < 500:
                raise
            if attempt == attempts - 1:
                raise
            time.sleep(2 * (attempt + 1))


def chip(sale):
    row, lat, lon = int(sale["row"]), float(sale["lat"]), float(sale["long"])
    out = HERE / f"sale_{row:04d}.jpg"
    if out.exists():
        return
    mx, my = TO_MERCATOR.transform(lon, lat)
    half = HALF_SIDE_M / math.cos(math.radians(lat))  # Web Mercator stretches by 1 / cos(lat)
    query = {
        "bbox": f"{mx - half},{my - half},{mx + half},{my + half}",
        "bboxSR": 3857,
        "imageSR": 3857,
        "size": f"{SIZE},{SIZE}",
        "format": "jpg",
        "f": "image",
    }
    data = fetch(SERVICE + "?" + urllib.parse.urlencode(query))
    if not data.startswith(b"\xff\xd8"):
        raise RuntimeError(f"sale {row}: not a JPEG: {data[:120]!r}")
    out.write_bytes(data)


def main():
    with open(HERE.parent / "king_county" / "sales_sample.csv") as f:
        sales = list(csv.DictReader(f))
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(chip, sales))
    print(len(list(HERE.glob("sale_*.jpg"))), "chips")


if __name__ == "__main__":
    main()
