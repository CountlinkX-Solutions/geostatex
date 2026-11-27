# GeoFrame is a private sibling project and an optional test-only dependency (see mix.exs);
# its integration tests run only when GEOSTATEX_GEOFRAME_PATH points at a checkout.
unless Code.ensure_loaded?(GeoFrame.GeoDataFrame), do: ExUnit.configure(exclude: [:geoframe])

ExUnit.start()
