defmodule Geostatex.GeoFrameTest do
  use ExUnit.Case, async: true

  @moduletag :geoframe

  require Explorer.DataFrame

  alias GeoFrame.GeoDataFrame, as: GDF
  alias GeoFrame.GeoSeries, as: GS
  alias Geostatex.{Kriging, Variogram}

  @csv Path.expand("../../notebooks/data/meuse/meuse.csv", __DIR__)

  setup_all do
    df = Explorer.DataFrame.from_csv!(@csv)
    df = Explorer.DataFrame.mutate(df, log_zinc: log(zinc))
    xs = Explorer.Series.to_list(df["x"]) |> Enum.map(&(&1 * 1.0))
    ys = Explorer.Series.to_list(df["y"]) |> Enum.map(&(&1 * 1.0))
    frame = GDF.new(df, GS.from_xy(xs, ys, crs: "EPSG:28992"))
    %{frame: frame, points: Enum.zip(xs, ys), values: Explorer.Series.to_list(df["log_zinc"])}
  end

  test "variograms and models from a frame are those of its points", ctx do
    v = Geostatex.GeoFrame.variogram(ctx.frame, "log_zinc", n_lags: 8)
    assert v == Variogram.new(ctx.points, ctx.values, n_lags: 8)

    k = Geostatex.GeoFrame.kriging(ctx.frame, :log_zinc, :ordinary, variogram_model: :spherical)
    direct = Kriging.ordinary(ctx.points, ctx.values, variogram_model: :spherical)
    assert k.variogram_model_parameters == direct.variogram_model_parameters

    predicted = Geostatex.GeoFrame.predict(k, ctx.frame)
    {xs, ys} = Enum.unzip(ctx.points)
    {z, s} = Kriging.execute(direct, :points, xs, ys)
    assert Explorer.Series.to_list(GDF.pull(predicted, "prediction")) == z
    assert Explorer.Series.to_list(GDF.pull(predicted, "variance")) == s

    cv =
      Geostatex.GeoFrame.cross_validate(ctx.frame, "log_zinc", :ordinary,
        variogram_model: :spherical
      )

    assert length(cv.residuals) == 155
  end

  test "grids", ctx do
    k = Geostatex.GeoFrame.kriging(ctx.frame, "log_zinc", :ordinary, variogram_model: :spherical)
    g = Geostatex.GeoFrame.grid(k, ctx.frame, nx: 10, ny: 12)
    assert GDF.n_rows(g) == 120 and GDF.crs(g) == "EPSG:28992"
    near = Geostatex.GeoFrame.grid(k, ctx.frame, cell: 100.0, max_distance: 150.0)
    full = Geostatex.GeoFrame.grid(k, ctx.frame, cell: 100.0)
    assert GDF.n_rows(near) < GDF.n_rows(full)
    assert_raise ArgumentError, fn -> Geostatex.GeoFrame.grid(k, ctx.frame, cell: -1) end
  end

  test "geographic frames", ctx do
    geo = GDF.set_crs(ctx.frame, "EPSG:4326", allow_override: true)

    assert_raise ArgumentError, ~r/geographic/, fn ->
      Geostatex.GeoFrame.variogram(geo, "log_zinc")
    end

    assert_raise ArgumentError, ~r/geographic/, fn ->
      Geostatex.GeoFrame.kriging(geo, "log_zinc", :universal)
    end

    assert %Variogram{} = Geostatex.GeoFrame.variogram(geo, "log_zinc", allow_geographic: true)
    assert_raise ArgumentError, fn -> Geostatex.GeoFrame.variogram(ctx.frame, "geometry") end

    assert_raise ArgumentError, fn ->
      Geostatex.GeoFrame.kriging(ctx.frame, "log_zinc", :simple)
    end
  end
end
