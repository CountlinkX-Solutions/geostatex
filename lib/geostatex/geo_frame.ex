defmodule Geostatex.GeoFrame do
  @moduledoc """
  Adapter between [GeoFrame](https://github.com/CountlinkX-Solutions/geoframe) and Geostatex.

  Geostatex does not depend on GeoFrame. This module calls GeoFrame's public API (and Explorer's,
  which GeoFrame brings) at runtime, and raises an `ArgumentError` saying so when GeoFrame is not
  loaded. Add `{:geoframe, ...}` to your own dependencies to use it.

  The observations are the centroids of the geometries of a `GeoFrame.GeoDataFrame` (a point is
  its own centroid) and the values one of its numeric columns:

    * `variogram/3` - a `Geostatex.Variogram`;
    * `kriging/4` - a `Geostatex.Kriging` model (`:ordinary` or `:universal`);
    * `cross_validate/4` - `Geostatex.Kriging.cross_validate/4`;
    * `predict/3` - predictions and kriging variances at the centroids of another frame, as
      columns of that frame;
    * `grid/3` - predictions on a regular grid, as a frame of points.

  ## Geographic coordinates

  Variograms and universal kriging are Euclidean, so they raise `ArgumentError` on a geographic
  CRS unless `allow_geographic: true`. `kriging/4` with `:ordinary` instead uses great-circle
  distances on a geographic frame (`coordinates_type: :geographic`) unless told otherwise.

  ## Examples

      samples = GeoFrame.read_file!("meuse.gpkg")
      model = Geostatex.GeoFrame.kriging(samples, "log_zinc", :ordinary, variogram_model: :spherical)
      Geostatex.GeoFrame.grid(model, samples, cell: 40.0)
  """

  alias Geostatex.{Kriging, Variogram}

  @gdf GeoFrame.GeoDataFrame
  @gs GeoFrame.GeoSeries
  @df Explorer.DataFrame
  @series Explorer.Series

  @doc """
  The centroid coordinates of a frame's (or series') geometries, as `{xs, ys}` lists.
  """
  @spec points(struct()) :: {[float()], [float()]}
  def points(frame) do
    ensure_geoframe!()
    centroids = call(@gs, :centroid, [series(frame)])
    {to_list(call(@gs, :x, [centroids])), to_list(call(@gs, :y, [centroids]))}
  end

  @doc """
  The experimental variogram (`Geostatex.Variogram.new/3`) of `column` at the frame's centroids.
  Options of `Geostatex.Variogram.new/3`, plus `allow_geographic: true`.
  """
  @spec variogram(struct(), String.t() | atom(), keyword()) :: Variogram.t()
  def variogram(frame, column, opts \\ []) do
    {allow, opts} = Keyword.pop(opts, :allow_geographic, false)
    euclidean!(frame, allow, "a variogram")
    Variogram.new(points(frame), values(frame, column), opts)
  end

  @doc """
  A kriging model of `column` at the frame's centroids: `kind` is `:ordinary` (default) or
  `:universal`, with the options of `Geostatex.Kriging.ordinary/3` or `universal/3`. A geographic
  frame uses great-circle distances for ordinary kriging unless `:coordinates_type` is given.
  """
  @spec kriging(struct(), String.t() | atom(), :ordinary | :universal, keyword()) :: Kriging.t()
  def kriging(frame, column, kind \\ :ordinary, opts \\ []) do
    {build, opts} = model_options(frame, kind, opts)
    build.(points(frame), values(frame, column), opts)
  end

  @doc """
  Leave-one-out cross-validation (`Geostatex.Kriging.cross_validate/4`) of `column` at the frame's
  centroids, with the options of `kriging/4` (and `:n_closest_points`).
  """
  @spec cross_validate(struct(), String.t() | atom(), :ordinary | :universal, keyword()) :: map()
  def cross_validate(frame, column, kind \\ :ordinary, opts \\ []) do
    {_build, opts} = model_options(frame, kind, opts)
    Kriging.cross_validate(kind, points(frame), values(frame, column), opts)
  end

  @doc """
  Predicts at the centroids of `targets` (a `GeoFrame.GeoDataFrame`) and returns it with two new
  columns, `"prediction"` and `"variance"` (or `prefix <> "prediction"` with `prefix:`). Other
  options go to `Geostatex.Kriging.execute/5` (`:n_closest_points`, `:specified_drift_arrays`).
  """
  @spec predict(Kriging.t(), struct(), keyword()) :: struct()
  def predict(%Kriging{} = model, targets, opts \\ []) do
    {prefix, opts} = Keyword.pop(opts, :prefix, "")
    {xs, ys} = points(targets)
    {z, s} = Kriging.execute(model, :points, xs, ys, opts)

    targets
    |> then(&call(@gdf, :put, [&1, prefix <> "prediction", z]))
    |> then(&call(@gdf, :put, [&1, prefix <> "variance", s]))
  end

  @doc """
  Predictions on a regular grid of points, as a `GeoFrame.GeoDataFrame` with the columns `"x"`,
  `"y"`, `"prediction"` and `"variance"`. `extent` is a frame (its total bounds) or
  `{min_x, min_y, max_x, max_y}`. Options:

    * `:cell` - the grid spacing, or `:nx` and `:ny`, the number of columns and rows (default 50
      each);
    * `:max_distance` - drop the cells farther than this from every observation;
    * `:crs` - the CRS of the result (default the extent frame's);
    * other options go to `Geostatex.Kriging.execute/5` (`:n_closest_points`).
  """
  @spec grid(Kriging.t(), struct() | tuple(), keyword()) :: struct()
  def grid(%Kriging{} = model, extent, opts \\ []) do
    ensure_geoframe!()
    {grid_opts, opts} = Keyword.split(opts, [:cell, :nx, :ny, :max_distance, :crs, :observations])
    {x0, y0, x1, y1} = bounds(extent)
    {xs, ys} = axes({x0, y0, x1, y1}, grid_opts)
    cells = for y <- ys, x <- xs, do: {x, y}
    cells = keep_near(cells, grid_opts[:max_distance], grid_opts[:observations] || extent)
    {cx, cy} = Enum.unzip(cells)
    {z, s} = Kriging.execute(model, :points, cx, cy, opts)
    crs = Keyword.get_lazy(grid_opts, :crs, fn -> crs_of(extent) end)
    geometry = call(@gs, :from_xy, [cx, cy, [crs: crs]])
    data = call(@df, :new, [%{"x" => cx, "y" => cy, "prediction" => z, "variance" => s}])
    call(@gdf, :new, [data, geometry])
  end

  defp axes({x0, y0, x1, y1}, opts) do
    case opts[:cell] do
      cell when is_number(cell) and cell > 0 ->
        {steps(x0, x1, cell), steps(y0, y1, cell)}

      nil ->
        nx = Keyword.get(opts, :nx, 50)
        ny = Keyword.get(opts, :ny, 50)
        {spread(x0, x1, nx), spread(y0, y1, ny)}

      other ->
        raise ArgumentError, ":cell must be a positive number, got: #{inspect(other)}"
    end
  end

  defp steps(a, b, cell), do: for(i <- 0..trunc((b - a) / cell), do: a + i * cell)

  defp spread(a, _b, 1), do: [a * 1.0]

  defp spread(a, b, n) when is_integer(n) and n > 1,
    do: for(i <- 0..(n - 1), do: a + i * (b - a) / (n - 1))

  defp spread(_a, _b, n),
    do: raise(ArgumentError, ":nx and :ny must be positive integers, got: #{inspect(n)}")

  defp keep_near(cells, nil, _observations), do: cells

  defp keep_near(cells, max, observations) when is_number(max) do
    {xs, ys} = points(observations)
    obs = Enum.zip(xs, ys)
    max2 = max * max

    Enum.filter(cells, fn {x, y} ->
      Enum.any?(obs, fn {a, b} -> (x - a) ** 2 + (y - b) ** 2 <= max2 end)
    end)
  end

  defp bounds({_, _, _, _} = b), do: b

  defp bounds(frame) do
    ensure_geoframe!()
    call(@gs, :total_bounds, [series(frame)]) || raise ArgumentError, "the extent frame is empty"
  end

  defp crs_of({_, _, _, _}), do: nil
  defp crs_of(frame), do: call(@gs, :crs, [series(frame)])

  defp model_options(frame, kind, opts) do
    {allow, opts} = Keyword.pop(opts, :allow_geographic, false)
    ensure_geoframe!()
    geographic = call(@gs, :is_geographic, [series(frame)]) == true

    case kind do
      :ordinary when geographic ->
        {&Kriging.ordinary/3, Keyword.put_new(opts, :coordinates_type, :geographic)}

      :ordinary ->
        {&Kriging.ordinary/3, opts}

      :universal ->
        euclidean!(frame, allow, "universal kriging")
        {&Kriging.universal/3, opts}

      other ->
        raise ArgumentError, "kind must be :ordinary or :universal, got: #{inspect(other)}"
    end
  end

  defp euclidean!(frame, allow, what) do
    ensure_geoframe!()

    if allow != true and call(@gs, :is_geographic, [series(frame)]) == true do
      raise ArgumentError,
            "the geometry has a geographic CRS, so #{what} would measure degrees: project it " <>
              "first (GeoFrame.GeoDataFrame.to_crs/2) or pass allow_geographic: true"
    end
  end

  defp values(frame, column) do
    ensure_geoframe!()

    case call(@gdf, :pull, [frame, to_string(column)]) do
      %{__struct__: @series} = s ->
        to_list(s)

      other ->
        raise ArgumentError,
              "#{inspect(column)} is not an attribute column: #{inspect(other, limit: 3)}"
    end
  end

  defp to_list(series), do: call(@series, :to_list, [series])

  defp series(%{__struct__: @gdf} = frame), do: call(@gdf, :geometry, [frame])
  defp series(%{__struct__: @gs} = series), do: series

  defp series(other) do
    raise ArgumentError,
          "expected a GeoFrame.GeoDataFrame or GeoFrame.GeoSeries, got: #{inspect(other, limit: 3)}"
  end

  defp ensure_geoframe! do
    unless Code.ensure_loaded?(@gdf) do
      raise ArgumentError,
            "Geostatex.GeoFrame needs GeoFrame: add {:geoframe, ...} to your dependencies"
    end
  end

  defp call(mod, fun, args), do: apply(mod, fun, args)
end
