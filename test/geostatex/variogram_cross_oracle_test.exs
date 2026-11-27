defmodule Geostatex.VariogramCrossOracleTest do
  use ExUnit.Case, async: true

  alias Geostatex.{Reference, Variogram}

  # The fixtures name options with strings converted by String.to_existing_atom/1; the atoms are
  # defined by Geostatex.Variogram, which is not loaded yet on every Elixir version.
  setup_all do
    Code.ensure_loaded!(Variogram)
    :ok
  end

  @tol 1.0e-9
  @fixture Reference.load!("variogram/cross.json")
  @datasets @fixture["datasets"]

  defp points(name) do
    @datasets[name]["coordinates"] |> Enum.map(fn [x, y] -> {x, y} end)
  end

  defp values(name, variable), do: @datasets[name]["values"][variable]

  defp opts(params) do
    {direction, params} = Map.split(params, ["azimuth", "tolerance"])

    params
    |> Enum.map(fn
      {"estimator", v} -> {:estimator, String.to_existing_atom(v)}
      {"bin_func", v} -> {:bin_func, String.to_existing_atom(v)}
      {"maxlag", v} when is_binary(v) -> {:maxlag, String.to_existing_atom(v)}
      {"maxlag", v} -> {:maxlag, v}
      {"dist_func", v} -> {:dist_func, String.to_existing_atom(v)}
      {"n_lags", v} -> {:n_lags, v}
    end)
    |> Kernel.++(
      if direction == %{},
        do: [],
        else: [direction: [azimuth: direction["azimuth"], tolerance: direction["tolerance"]]]
    )
    |> Keyword.put(:fit_method, nil)
  end

  defp assert_matches(v, c) do
    assert v.n_lags == c["n_lags"]
    assert v.bin_count == c["bin_count"]
    Reference.assert_close(v.bins, c["bins"], @tol)
    Reference.assert_close(v.experimental, c["experimental"], @tol)
    if c["maxlag"], do: Reference.assert_close(v.maxlag, c["maxlag"], @tol)
  end

  for {c, i} <- Enum.with_index(@fixture["cases"]) do
    test "case #{i}: #{c["cross"]} #{c["dataset"]} #{c["primary"]}/#{c["co"]} #{inspect(c["params"])}" do
      c = Enum.at(@fixture["cases"], unquote(i))
      cross = String.to_existing_atom(c["cross"])

      v =
        Variogram.new(
          points(c["dataset"]),
          values(c["dataset"], c["primary"]),
          [co_values: values(c["dataset"], c["co"]), cross: cross] ++ opts(c["params"])
        )

      assert v.cross == cross
      assert_matches(v, c)
    end
  end

  for {g, i} <- Enum.with_index(@fixture["grids"]) do
    test "grid #{i}: #{g["dataset"]} #{inspect(g["variables"])}" do
      g = Enum.at(@fixture["grids"], unquote(i))
      vars = Enum.map(g["variables"], &values(g["dataset"], &1))
      grid = Variogram.cross_variograms(points(g["dataset"]), vars, opts(g["params"]))

      assert length(grid) == length(vars)

      for {row, expected_row} <- Enum.zip(grid, g["cells"]),
          {v, cell} <- Enum.zip(row, expected_row) do
        assert v.cross == if(cell["cross"], do: :skgstat, else: nil)
        assert_matches(v, cell)
      end
    end
  end
end
