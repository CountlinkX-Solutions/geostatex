defmodule Geostatex.VariogramExperimentalOracleTest do
  use ExUnit.Case, async: true

  alias Geostatex.{Reference, Variogram}

  # The fixtures name options with strings converted by String.to_existing_atom/1; the atoms are
  # defined by Geostatex.Variogram, which is not loaded yet on every Elixir version.
  setup_all do
    Code.ensure_loaded!(Variogram)
    :ok
  end

  @tol 1.0e-9
  @fixture Reference.load!("variogram/experimental.json")
  @datasets @fixture["datasets"]

  defp points(name) do
    @datasets[name]["coordinates"] |> Enum.map(fn [x, y] -> {x, y} end)
  end

  defp opts(params) do
    Enum.map(params, fn
      {"estimator", v} -> {:estimator, String.to_existing_atom(v)}
      {"bin_func", v} when is_binary(v) -> {:bin_func, String.to_existing_atom(v)}
      {"bin_func", v} when is_list(v) -> {:bin_func, v}
      {"maxlag", v} when is_binary(v) -> {:maxlag, String.to_existing_atom(v)}
      {"maxlag", v} -> {:maxlag, v}
      {"dist_func", v} -> {:dist_func, String.to_existing_atom(v)}
      {"n_lags", v} -> {:n_lags, v}
      {"percentile", v} -> {:percentile, v}
      {"entropy_bins", v} -> {:entropy_bins, v}
    end)
  end

  for {c, i} <- Enum.with_index(@fixture["cases"]) do
    test "case #{i}: #{c["dataset"]} #{inspect(c["params"])}" do
      c = Enum.at(@fixture["cases"], unquote(i))

      v =
        Variogram.new(points(c["dataset"]), @datasets[c["dataset"]]["values"], opts(c["params"]))

      assert v.n_lags == c["n_lags"]
      assert v.bin_count == c["bin_count"]
      Reference.assert_close(v.bins, c["bins"], @tol)
      Reference.assert_close(v.experimental, c["experimental"], @tol)
      if c["maxlag"], do: Reference.assert_close(v.maxlag, c["maxlag"], @tol)
    end
  end
end
