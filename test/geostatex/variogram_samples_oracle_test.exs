defmodule Geostatex.VariogramSamplesOracleTest do
  use ExUnit.Case, async: true

  alias Geostatex.{Reference, Variogram}

  # The fixtures name options with strings converted by String.to_existing_atom/1; the atoms are
  # defined by Geostatex.Variogram, which is not loaded yet on every Elixir version.
  setup_all do
    Code.ensure_loaded!(Variogram)
    :ok
  end

  @tol 1.0e-9
  @fixture Reference.load!("variogram/samples.json")
  @datasets @fixture["datasets"]

  defp points(name) do
    @datasets[name]["coordinates"] |> Enum.map(fn [x, y] -> {x, y} end)
  end

  defp opts(params) do
    Enum.map(params, fn
      {"samples", v} -> {:samples, v}
      {"binning_random_state", v} -> {:seed, String.to_integer(v)}
      {"estimator", v} -> {:estimator, String.to_existing_atom(v)}
      {"bin_func", v} -> {:bin_func, String.to_existing_atom(v)}
      {"maxlag", v} when is_binary(v) -> {:maxlag, String.to_existing_atom(v)}
      {"maxlag", v} -> {:maxlag, v}
      {"n_lags", v} -> {:n_lags, v}
    end) ++ [fit_method: nil]
  end

  for {c, i} <- Enum.with_index(@fixture["cases"]) do
    test "case #{i}: #{c["dataset"]} #{inspect(c["params"])}" do
      c = Enum.at(@fixture["cases"], unquote(i))
      opts = opts(c["params"])
      v = Variogram.new(points(c["dataset"]), @datasets[c["dataset"]]["values"], opts)

      assert v.samples == opts[:samples]
      assert v.seed == opts[:seed]
      assert v.n_lags == c["n_lags"]
      assert v.bin_count == c["bin_count"]
      assert Enum.sum(v.bin_count) <= c["pairs"]
      Reference.assert_close(v.bins, c["bins"], @tol)
      Reference.assert_close(v.experimental, c["experimental"], @tol)
      if c["maxlag"], do: Reference.assert_close(v.maxlag, c["maxlag"], @tol)
    end
  end
end
