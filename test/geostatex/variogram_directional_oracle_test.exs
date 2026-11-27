defmodule Geostatex.VariogramDirectionalOracleTest do
  use ExUnit.Case, async: true

  alias Geostatex.{Reference, Variogram}

  # The fixtures name options with strings converted by String.to_existing_atom/1; the atoms are
  # defined by Geostatex.Variogram, which is not loaded yet on every Elixir version.
  setup_all do
    Code.ensure_loaded!(Variogram)
    :ok
  end

  @tol 1.0e-9
  @fixture Reference.load!("variogram/directional.json")
  @datasets @fixture["datasets"]
  @direction_keys ["azimuth", "tolerance", "bandwidth", "directional_model"]

  defp points(name), do: Enum.map(@datasets[name]["coordinates"], fn [x, y] -> {x, y} end)

  defp direction(params) do
    params
    |> Map.take(@direction_keys)
    |> Enum.map(fn
      {"azimuth", v} -> {:azimuth, v}
      {"tolerance", v} -> {:tolerance, v}
      {"bandwidth", "q" <> q} -> {:bandwidth, {:percentile, String.to_integer(q)}}
      {"bandwidth", v} -> {:bandwidth, v}
      {"directional_model", v} -> {:model, String.to_existing_atom(v)}
    end)
  end

  defp opts(params) do
    params
    |> Map.drop(@direction_keys)
    |> Enum.map(fn
      {"estimator", v} -> {:estimator, String.to_existing_atom(v)}
      {"bin_func", v} -> {:bin_func, String.to_existing_atom(v)}
      {"maxlag", v} when is_binary(v) -> {:maxlag, String.to_existing_atom(v)}
      {"maxlag", v} -> {:maxlag, v}
      {"dist_func", v} -> {:dist_func, String.to_existing_atom(v)}
      {"n_lags", v} -> {:n_lags, v}
    end)
    |> Keyword.put(:direction, direction(params))
  end

  defp fit_opts(nil), do: []

  defp fit_opts(fit),
    do: [
      model: String.to_existing_atom(fit["model"]),
      fit_method: String.to_existing_atom(fit["fit_method"])
    ]

  for {c, i} <- Enum.with_index(@fixture["cases"]) do
    test "case #{i}: #{c["dataset"]} #{inspect(c["params"])}#{if c["fit"], do: " fit #{c["fit"]["model"]}"}" do
      c = Enum.at(@fixture["cases"], unquote(i))
      data = @datasets[c["dataset"]]

      v =
        Variogram.new(
          points(c["dataset"]),
          data["values"],
          opts(c["params"]) ++ fit_opts(c["fit"])
        )

      assert v.n_lags == c["n_lags"]
      assert v.bin_count == c["bin_count"]
      Reference.assert_close(v.bins, c["bins"], @tol)
      Reference.assert_close(v.experimental, c["experimental"], @tol)
      Reference.assert_close(v.direction.bandwidth, c["bandwidth"], @tol)
      if c["maxlag"], do: Reference.assert_close(v.maxlag, c["maxlag"], @tol)

      if c["fit"] do
        assert v.rmse <= c["rmse"] * (1 + 1.0e-7)
        Reference.assert_close(v.model_at_bins, c["model_at_bins"], 1.0e-3)
      end
    end
  end
end
