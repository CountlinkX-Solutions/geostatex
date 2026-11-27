defmodule Geostatex.VariogramSpaceTimeOracleTest do
  use ExUnit.Case, async: true

  alias Geostatex.{Reference, SpaceTimeVariogram, Variogram}

  # The fixtures name options with strings converted by String.to_existing_atom/1; the atoms are
  # defined by Geostatex.SpaceTimeVariogram, which is not loaded yet on every Elixir version.
  setup_all do
    Code.ensure_loaded!(SpaceTimeVariogram)
    :ok
  end

  @tol 1.0e-9
  @fixture Reference.load!("variogram/spacetime.json")
  @datasets @fixture["datasets"]

  defp points(name), do: Enum.map(@datasets[name]["coordinates"], fn [x, y] -> {x, y} end)

  defp opts(params) do
    Enum.map(params, fn
      {"model", v} ->
        {:model, v |> String.replace("-", "_") |> String.to_existing_atom()}

      {"estimator", v} ->
        {:estimator, String.to_existing_atom(v)}

      {"xbins", v} ->
        {:xbins, String.to_existing_atom(v)}

      {"tbins", v} ->
        {:tbins, String.to_existing_atom(v)}

      {"maxlag", v} when is_binary(v) ->
        {:maxlag, String.to_existing_atom(v)}

      {"xdist_func", v} ->
        {:xdist_func, String.to_existing_atom(v)}

      {key, v} when key in ["x_lags", "t_lags", "maxlag", "use_nugget"] ->
        {String.to_existing_atom(key), v}
    end)
  end

  # Least-squares objective over the non-empty lags.
  defp objective(values, experimental) do
    values
    |> Enum.zip(experimental)
    |> Enum.reject(fn {_m, e} -> e == :nan end)
    |> Enum.map(fn {m, e} -> (m - e) ** 2 end)
    |> Enum.sum()
  end

  defp assert_marginal(%Variogram{} = v, expected) do
    assert v.n_lags == expected["n_lags"]
    Reference.assert_close(v.bins, expected["bins"], @tol)
    Reference.assert_close(v.experimental, expected["experimental"], @tol)
    ours = objective(v.model_at_bins, v.experimental)
    theirs = objective(Variogram.model_values(v.model, expected["cof"], v.bins), v.experimental)
    assert ours <= theirs * (1 + 1.0e-6) + 1.0e-12, "marginal objective #{ours} vs #{theirs}"
  end

  for {c, i} <- Enum.with_index(@fixture["cases"]) do
    test "case #{i}: #{c["dataset"]} #{inspect(c["params"])}" do
      c = Enum.at(@fixture["cases"], unquote(i))
      data = @datasets[c["dataset"]]
      st = SpaceTimeVariogram.new(points(c["dataset"]), data["values"], opts(c["params"]))

      assert {st.x_lags, st.t_lags} == {c["x_lags"], c["t_lags"]}
      Reference.assert_close(st.xbins, c["xbins"], @tol)
      Reference.assert_close(st.tbins, c["tbins"], @tol)
      Reference.assert_close(List.flatten(st.experimental), c["experimental"], @tol)
      assert_marginal(st.x_marginal, c["x_marginal"])
      assert_marginal(st.t_marginal, c["t_marginal"])

      for {key, value} <- c["model_params"] do
        Reference.assert_close(Map.fetch!(st.sills, String.to_existing_atom(key)), value, 1.0e-3)
      end

      if expected = c["cof"] do
        grid = List.flatten(st.experimental)
        ours = objective(List.flatten(st.model_at_bins), grid)
        theirs = objective(List.flatten(SpaceTimeVariogram.model_grid(st, expected)), grid)
        assert ours <= theirs * (1 + 1.0e-6), "objective #{ours} worse than scipy's #{theirs}"

        Reference.assert_close(
          st.model_at_bins,
          SpaceTimeVariogram.model_grid(st, expected),
          1.0e-3
        )
      end
    end
  end
end
