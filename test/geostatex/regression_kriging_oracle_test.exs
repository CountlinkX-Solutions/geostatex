defmodule Geostatex.RegressionKrigingOracleTest do
  # pykrige.rk.RegressionKriging with sklearn's LinearRegression
  # (fixtures/kriging/regression.json).
  use ExUnit.Case, async: true

  alias Geostatex.{Kriging, Reference, RegressionKriging}

  @fixture Reference.load!("kriging/regression.json")
  @n @fixture["fit_rows"]

  setup_all do
    Code.ensure_loaded!(Kriging)
    :ok
  end

  defp split(key), do: Enum.split(@fixture[key], @n)
  defp points(key), do: key |> split() |> then(fn {a, b} -> {tuples(a), tuples(b)} end)
  defp tuples(rows), do: Enum.map(rows, &List.to_tuple/1)

  defp opts(params) do
    Enum.map(params, fn
      {"method", v} ->
        {:method, String.to_existing_atom(v)}

      {"variogram_model", v} ->
        {:variogram_model, String.to_existing_atom(v)}

      {"nlags", v} ->
        {:nlags, v}

      {"n_closest_points", v} ->
        {:n_closest_points, v}

      {"drift_terms", v} ->
        {:drift_terms, Enum.map(v, &String.to_existing_atom/1)}

      {"variogram_parameters", v} ->
        {:variogram_parameters, Map.new(v, fn {k, x} -> {String.to_existing_atom(k), x} end)}
    end)
  end

  # PyKrige's fitted parameters as explicit ones.
  defp explicit(:linear, [slope, nugget]), do: %{slope: slope, nugget: nugget}
  defp explicit(_model, [psill, range, nugget]), do: %{psill: psill, range: range, nugget: nugget}

  for {c, i} <- Enum.with_index(@fixture["cases"]) do
    test "case #{i}: #{inspect(c["params"])}" do
      c = Enum.at(@fixture["cases"], unquote(i))
      {p_fit, p_new} = split("predictors")
      {x_fit, x_new} = points("coordinates")
      {z_fit, z_new} = split("values")
      opts = opts(c["params"])

      rk = RegressionKriging.fit(p_fit, x_fit, z_fit, opts)
      Reference.assert_close(rk.coef, c["coef"], 1.0e-9)
      Reference.assert_close(rk.intercept, c["intercept"], 1.0e-9)
      Reference.assert_close(rk.kriging.lags, c["lags"], 1.0e-9)
      Reference.assert_close(rk.kriging.semivariance, c["semivariance"], 1.0e-9)

      pykrige = c["variogram_model_parameters"]

      rk =
        if Keyword.has_key?(opts, :variogram_parameters) do
          rk
        else
          # The automatic variogram of the residuals reaches PyKrige's objective or a lower one;
          # the predictions are compared with PyKrige's parameters.
          ours = Kriging.variogram_objective(rk.kriging, rk.kriging.variogram_model_parameters)
          theirs = Kriging.variogram_objective(rk.kriging, pykrige)
          assert ours <= theirs * (1 + 1.0e-7), "objective #{ours} worse than PyKrige's #{theirs}"
          model = Keyword.get(opts, :variogram_model, :linear)
          params = explicit(model, pykrige)
          RegressionKriging.fit(p_fit, x_fit, z_fit, [variogram_parameters: params] ++ opts)
        end

      Reference.assert_close(RegressionKriging.predict(rk, p_new, x_new), c["prediction"], 1.0e-8)
      Reference.assert_close(RegressionKriging.krige_residual(rk, x_new), c["residual"], 1.0e-8)
      Reference.assert_close(RegressionKriging.score(rk, p_new, x_new, z_new), c["score"], 1.0e-8)
    end
  end
end
