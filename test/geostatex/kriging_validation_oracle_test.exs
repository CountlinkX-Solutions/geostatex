defmodule Geostatex.KrigingValidationOracleTest do
  use ExUnit.Case, async: true

  alias Geostatex.{Kriging, Reference}

  # The fixtures name options with strings converted by String.to_existing_atom/1; the atoms are
  # defined by Geostatex.Kriging, which is not loaded yet on every Elixir version.
  setup_all do
    Code.ensure_loaded!(Kriging)
    :ok
  end

  @tol 1.0e-9
  @fixture Reference.load!("kriging/validation.json")
  @datasets @fixture["datasets"]

  defp points(name), do: Enum.map(@datasets[name]["coordinates"], &List.to_tuple/1)

  # PyKrige's own parameters, so that the comparison does not depend on the automatic fit.
  defp parameters("linear", [s, n]), do: %{slope: s, nugget: n}
  defp parameters("power", [s, e, n]), do: %{scale: s, exponent: e, nugget: n}
  defp parameters(_model, [p, r, n]), do: %{psill: p, range: r, nugget: n}

  defp opts(c) do
    model = c["init"]["variogram_model"]

    Enum.flat_map(c["init"], fn
      {"variogram_model", v} -> [variogram_model: String.to_existing_atom(v)]
      {"anisotropy_scaling", v} -> [anisotropy_scaling: v]
      {"anisotropy_angle", v} -> [anisotropy_angle: v]
      {"coordinates_type", v} -> [coordinates_type: String.to_existing_atom(v)]
      {"pseudo_inv", v} -> [pseudo_inv: v]
      {"drift_terms", v} -> [drift_terms: Enum.map(v, &String.to_existing_atom/1)]
      {"specified_drift", v} -> [specified_drift: v]
    end) ++ [variogram_parameters: parameters(model, c["params"])]
  end

  defp kind("ordinary3d"), do: :ordinary_3d
  defp kind(k), do: String.to_existing_atom(k)

  defp build(:ordinary, p, v, o), do: Kriging.ordinary(p, v, o)
  defp build(:universal, p, v, o), do: Kriging.universal(p, v, o)
  defp build(:ordinary_3d, p, v, o), do: Kriging.ordinary_3d(p, v, o)

  for {c, i} <- Enum.with_index(@fixture["statistics"]) do
    test "statistics #{i}: #{c["kind"]} #{c["dataset"]} #{inspect(c["init"])}" do
      c = Enum.at(@fixture["statistics"], unquote(i))
      k = build(kind(c["kind"]), points(c["dataset"]), @datasets[c["dataset"]]["values"], opts(c))
      s = Kriging.statistics(k)
      Reference.assert_close(s.delta, c["delta"], @tol)
      Reference.assert_close(s.sigma, c["sigma"], @tol)
      Reference.assert_close(s.epsilon, c["epsilon"], @tol)
      Reference.assert_close([s.q1, s.q2, s.cr], [c["Q1"], c["Q2"], c["cR"]], @tol)
    end
  end

  for {c, i} <- Enum.with_index(@fixture["loo"]) do
    test "leave-one-out #{i}: #{c["kind"]} #{c["dataset"]} #{inspect(c["init"]) |> String.slice(0, 80)} #{inspect(c["execute"])}" do
      c = Enum.at(@fixture["loo"], unquote(i))
      values = @datasets[c["dataset"]]["values"]
      extra = if n = c["execute"]["n_closest_points"], do: [n_closest_points: n], else: []
      cv = Kriging.cross_validate(kind(c["kind"]), points(c["dataset"]), values, opts(c) ++ extra)
      Reference.assert_close(cv.predictions, c["predictions"], @tol)
      Reference.assert_close(cv.variances, c["variances"], @tol)
      residuals = Enum.zip_with(c["predictions"], values, &(&1 - &2))
      Reference.assert_close(cv.residuals, residuals, @tol)
      mse = Enum.sum(Enum.map(residuals, &(&1 * &1))) / length(residuals)
      Reference.assert_close([cv.mse, cv.rmse], [mse, :math.sqrt(mse)], @tol)
    end
  end
end
