defmodule Geostatex.VariogramFitOracleTest do
  use ExUnit.Case, async: true

  alias Geostatex.{Reference, Variogram}

  # The fixtures name options with strings converted by String.to_existing_atom/1; the atoms are
  # defined by Geostatex.Variogram, which is not loaded yet on every Elixir version.
  setup_all do
    Code.ensure_loaded!(Variogram)
    :ok
  end

  @fixture Reference.load!("variogram/fit.json")
  @datasets @fixture["datasets"]

  # Fits where scipy's TRF ends in a local minimum: Geostatex reaches a lower RMSE. The check
  # that it is a local minimum is that the nugget fit is worse than the same model without a
  # nugget, whose optimum the nugget model contains (nugget = 0).
  @scipy_local_minima [
    {"meuse_lead", "cubic"},
    {"meuse_lead", "matern"},
    {"meuse_zinc", "gaussian"},
    {"meuse_zinc", "matern"},
    {"pancake", "spherical"}
  ]

  defp points(name), do: Enum.map(@datasets[name]["coordinates"], fn [x, y] -> {x, y} end)

  defp opts(params) do
    Enum.flat_map(params, fn
      {"model", v} -> [model: String.to_existing_atom(v)]
      {"use_nugget", v} -> [use_nugget: v]
      {"fit_method", v} -> [fit_method: String.to_existing_atom(v)]
      {"fit_sigma", v} -> [fit_sigma: String.to_existing_atom(v)]
      {"fit_range", v} -> [fit_range: v]
      {"fit_sill", v} -> [fit_sill: v]
      {"fit_nugget", v} -> [fit_nugget: v]
      {"fit_shape", v} -> [fit_shape: v]
    end)
  end

  # The least-squares objective scikit-gstat minimises: residuals over the non-empty lags,
  # divided by fit_sigma.
  defp objective(c, cof) do
    model = String.to_existing_atom(c["params"]["model"])
    max = Enum.max(c["bins"])

    sigma =
      Enum.map(c["bins"], fn b ->
        x = b / max

        case c["params"]["fit_sigma"] do
          nil -> 1.0
          "linear" -> x
          "exp" -> 1.0 / :math.exp(1.0 / x)
          "sqrt" -> :math.sqrt(x)
          "sq" -> x * x
        end
      end)

    [Variogram.model_values(model, cof, c["bins"]), c["experimental"], sigma]
    |> Enum.zip_with(fn [m, e, s] -> {m, e, s} end)
    |> Enum.reject(fn {_m, e, _s} -> e == :nan end)
    |> Enum.map(fn {m, e, s} -> ((m - e) / s) ** 2 end)
    |> Enum.sum()
  end

  describe "model formulas with scikit-gstat's coefficients" do
    for {m, i} <- Enum.with_index(@fixture["models"]) do
      test "#{m["model"]} s=#{inspect(m["params"]["s"])}" do
        m = Enum.at(@fixture["models"], unquote(i))
        p = m["params"]
        model = String.to_existing_atom(m["model"])
        cof = if p["s"], do: [p["r"], p["c0"], p["s"], p["b"]], else: [p["r"], p["c0"], p["b"]]
        Reference.assert_close(Variogram.model_values(model, cof, m["h"]), m["values"], 1.0e-12)
      end
    end
  end

  describe "fits scikit-gstat cannot do" do
    for {c, i} <- Enum.with_index(@fixture["cases"]), Map.has_key?(c, "skgstat_fit_failed") do
      test "case #{i}: #{c["dataset"]} #{inspect(c["params"])} (#{c["skgstat_fit_failed"]} in scikit-gstat)" do
        c = Enum.at(@fixture["cases"], unquote(i))

        v =
          Variogram.new(
            points(c["dataset"]),
            @datasets[c["dataset"]]["values"],
            [n_lags: 12] ++ opts(c["params"])
          )

        assert is_float(v.rmse) and Enum.all?(v.cof, &(is_float(&1) and &1 >= 0))
      end
    end
  end

  describe "fits" do
    for {c, i} <- Enum.with_index(@fixture["cases"]), not Map.has_key?(c, "skgstat_fit_failed") do
      test "case #{i}: #{c["dataset"]} #{inspect(c["params"])}" do
        c = Enum.at(@fixture["cases"], unquote(i))

        v =
          Variogram.new(
            points(c["dataset"]),
            @datasets[c["dataset"]]["values"],
            [n_lags: 12] ++ opts(c["params"])
          )

        Reference.assert_close(v.bins, c["bins"], 1.0e-9)

        local_minimum? =
          {c["dataset"], c["params"]["model"]} in @scipy_local_minima and
            c["params"]["use_nugget"] == true and not Map.has_key?(c["params"], "fit_sigma")

        cond do
          c["params"]["fit_method"] == "manual" ->
            Reference.assert_close(v.cof, c["cof"], 1.0e-12)
            Reference.assert_close(v.rmse, c["rmse"], 1.0e-9)

          local_minimum? ->
            assert v.rmse < c["rmse"] * (1 - 1.0e-3),
                   "expected a lower RMSE than scipy's local minimum #{c["rmse"]}, got #{v.rmse}"

          true ->
            # scipy's objective, evaluated with its own coefficients and the (exact) models.
            ours = objective(c, v.cof)
            scipy = objective(c, c["cof"])
            assert ours <= scipy * (1 + 1.0e-7), "objective #{ours} worse than scipy's #{scipy}"
            assert abs(ours - scipy) <= 1.0e-6 * scipy, "objective #{ours} vs scipy's #{scipy}"
            # Compare the fitted curves: on flat optima different coefficients give one curve.
            Reference.assert_close(v.model_at_bins, c["model_at_bins"], 1.0e-3)
        end
      end
    end
  end
end
