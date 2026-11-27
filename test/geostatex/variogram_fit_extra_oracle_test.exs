defmodule Geostatex.VariogramFitExtraOracleTest do
  # The M10 fitting options against scikit-gstat 1.0.24 (fixtures/variogram/fit_extra.json).
  use ExUnit.Case, async: true

  alias Geostatex.{Reference, Variogram}

  # The fixtures name options with strings converted by String.to_existing_atom/1; the atoms are
  # defined by Geostatex.Variogram, which is not loaded yet on every Elixir version.
  setup_all do
    Code.ensure_loaded!(Variogram)
    :ok
  end

  @fixture Reference.load!("variogram/fit_extra.json")
  @datasets @fixture["datasets"]

  defp points(name), do: Enum.map(@datasets[name]["coordinates"], fn [x, y] -> {x, y} end)

  defp new(c, opts) do
    params = Enum.map(c["params"], fn {k, v} -> {String.to_existing_atom(k), atomize(k, v)} end)
    Variogram.new(points(c["dataset"]), @datasets[c["dataset"]]["values"], params ++ opts)
  end

  defp atomize("model", names) when is_list(names),
    do: Enum.map(names, &String.to_existing_atom/1)

  defp atomize(key, v) when key in ["model", "fit_sigma"], do: String.to_existing_atom(v)
  defp atomize(_key, v), do: v

  # scipy's weighted least squares over the non-empty lags (sigma holds one value per
  # non-empty lag), evaluated with the exact model formulas.
  defp objective(c, cof, sigma) do
    model = atomize("model", c["params"]["model"])

    [Variogram.model_values(model, cof, c["bins"]), c["experimental"]]
    |> Enum.zip_with(fn [m, e] -> {m, e} end)
    |> Enum.reject(fn {_m, e} -> e == :nan end)
    |> Enum.zip_with(sigma, fn {m, e}, s -> ((m - e) / s) ** 2 end)
    |> Enum.sum()
  end

  defp assert_fit(v, c, sigma) do
    Reference.assert_close(v.bins, c["bins"], 1.0e-9)
    Reference.assert_close(v.experimental, c["experimental"], 1.0e-9)
    ours = objective(c, v.cof, sigma)
    scipy = objective(c, c["cof"], sigma)
    assert ours <= scipy * (1 + 1.0e-7), "objective #{ours} worse than scipy's #{scipy}"

    # At scipy's optimum the curves agree (on flat optima different coefficients give one
    # curve); strictly below it Geostatex found another, better point and the curves may differ
    # (a local minimum of scipy's, or a flat valley scipy stopped short in).
    if ours >= scipy * (1 - 1.0e-9),
      do: Reference.assert_close(v.model_at_bins, c["model_at_bins"], 1.0e-3)
  end

  describe "summed model formulas with scikit-gstat's coefficients" do
    for {m, i} <- Enum.with_index(@fixture["sum_models"]) do
      test "#{Enum.join(m["model"], "+")} #{inspect(m["args"])}" do
        m = Enum.at(@fixture["sum_models"], unquote(i))
        model = atomize("model", m["model"])

        Reference.assert_close(
          Variogram.model_values(model, m["args"], m["h"]),
          m["values"],
          1.0e-12
        )
      end
    end
  end

  describe "summed-model fits" do
    for {c, i} <- Enum.with_index(@fixture["sums"]) do
      failed = c["skgstat_fit_failed"]

      test "case #{i}: #{c["dataset"]} #{inspect(c["params"])}#{if failed, do: " (#{failed} in scikit-gstat)"}" do
        c = Enum.at(@fixture["sums"], unquote(i))
        v = new(c, n_lags: 12)
        assert v.model == atomize("model", c["params"]["model"])

        if c["skgstat_fit_failed"] do
          # scipy's curve_fit ran out of evaluations.
          assert is_float(v.rmse) and Enum.all?(v.cof, &(is_float(&1) and &1 >= 0))
        else
          assert_fit(v, c, List.duplicate(1.0, Enum.count(c["experimental"], &(&1 != :nan))))
        end
      end
    end
  end

  describe "model: :harmonize" do
    for {c, i} <- Enum.with_index(@fixture["harmonize"]) do
      test "case #{i}: #{c["dataset"]} #{inspect(c["params"])}" do
        c = Enum.at(@fixture["harmonize"], unquote(i))

        v =
          Variogram.new(points(c["dataset"]), @datasets[c["dataset"]]["values"],
            model: :harmonize,
            n_lags: c["params"]["n_lags"],
            use_nugget: c["params"]["use_nugget"]
          )

        Reference.assert_close(v.bins, c["bins"], 1.0e-9)
        Reference.assert_close(v.experimental, c["experimental"], 1.0e-9)
        Reference.assert_close(v.model_at_bins, c["model_at_bins"], 1.0e-9)
        Reference.assert_close(Variogram.transform(v, c["h"]), c["model_at_h"], 1.0e-9)
        # The range is the distance scikit-gstat's index points at.
        Reference.assert_close(v.cof, [c["range"], c["sill"], c["nugget"]], 1.0e-9)
        Reference.assert_close(v.parameters, [c["range"], c["sill"], c["nugget"]], 1.0e-9)
        Reference.assert_close(v.rmse, c["rmse"], 1.0e-9)
        Reference.assert_close(v.nse, c["nse"], 1.0e-9)
      end
    end
  end

  describe "fit_method: :ml" do
    for {c, i} <- Enum.with_index(@fixture["ml"]) do
      failed = c["skgstat_fit_failed"]

      test "case #{i}: #{c["dataset"]} #{inspect(c["params"])}#{if failed, do: " (#{failed} in scikit-gstat)"}" do
        c = Enum.at(@fixture["ml"], unquote(i))
        v = new(c, n_lags: 12, fit_method: :ml)
        assert v.fit_method == :ml

        if c["skgstat_fit_failed"] do
          # SLSQP stepped to a zero range (the model divides by zero) or did not converge.
          assert is_float(v.rmse) and Enum.all?(v.cof, &(is_float(&1) and &1 >= 0))
        else
          # sum((pred - y)^2 / sigma): residuals divided by sqrt(sigma).
          assert_fit(v, c, Enum.map(c["sigma"], &:math.sqrt/1))
        end
      end
    end
  end

  describe "fit_sigma: :entropy" do
    for {c, i} <- Enum.with_index(@fixture["entropy"]) do
      test "case #{i}: #{c["dataset"]} #{inspect(c["params"])}" do
        c = Enum.at(@fixture["entropy"], unquote(i))
        v = new(c, n_lags: 12, fit_sigma: :entropy)
        assert v.fit_sigma == :entropy
        assert_fit(v, c, c["sigma"])
      end
    end
  end
end
