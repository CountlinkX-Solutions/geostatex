defmodule Geostatex.VariogramFitControlsOracleTest do
  use ExUnit.Case, async: true

  alias Geostatex.{Reference, Variogram}

  setup_all do
    Code.ensure_loaded!(Variogram)
    :ok
  end

  @fixture Reference.load!("variogram/fit_controls.json")
  @datasets @fixture["datasets"]

  defp points(name), do: Enum.map(@datasets[name]["coordinates"], fn [x, y] -> {x, y} end)

  defp opts(params) do
    Enum.flat_map(params, fn
      {"model", v} -> [model: String.to_existing_atom(v)]
      {"use_nugget", v} -> [use_nugget: v]
      {"fit_method", v} -> [fit_method: String.to_existing_atom(v)]
      {"fit_bounds", [lower, upper]} -> [fit_bounds: {lower, upper}]
      {"fit_p0", v} -> [fit_p0: v]
    end)
  end

  # scikit-gstat's least-squares objective over the non-empty lags.
  defp objective(c, cof) do
    model = String.to_existing_atom(c["params"]["model"])

    [Variogram.model_values(model, cof, c["bins"]), c["experimental"]]
    |> Enum.zip_with(fn [m, e] -> {m, e} end)
    |> Enum.reject(fn {_m, e} -> e == :nan end)
    |> Enum.map(fn {m, e} -> (m - e) ** 2 end)
    |> Enum.sum()
  end

  for c <- @fixture["cases"] do
    @case c
    test "scikit-gstat #{c["params"]["fit_method"] || "trf"} #{c["params"]["model"]} with #{inspect(Map.keys(c["params"]) -- ["model", "fit_method", "use_nugget"])} on #{c["dataset"]}" do
      c = @case

      v =
        Variogram.new(
          points(c["dataset"]),
          @datasets[c["dataset"]]["values"],
          [n_lags: 12] ++ opts(c["params"])
        )

      ours = objective(c, v.cof)
      scipy = objective(c, c["cof"])
      assert ours <= scipy * (1 + 1.0e-7), "objective #{ours} worse than scikit-gstat's #{scipy}"

      # Within the box the user gave (or (0, p0) for maximum likelihood).
      box =
        case c["params"] do
          %{"fit_method" => "ml", "fit_p0" => p0} -> {Enum.map(p0, fn _ -> 0.0 end), p0}
          %{"fit_bounds" => [l, u]} -> {l, u}
          _ -> nil
        end

      if box do
        {l, u} = box

        for {x, lo, hi} <- Enum.zip([v.cof, l, u]),
            do:
              assert(
                x >= lo - 1.0e-9 and x <= hi + 1.0e-9,
                "#{inspect(v.cof)} outside #{inspect(box)}"
              )

        # Where a bound binds in scikit-gstat's solution, the same bound binds here (elsewhere
        # the objective can be flat enough for scipy to stop 1e-4 away).
        for {ours_x, theirs, lo, hi} <- Enum.zip([v.cof, c["cof"], l, u]),
            bound <- [lo, hi],
            abs(theirs - bound) <= 1.0e-9 * max(abs(bound), 1.0),
            do: assert_in_delta(ours_x, bound, 1.0e-6 * max(abs(bound), 1.0))
      end
    end
  end

  test "fit controls are validated" do
    p = points("pancake")
    values = @datasets["pancake"]["values"]

    assert_raise ArgumentError, ~r/fit_p0 needs 2 values/, fn ->
      Variogram.new(p, values, fit_p0: [1.0, 2.0, 3.0])
    end

    assert_raise ArgumentError, ~r/:fit_bounds/, fn ->
      Variogram.new(p, values, fit_bounds: [0, 1])
    end

    assert_raise ArgumentError, ~r/summed/, fn ->
      Variogram.new(p, values, model: [:spherical, :gaussian], fit_p0: [1.0, 1.0, 1.0, 1.0])
    end
  end
end
