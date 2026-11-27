defmodule Geostatex.KrigingCustomOracleTest do
  # PyKrige's variogram_model="custom" (fixtures/kriging/custom.json).
  use ExUnit.Case, async: true

  alias Geostatex.{Kriging, Reference}

  @fixture Reference.load!("kriging/custom.json")
  @datasets @fixture["datasets"]

  setup_all do
    Code.ensure_loaded!(Kriging)
    :ok
  end

  # The fixture's functions (test/reference/generate_kriging_custom.py).
  defp function("stable"),
    do: fn [s, r, a, n], d -> s * (1.0 - :math.exp(-:math.pow(d / r, a))) + n end

  defp function("hole"),
    do: fn [s, a, b, n], d -> s * (1.0 - :math.cos(d / a) * :math.exp(-d / b)) + n end

  defp points(name), do: Enum.map(@datasets[name]["coordinates"], &List.to_tuple/1)

  defp build(c) do
    opts =
      [
        variogram_model: :custom,
        variogram_parameters: c["params"],
        variogram_function: function(c["function"])
      ] ++
        Enum.map(c["init"], fn
          {"exact_values", v} -> {:exact_values, v}
          {"anisotropy_scaling", v} -> {:anisotropy_scaling, v}
          {"anisotropy_angle", v} -> {:anisotropy_angle, v}
          {"coordinates_type", v} -> {:coordinates_type, String.to_existing_atom(v)}
          {"pseudo_inv", v} -> {:pseudo_inv, v}
          {"drift_terms", v} -> {:drift_terms, Enum.map(v, &String.to_existing_atom/1)}
        end)

    {pts, values} = {points(c["dataset"]), @datasets[c["dataset"]]["values"]}

    case c["kind"] do
      "ordinary" -> Kriging.ordinary(pts, values, opts)
      "universal" -> Kriging.universal(pts, values, opts)
      "ordinary_3d" -> Kriging.ordinary_3d(pts, values, opts)
    end
  end

  defp execute(k, c) do
    t = c["targets"]
    opts = if n = c["execute"]["n_closest_points"], do: [n_closest_points: n], else: []

    {z, s} =
      case {c["kind"], c["style"]} do
        {"ordinary_3d", "points"} ->
          [xs, ys, zs] = t["coords"]
          Kriging.execute_3d(k, :points, xs, ys, zs, opts)

        {_, "points"} ->
          [xs, ys] = t["coords"]
          Kriging.execute(k, :points, xs, ys, opts)

        {_, "grid"} ->
          Kriging.execute(k, :grid, t["xs"], t["ys"], opts)

        {_, "masked"} ->
          Kriging.execute(k, :masked, t["xs"], t["ys"], [mask: t["mask"]] ++ opts)
      end

    {List.flatten(z), List.flatten(s)}
  end

  for {c, i} <- Enum.with_index(@fixture["cases"]) do
    test "case #{i}: #{c["kind"]} #{c["function"]} #{c["style"]} #{inspect(c["init"])} #{inspect(c["execute"])}" do
      c = Enum.at(@fixture["cases"], unquote(i))
      k = build(c)
      assert k.variogram_model == :custom
      assert k.variogram_model_parameters == c["params"]
      Reference.assert_close(k.lags, c["lags"], 1.0e-9)
      Reference.assert_close(k.semivariance, c["semivariance"], 1.0e-9)

      {z, s} = execute(k, c)
      Reference.assert_close(z, List.flatten(c["z"]), 1.0e-8)
      Reference.assert_close(s, List.flatten(c["ss"]), 1.0e-8)

      if stats = c["statistics"] do
        ours = Kriging.statistics(k)

        for key <- ~w(delta sigma epsilon q1 q2 cr),
            do: Reference.assert_close(ours[String.to_existing_atom(key)], stats[key], 1.0e-8)
      end
    end
  end
end
