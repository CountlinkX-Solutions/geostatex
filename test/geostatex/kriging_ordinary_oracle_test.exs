defmodule Geostatex.KrigingOrdinaryOracleTest do
  use ExUnit.Case, async: true

  alias Geostatex.{Kriging, Reference}

  @fixture Reference.load!("kriging/ordinary.json")
  @datasets @fixture["datasets"]

  setup_all do
    Code.ensure_loaded!(Kriging)
    :ok
  end

  defp points(name), do: Enum.map(@datasets[name]["coordinates"], fn [x, y] -> {x, y} end)

  defp model(name), do: name |> String.replace("-", "_") |> String.to_existing_atom()

  # PyKrige's own fitted (or converted) parameters, as a psill / slope / scale map.
  defp explicit(c) do
    m = c["init"]["variogram_model"]
    p = c["variogram_model_parameters"]

    params =
      case m do
        "linear" -> %{slope: Enum.at(p, 0), nugget: Enum.at(p, 1)}
        "power" -> %{scale: Enum.at(p, 0), exponent: Enum.at(p, 1), nugget: Enum.at(p, 2)}
        _ -> %{psill: Enum.at(p, 0), range: Enum.at(p, 1), nugget: Enum.at(p, 2)}
      end

    opts =
      Enum.flat_map(c["init"], fn
        {"exact_values", v} -> [exact_values: v]
        {"anisotropy_scaling", v} -> [anisotropy_scaling: v]
        {"anisotropy_angle", v} -> [anisotropy_angle: v]
        {"coordinates_type", v} -> [coordinates_type: String.to_existing_atom(v)]
        {"pseudo_inv", v} -> [pseudo_inv: v]
        {"pseudo_inv_type", v} -> [pseudo_inv_type: String.to_existing_atom(v)]
        _ -> []
      end)

    [variogram_model: model(m), variogram_parameters: params] ++ opts
  end

  defp execute(k, c) do
    w = c["where"]
    opts = if n = c["execute"]["n_closest_points"], do: [n_closest_points: n], else: []
    style = String.to_existing_atom(c["style"])

    opts =
      if style == :masked,
        do: [mask: Enum.map(w["mask"], fn row -> Enum.map(row, &(&1 == 1)) end)] ++ opts,
        else: opts

    {z, s} = Kriging.execute(k, style, w["x"], w["y"], opts)
    {List.flatten(z), List.flatten(s)}
  end

  describe "predictions with PyKrige's variogram parameters" do
    for {c, i} <- Enum.with_index(@fixture["cases"]) do
      test "case #{i}: #{c["dataset"]} #{c["init"]["variogram_model"]} #{c["style"]}" do
        c = Enum.at(@fixture["cases"], unquote(i))
        k = Kriging.ordinary(points(c["dataset"]), @datasets[c["dataset"]]["values"], explicit(c))
        {z, s} = execute(k, c)
        Reference.assert_close(z, c["z"], 1.0e-8)
        Reference.assert_close(s, c["sigmasq"], 1.0e-8)
      end
    end
  end

  describe "automatic variograms" do
    for {c, i} <- Enum.with_index(@fixture["cases"]), Map.has_key?(c["init"], "nlags") do
      test "case #{i}: #{c["dataset"]} #{c["init"]["variogram_model"]} nlags #{c["init"]["nlags"]}" do
        c = Enum.at(@fixture["cases"], unquote(i))

        opts = [
          variogram_model: model(c["init"]["variogram_model"]),
          nlags: c["init"]["nlags"],
          weight: c["init"]["weight"]
        ]

        k = Kriging.ordinary(points(c["dataset"]), @datasets[c["dataset"]]["values"], opts)
        Reference.assert_close(k.lags, c["lags"], 1.0e-12)
        Reference.assert_close(k.semivariance, c["semivariance"], 1.0e-12)
        ours = Kriging.variogram_objective(k, k.variogram_model_parameters)
        theirs = Kriging.variogram_objective(k, c["variogram_model_parameters"])
        assert ours <= theirs * (1 + 1.0e-7), "objective #{ours} worse than PyKrige's #{theirs}"
      end
    end
  end
end
