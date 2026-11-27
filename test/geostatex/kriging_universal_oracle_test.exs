defmodule Geostatex.KrigingUniversalOracleTest do
  use ExUnit.Case, async: true

  alias Geostatex.{Kriging, Reference}

  @fixture Reference.load!("kriging/universal.json")
  @points Enum.map(@fixture["data"]["coordinates"], fn [x, y] -> {x, y} end)
  @values @fixture["data"]["values"]

  setup_all do
    Code.ensure_loaded!(Kriging)
    :ok
  end

  # The functional drifts of the generator.
  defp functions do
    [
      fn x, y -> (x - 179_000.0) * (y - 331_000.0) / 1.0e6 end,
      fn x, y -> :math.sqrt((x - 178_500.0) ** 2 + (y - 333_600.0) ** 2) / 1000.0 end
    ]
  end

  defp spec(xs, ys),
    do: [Enum.map(xs, &(&1 / 1000.0)), Enum.map(ys, &((&1 - @fixture["y_mean"]) / 1000.0))]

  defp drift_opts("regional"), do: [drift_terms: [:regional_linear]]
  defp drift_opts("point_log"), do: [drift_terms: [:point_log], point_drift: wells()]
  defp drift_opts("external"), do: [drift_terms: [:external_z]] ++ external()
  defp drift_opts("functional"), do: [drift_terms: [:functional], functional_drift: functions()]

  defp drift_opts("specified") do
    {xs, ys} = Enum.unzip(@points)
    [drift_terms: [:specified], specified_drift: spec(xs, ys)]
  end

  defp drift_opts("combined"),
    do:
      [drift_terms: [:regional_linear, :point_log, :external_z], point_drift: wells()] ++
        external()

  defp wells, do: Enum.map(@fixture["wells"], &List.to_tuple/1)

  defp external do
    e = @fixture["external"]
    [external_drift: e["z"], external_drift_x: e["x"], external_drift_y: e["y"]]
  end

  defp vario do
    v = @fixture["variogram"]["variogram_parameters"]

    [
      variogram_model: :spherical,
      variogram_parameters: %{psill: v["psill"], range: v["range"], nugget: v["nugget"]}
    ]
  end

  defp aniso(a),
    do:
      Enum.map(a, fn
        {"anisotropy_scaling", v} -> {:anisotropy_scaling, v}
        {"anisotropy_angle", v} -> {:anisotropy_angle, v}
      end)

  for {c, i} <- Enum.with_index(@fixture["cases"]) do
    test "case #{i}: #{c["name"]} #{c["style"]} #{if c["anisotropy"] == %{}, do: "", else: "anisotropic"}" do
      c = Enum.at(@fixture["cases"], unquote(i))

      k =
        Kriging.universal(
          @points,
          @values,
          vario() ++ drift_opts(c["name"]) ++ aniso(c["anisotropy"])
        )

      %{"x" => xs, "y" => ys} = c["where"]

      exec =
        cond do
          c["name"] != "specified" ->
            []

          c["style"] == "points" ->
            [specified_drift_arrays: spec(xs, ys)]

          true ->
            [
              specified_drift_arrays:
                for(
                  row <- spec(for(_y <- ys, x <- xs, do: x), for(y <- ys, _x <- xs, do: y)),
                  do: Enum.chunk_every(row, length(xs))
                )
            ]
        end

      {z, s} = Kriging.execute(k, String.to_existing_atom(c["style"]), xs, ys, exec)
      Reference.assert_close(List.flatten(z), c["z"], 1.0e-8)
      Reference.assert_close(List.flatten(s), c["sigmasq"], 1.0e-8)
    end
  end

  test "automatic variogram with a regional linear drift" do
    a = @fixture["auto"]

    k =
      Kriging.universal(@points, @values,
        variogram_model: :exponential,
        nlags: 8,
        drift_terms: [:regional_linear]
      )

    Reference.assert_close(k.lags, a["lags"], 1.0e-12)
    Reference.assert_close(k.semivariance, a["semivariance"], 1.0e-12)

    assert Kriging.variogram_objective(k, k.variogram_model_parameters) <=
             Kriging.variogram_objective(k, a["variogram_model_parameters"]) * (1 + 1.0e-7)
  end
end
