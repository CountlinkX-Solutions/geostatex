defmodule Geostatex.Kriging3DOracleTest do
  use ExUnit.Case, async: true

  alias Geostatex.{Kriging, Reference}

  @fixture Reference.load!("kriging/kriging3d.json")
  @d @fixture["data"]
  @points Enum.zip([@d["x"], @d["y"], @d["z"]]) |> Enum.map(fn {x, y, z} -> {x, y, z} end)
  @values @d["values"]

  setup_all do
    Code.ensure_loaded!(Kriging)
    :ok
  end

  defp model(name), do: name |> String.replace("-", "_") |> String.to_existing_atom()

  defp params(c) do
    p = c["variogram_model_parameters"]

    case c["init"]["variogram_model"] do
      "linear" -> %{slope: Enum.at(p, 0), nugget: Enum.at(p, 1)}
      "power" -> %{scale: Enum.at(p, 0), exponent: Enum.at(p, 1), nugget: Enum.at(p, 2)}
      _ -> %{psill: Enum.at(p, 0), range: Enum.at(p, 1), nugget: Enum.at(p, 2)}
    end
  end

  defp opts(c) do
    rename = %{
      "anisotropy_scaling_y" => :anisotropy_scaling_y,
      "anisotropy_scaling_z" => :anisotropy_scaling_z,
      "anisotropy_angle_x" => :anisotropy_angle_x,
      "anisotropy_angle_y" => :anisotropy_angle_y,
      "anisotropy_angle_z" => :anisotropy_angle_z,
      "exact_values" => :exact_values,
      "pseudo_inv" => :pseudo_inv
    }

    base = [variogram_model: model(c["init"]["variogram_model"]), variogram_parameters: params(c)]
    base ++ for({k, v} <- c["init"], Map.has_key?(rename, k), do: {rename[k], v})
  end

  defp drifts("uk_regional" <> _), do: [drift_terms: [:regional_linear]]

  defp drifts("uk_specified"),
    do: [drift_terms: [:specified], specified_drift: [Enum.map(@d["z"], &(&1 / 10.0))]]

  defp drifts("uk_functional"),
    do: [
      drift_terms: [:functional],
      functional_drift: [fn a, _b, c -> (a - 50.0) * (c - 10.0) / 100.0 end]
    ]

  for {c, i} <- Enum.with_index(@fixture["cases"]) do
    test "case #{i}: #{c["name"]} #{c["style"]}" do
      c = Enum.at(@fixture["cases"], unquote(i))

      k =
        if String.starts_with?(c["name"], "uk_"),
          do: Kriging.universal_3d(@points, @values, opts(c) ++ drifts(c["name"])),
          else: Kriging.ordinary_3d(@points, @values, opts(c))

      %{"x" => xs, "y" => ys, "z" => zs} = c["where"]
      style = String.to_existing_atom(c["style"])

      exec =
        Enum.concat([
          if(n = c["execute"]["n_closest_points"], do: [n_closest_points: n], else: []),
          if(style == :masked,
            do: [
              mask:
                Enum.map(@fixture["mask"], fn plane ->
                  Enum.map(plane, fn row -> Enum.map(row, &(&1 == 1)) end)
                end)
            ],
            else: []
          ),
          if(c["name"] == "uk_specified",
            do: [specified_drift_arrays: [Enum.map(zs, &(&1 / 10.0))]],
            else: []
          )
        ])

      {z, s} = Kriging.execute_3d(k, style, xs, ys, zs, exec)
      Reference.assert_close(List.flatten(z), c["z"], 1.0e-8)
      Reference.assert_close(List.flatten(s), c["sigmasq"], 1.0e-8)
    end
  end

  test "the automatic variogram's lags and objective" do
    c = Enum.find(@fixture["cases"], &(&1["name"] == "auto"))
    k = Kriging.ordinary_3d(@points, @values, variogram_model: :exponential, nlags: 8)
    Reference.assert_close(k.lags, c["lags"], 1.0e-12)
    Reference.assert_close(k.semivariance, c["semivariance"], 1.0e-12)

    assert Kriging.variogram_objective(k, k.variogram_model_parameters) <=
             Kriging.variogram_objective(k, c["variogram_model_parameters"]) * (1 + 1.0e-7)
  end
end
