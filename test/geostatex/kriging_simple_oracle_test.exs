defmodule Geostatex.KrigingSimpleOracleTest do
  use ExUnit.Case, async: true

  alias Geostatex.{Kriging, Reference}

  @fixture Reference.load!("kriging/simple.json")

  setup_all do
    Code.ensure_loaded!(Kriging)
    :ok
  end

  defp opts(c) do
    [psill, range, nugget] = c["parameters"]

    base = [
      variogram_model: String.to_existing_atom(c["variogram_model"]),
      variogram_parameters: %{psill: psill, range: range, nugget: nugget},
      mean: c["mean"],
      exact_values: c["exact_values"],
      pseudo_inv: c["pseudo_inv"] != nil,
      pseudo_inv_type: String.to_existing_atom(c["pseudo_inv"] || "pinv")
    ]

    if c["dim"] == 2,
      do:
        base ++
          [anisotropy_scaling: c["anisotropy_scaling"], anisotropy_angle: c["anisotropy_angle"]],
      else: base
  end

  defp predict(%{"dim" => 2} = c) do
    d = @fixture["data"]["d2"]
    k = Kriging.simple(Enum.zip(d["x"], d["y"]), d["values"], opts(c))
    w = c["where"]
    style = if c["style"] == "points", do: :points, else: :grid
    {z, s} = Kriging.execute(k, style, w["x"], w["y"])
    {List.flatten(z), List.flatten(s)}
  end

  defp predict(%{"dim" => 3} = c) do
    d = @fixture["data"]["d3"]
    k = Kriging.simple_3d(Enum.zip([d["x"], d["y"], d["z"]]), d["values"], opts(c))
    w = c["where"]
    style = if c["style"] == "points", do: :points, else: :grid
    {z, s} = Kriging.execute_3d(k, style, w["x"], w["y"], w["z"])
    {List.flatten(z), List.flatten(s)}
  end

  for c <- @fixture["cases"] do
    @case c
    test "gstools Simple: #{c["name"]}" do
      {z, s} = predict(@case)

      for {got, want, what} <- [{z, @case["z"], "estimate"}, {s, @case["variance"], "variance"}],
          {a, b} <- Enum.zip(got, want) do
        assert_in_delta a, b, 1.0e-8, "#{@case["name"]} #{what}"
      end

      assert length(z) == length(@case["z"])
    end
  end
end
