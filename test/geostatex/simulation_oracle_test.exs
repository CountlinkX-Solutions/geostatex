defmodule Geostatex.SimulationOracleTest do
  use ExUnit.Case, async: true

  alias Geostatex.{Kriging, Reference, Simulation}

  @fixture Reference.load!("simulation/srf.json")
  @data @fixture["data"]

  setup_all do
    Code.ensure_loaded!(Simulation)
    :ok
  end

  defp params([psill, range, nugget]), do: %{psill: psill, range: range, nugget: nugget}
  defp style(c), do: if(c["style"] == "points", do: :points, else: :grid)

  # `loose` holds the indices compared to 1e-7 instead of 1e-9.
  defp assert_close(got, want, name, loose \\ MapSet.new()) do
    assert length(got) == length(want), name

    for {{a, b}, i} <- got |> Enum.zip(want) |> Enum.with_index() do
      delta = if i in loose, do: 1.0e-7, else: 1.0e-9
      assert_in_delta a, b, delta, "#{name} [#{i}]: #{a} vs #{b} (#{a - b})"
    end
  end

  # Targets at data points: there the kriging variance is rounding noise (about 1e-16) and its
  # square root (about 1e-8) scales the field, in GSTools as in Geostatex.
  defp at_data(c) do
    data = MapSet.new(Enum.zip(@data["x"], @data["y"]))

    if c["style"] == "points",
      do:
        Enum.zip(c["where"]["x"], c["where"]["y"])
        |> Enum.with_index()
        |> Enum.filter(fn {p, _} -> p in data end)
        |> MapSet.new(fn {_, i} -> i end),
      else: MapSet.new()
  end

  for c <- @fixture["srf"] do
    @case c
    test "gstools SRF: #{c["name"]}" do
      c = @case

      %{field: field, seed: seed} =
        Simulation.unconditional(style(c), c["where"]["x"], c["where"]["y"],
          variogram_model: String.to_existing_atom(c["variogram_model"]),
          variogram_parameters: params(c["parameters"]),
          mean: c["mean"],
          mode_no: c["mode_no"],
          seed: c["seed"],
          anisotropy_scaling: c["anisotropy_scaling"],
          anisotropy_angle: c["anisotropy_angle"]
        )

      assert seed == c["seed"]
      assert_close(List.flatten(field), c["field"], c["name"])
    end
  end

  for c <- @fixture["cond_srf"] do
    @case c
    test "gstools CondSRF: #{c["name"]}" do
      c = @case
      points = Enum.zip(@data["x"], @data["y"])

      opts = [
        variogram_model: String.to_existing_atom(c["variogram_model"]),
        variogram_parameters: params(c["parameters"]),
        exact_values: true
      ]

      k =
        case c["kind"] do
          "simple" -> Kriging.simple(points, @data["values"], [mean: c["mean"]] ++ opts)
          "ordinary" -> Kriging.ordinary(points, @data["values"], opts)
        end

      %{field: field} =
        Simulation.conditional(k, style(c), c["where"]["x"], c["where"]["y"],
          mode_no: c["mode_no"],
          seed: c["seed"]
        )

      assert_close(List.flatten(field), c["field"], c["name"], at_data(c))
    end
  end
end
