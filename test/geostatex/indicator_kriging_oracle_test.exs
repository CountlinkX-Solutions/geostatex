defmodule Geostatex.IndicatorKrigingOracleTest do
  use ExUnit.Case, async: true

  alias Geostatex.{IndicatorKriging, Kriging, Reference}

  @fixture Reference.load!("kriging/indicator.json")
  @data @fixture["data"]
  @points Enum.zip(@data["x"], @data["y"])

  # The fixtures name models with strings converted by String.to_existing_atom/1; the atoms are
  # defined by Geostatex.Kriging, which another test may not have loaded yet.
  setup_all do
    Code.ensure_loaded!(IndicatorKriging)
    Code.ensure_loaded!(Geostatex.Kriging)
    :ok
  end

  defp map([psill, range, nugget]), do: %{psill: psill, range: range, nugget: nugget}

  # Per target, one value per threshold, from `execute/5`'s raw distributions.
  defp per_threshold(raw_cdf, :points), do: raw_cdf |> Enum.zip_with(& &1)
  defp per_threshold(raw_cdf, :grid), do: raw_cdf |> Enum.concat() |> Enum.zip_with(& &1)

  for c <- @fixture["cases"] do
    @case c
    test "pykrige ordinary kriging of each threshold: #{c["name"]}" do
      c = @case
      style = if c["style"] == "points", do: :points, else: :grid

      # PyKrige's parameters (the automatic case's as PyKrige fitted them), one per threshold.
      ik =
        IndicatorKriging.new(@points, @data["values"],
          thresholds: @fixture["thresholds"],
          variogram_model: String.to_existing_atom(c["variogram_model"]),
          variogram_parameters: Enum.map(c["fitted"], &map/1),
          nlags: c["nlags"]
        )

      %{raw_cdf: raw} = IndicatorKriging.execute(ik, style, c["where"]["x"], c["where"]["y"])

      for {got, want} <- Enum.zip(per_threshold(raw, style), c["raw"]),
          {a, b} <- Enum.zip(got, want) do
        assert_in_delta a, b, 1.0e-8, c["name"]
      end
    end
  end

  test "an automatic fit per threshold reaches PyKrige's objective or a lower one" do
    c = Enum.find(@fixture["cases"], &(&1["name"] == "automatic"))

    ik =
      IndicatorKriging.new(@points, @data["values"],
        thresholds: @fixture["thresholds"],
        variogram_model: :spherical,
        nlags: c["nlags"]
      )

    for {k, pykrige} <- Enum.zip(ik.models, c["fitted"]) do
      ours = Kriging.variogram_objective(k, k.variogram_model_parameters)
      assert ours <= Kriging.variogram_objective(k, pykrige) * (1 + 1.0e-9)
    end
  end
end
