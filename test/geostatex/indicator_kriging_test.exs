defmodule Geostatex.IndicatorKrigingTest do
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias Geostatex.IndicatorKriging

  doctest Geostatex.IndicatorKriging

  @points [{0.0, 0.0}, {10.0, 0.0}, {0.0, 10.0}, {10.0, 10.0}]
  @values [0.5, 1.5, 1.7, 3.0]
  @opts [
    thresholds: [1.0, 2.0],
    variogram_model: :spherical,
    variogram_parameters: %{psill: 0.2, range: 15.0, nugget: 0.0}
  ]

  test "indicators are 1 at or below each threshold" do
    assert IndicatorKriging.indicators([1, 2, 3], [1.5, 2.5]) == [[1, 0, 0], [1, 1, 0]]
  end

  test "order relations: clip, then the average of an upward and a downward pass" do
    assert IndicatorKriging.order_relations([-0.1, 0.5, 0.4, 1.2]) == [0.0, 0.45, 0.45, 1.0]
    assert IndicatorKriging.order_relations([0.8, 0.2]) == [0.5, 0.5]
    assert IndicatorKriging.order_relations([0.1, 0.3, 0.9]) == [0.1, 0.3, 0.9]
    assert IndicatorKriging.order_relations([]) == []
  end

  property "corrected distributions lie in [0, 1], never decrease and stay as they are" do
    check all(cdf <- list_of(float(min: -0.5, max: 1.5), min_length: 1, max_length: 8)) do
      fixed = IndicatorKriging.order_relations(cdf)
      assert length(fixed) == length(cdf)
      assert Enum.all?(fixed, &(&1 >= 0.0 and &1 <= 1.0))
      assert fixed |> Enum.chunk_every(2, 1, :discard) |> Enum.all?(fn [a, b] -> a <= b end)
      assert IndicatorKriging.order_relations(fixed) == fixed
    end
  end

  test "class means of the data, midpoints of empty inner classes, thresholds of empty tails" do
    assert IndicatorKriging.new(@points, @values, @opts).class_means == [0.5, 1.6, 3.0]

    ik =
      IndicatorKriging.new(
        @points,
        @values,
        Keyword.put(@opts, :thresholds, [0.6, 0.8, 2.0, 10.0])
      )

    assert ik.class_means == [0.5, 0.7, 1.6, 3.0, 10.0]
  end

  test "the E-type mean integrates the corrected distribution over the class means" do
    ik = IndicatorKriging.new(@points, @values, @opts)
    # At a data point the kriged indicators are the datum's: the E-type is its class mean.
    # (to rounding: the kriging weights are a unit vector up to it).
    r = IndicatorKriging.execute(ik, :points, [10.0, 0.0], [0.0, 10.0])

    for {got, want} <- [
          {r.cdf, [[0.0, 1.0], [0.0, 1.0]]},
          {r.exceedance, [[1.0, 0.0], [1.0, 0.0]]},
          {r.e_type, [1.6, 1.6]}
        ],
        {a, b} <- Enum.zip(List.flatten(got), List.flatten(want)) do
      assert_in_delta a, b, 1.0e-12
    end

    %{cdf: [cdf], e_type: [e]} = IndicatorKriging.execute(ik, :points, [4.0], [6.0])
    [f1, f2] = cdf
    assert_in_delta e, f1 * 0.5 + (f2 - f1) * 1.6 + (1 - f2) * 3.0, 1.0e-12

    %{cdf: rows, e_type: e_rows} =
      IndicatorKriging.execute(ik, :grid, [0.0, 5.0, 10.0], [0.0, 10.0])

    assert length(rows) == 2 and Enum.all?(rows, &(length(&1) == 3))
    assert Enum.all?(Enum.concat(rows), &(length(&1) == 2))
    assert length(e_rows) == 2 and Enum.all?(e_rows, &(length(&1) == 3))
  end

  test "3D indicator kriging" do
    points = Enum.map(@points, fn {x, y} -> {x, y, x / 10} end)
    ik = IndicatorKriging.new_3d(points, @values, @opts)

    %{cdf: [[f1, f2]], e_type: [e]} =
      IndicatorKriging.execute_3d(ik, :points, [0.0], [10.0], [0.0])

    assert_in_delta f1, 0.0, 1.0e-12
    assert_in_delta f2, 1.0, 1.0e-12
    assert_in_delta e, 1.6, 1.0e-12
  end

  test "thresholds and per-threshold parameters are validated" do
    assert_raise ArgumentError, ~r/:thresholds/, fn ->
      IndicatorKriging.new(@points, @values, Keyword.delete(@opts, :thresholds))
    end

    assert_raise ArgumentError, ~r/increasing/, fn ->
      IndicatorKriging.new(@points, @values, Keyword.put(@opts, :thresholds, [2.0, 1.0]))
    end

    one = %{psill: 0.2, range: 15.0, nugget: 0.0}

    assert_raise ArgumentError, ~r/one per threshold/, fn ->
      IndicatorKriging.new(@points, @values, Keyword.put(@opts, :variogram_parameters, [one]))
    end

    ik =
      IndicatorKriging.new(
        @points,
        @values,
        Keyword.put(@opts, :variogram_parameters, [one, %{one | range: 9.0}])
      )

    assert Enum.map(ik.models, &Enum.at(&1.variogram_model_parameters, 1)) == [15.0, 9.0]
  end
end
