defmodule Geostatex.SimulationTest do
  use ExUnit.Case, async: true

  alias Geostatex.{Kriging, Simulation}

  doctest Geostatex.Simulation

  @spherical [
    variogram_model: :spherical,
    variogram_parameters: %{psill: 1.0, range: 20.0, nugget: 0.0},
    mode_no: 500
  ]

  test "a seed gives the same field, nil draws one and records it" do
    a = Simulation.unconditional(:points, [1.0, 7.0], [2.0, -3.0], [seed: 5] ++ @spherical)

    assert a ==
             Simulation.unconditional(:points, [1.0, 7.0], [2.0, -3.0], [seed: 5] ++ @spherical)

    b = Simulation.unconditional(:points, [1.0, 7.0], [2.0, -3.0], @spherical)
    assert is_integer(b.seed) and b.seed in 0..4_294_967_295

    assert b ==
             Simulation.unconditional(
               :points,
               [1.0, 7.0],
               [2.0, -3.0],
               [seed: b.seed] ++ @spherical
             )

    assert b.field != a.field or b.seed == 5

    %{field: [row1, row2]} =
      Simulation.unconditional(:grid, [0.0, 1.0, 2.0], [0.0, 5.0], @spherical)

    assert length(row1) == 3 and length(row2) == 3

    %{field: [[[_, _], [_, _]]]} =
      Simulation.unconditional_3d(:grid, [0.0, 1.0], [0.0, 2.0], [1.0], @spherical)
  end

  # Mean, variance and covariance at lag h of fields at {0, 0} and {h, 0} over many seeds.
  defp ensemble(opts, h, fun \\ &Simulation.unconditional(:points, [0.0, &2], [0.0, 0.0], &1)) do
    pairs =
      for seed <- 1..1500 do
        %{field: [a, b]} = fun.([seed: seed] ++ opts, h)
        {a, b}
      end

    {as, bs} = Enum.unzip(pairs)
    n = length(as)
    mean = Enum.sum(as) / n
    var = Enum.sum(Enum.map(as, &((&1 - mean) ** 2))) / n
    mean_b = Enum.sum(bs) / n
    cov = Enum.sum(Enum.zip_with(as, bs, &((&1 - mean) * (&2 - mean_b)))) / n
    {mean, var, cov}
  end

  test "fields have the model's mean, variance and covariance (spherical 2D, gaussian 3D)" do
    opts =
      Keyword.merge(@spherical,
        mean: 2.0,
        variogram_parameters: %{psill: 1.0, range: 20.0, nugget: 0.25}
      )

    {mean, var, cov} = ensemble(opts, 10.0)
    # Variance 1.25; spherical covariance at h = range / 2: 1 - 0.75 + 0.0625.
    assert_in_delta mean, 2.0, 0.1
    assert_in_delta var, 1.25, 0.15
    assert_in_delta cov, 0.3125, 0.12

    gaussian = [
      variogram_model: :gaussian,
      variogram_parameters: %{psill: 2.0, range: 14.0, nugget: 0.0},
      mode_no: 500
    ]

    fun = &Simulation.unconditional_3d(:points, [0.0, 0.0], [0.0, 0.0], [0.0, &2], &1)
    {_, var, cov} = ensemble(gaussian, 4.0, fun)
    assert_in_delta var, 2.0, 0.25
    # 2 exp(-(4 / 8)^2)
    assert_in_delta cov, 2.0 * :math.exp(-0.25), 0.2
  end

  test "conditioning reproduces the data and has the kriging mean and variance" do
    points = [{0.0, 0.0}, {10.0, 0.0}, {0.0, 12.0}, {15.0, 9.0}]
    values = [1.0, 2.5, 0.5, 1.8]

    for k <- [
          Kriging.simple(
            points,
            values,
            Keyword.take(@spherical, [:variogram_model, :variogram_parameters]) ++ [mean: 1.0]
          ),
          Kriging.ordinary(
            points,
            values,
            Keyword.take(@spherical, [:variogram_model, :variogram_parameters])
          )
        ] do
      %{field: at_data} = Simulation.conditional(k, :points, [0.0, 15.0], [0.0, 9.0], seed: 3)
      assert_in_delta Enum.at(at_data, 0), 1.0, 1.0e-6
      assert_in_delta Enum.at(at_data, 1), 1.8, 1.0e-6

      {[z], [s]} = Kriging.execute(k, :points, [6.0], [5.0])

      fields =
        for seed <- 1..1500 do
          %{field: [f]} =
            Simulation.conditional(k, :points, [6.0], [5.0], seed: seed, mode_no: 300)

          f
        end

      mean = Enum.sum(fields) / 1500
      var = Enum.sum(Enum.map(fields, &((&1 - mean) ** 2))) / 1500
      assert_in_delta mean, z, 4 * :math.sqrt(s / 1500)
      assert_in_delta var, s, 0.15 * s
    end

    k3 =
      Kriging.ordinary_3d(Enum.map(points, fn {x, y} -> {x, y, x / 5} end), values,
        variogram_model: :exponential,
        variogram_parameters: %{psill: 1.0, range: 30.0, nugget: 0.1}
      )

    %{field: [f]} = Simulation.conditional_3d(k3, :points, [10.0], [0.0], [2.0], seed: 9)
    assert_in_delta f, 2.5, 1.0e-6
  end

  test "options are validated" do
    assert_raise ArgumentError, ~r/spherical, exponential or gaussian/, fn ->
      Simulation.unconditional(:points, [0.0], [0.0],
        variogram_model: :linear,
        variogram_parameters: %{psill: 1.0, range: 1.0, nugget: 0.0}
      )
    end

    assert_raise ArgumentError, ~r/:variogram_parameters/, fn ->
      Simulation.unconditional(:points, [0.0], [0.0], variogram_model: :spherical)
    end

    assert_raise ArgumentError, ~r/:seed/, fn ->
      Simulation.unconditional(:points, [0.0], [0.0], [seed: -1] ++ @spherical)
    end

    assert_raise ArgumentError, ~r/:mode_no/, fn ->
      Simulation.unconditional(:points, [0.0], [0.0], [mode_no: 0] ++ @spherical)
    end

    assert_raise ArgumentError, ~r/unknown option/, fn ->
      Simulation.unconditional(:points, [0.0], [0.0], [bogus: 1] ++ @spherical)
    end

    uk =
      Kriging.universal([{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}], [1.0, 2.0, 3.0],
        variogram_model: :spherical,
        variogram_parameters: %{psill: 1.0, range: 5.0, nugget: 0.0},
        drift_terms: [:regional_linear]
      )

    assert_raise ArgumentError, ~r/simple or ordinary/, fn ->
      Simulation.conditional(uk, :points, [0.5], [0.5])
    end
  end
end
