defmodule Geostatex.CokrigingTest do
  use ExUnit.Case, async: true

  alias Geostatex.{Cokriging, Kriging}

  doctest Geostatex.Cokriging

  @a {[{0.0, 0.0}, {10.0, 0.0}, {0.0, 10.0}, {10.0, 10.0}, {5.0, 5.0}], [1.0, 2.0, 1.5, 3.0, 2.2]}
  @b {[{0.0, 0.0}, {10.0, 0.0}, {0.0, 10.0}, {10.0, 10.0}], [10.0, 12.0, 11.0, 14.0]}
  @structures [
    %{model: :nugget, sills: [[0.05, 0.02], [0.02, 0.1]]},
    %{model: :spherical, range: 20.0, sills: [[0.5, 0.6], [0.6, 1.2]]}
  ]

  test "estimates, variances and covariances per variable, on points and grids" do
    ck = Cokriging.new([@a, @b], structures: @structures)
    r = Cokriging.execute(ck, :points, [0.0, 4.0], [0.0, 6.0])
    # At a datum of both variables: the data, variance 0.
    assert_in_delta hd(Enum.at(r.estimates, 0)), 1.0, 1.0e-12
    assert_in_delta hd(Enum.at(r.estimates, 1)), 10.0, 1.0e-12
    assert_in_delta hd(Enum.at(r.variances, 0)), 0.0, 1.0e-12
    assert Enum.all?(Enum.at(r.variances, 1), &(&1 >= -1.0e-12))
    assert Map.keys(r.covariances) == [{0, 1}]

    %{estimates: [rows, _], variances: [_, vrows], covariances: %{{0, 1} => crows}} =
      Cokriging.execute(ck, :grid, [0.0, 5.0, 10.0], [0.0, 10.0])

    assert length(rows) == 2 and Enum.all?(rows, &(length(&1) == 3))
    assert length(vrows) == 2 and length(crows) == 2
  end

  test "one variable is ordinary kriging" do
    {points, values} = @a

    ck =
      Cokriging.new([@a],
        structures: [
          %{model: :nugget, sills: [[0.05]]},
          %{model: :spherical, range: 20.0, sills: [[0.5]]}
        ]
      )

    k =
      Kriging.ordinary(points, values,
        variogram_model: :spherical,
        variogram_parameters: %{psill: 0.5, range: 20.0, nugget: 0.05}
      )

    %{estimates: [z], variances: [s]} = Cokriging.execute(ck, :points, [3.0, 7.0], [4.0, 1.0])
    {zk, sk} = Kriging.execute(k, :points, [3.0, 7.0], [4.0, 1.0])
    for {a, b} <- Enum.zip(z ++ s, zk ++ sk), do: assert_in_delta(a, b, 1.0e-12)
  end

  test "the model and the data are validated" do
    assert_raise ArgumentError, ~r/positive semi-definite/, fn ->
      Cokriging.new([@a, @b],
        structures: [%{model: :spherical, range: 20.0, sills: [[1.0, 2.0], [2.0, 1.0]]}]
      )
    end

    assert_raise ArgumentError, ~r/:structures/, fn -> Cokriging.new([@a, @b], []) end

    assert_raise ArgumentError, ~r/:model/, fn ->
      Cokriging.new([@a, @b],
        structures: [%{model: :linear, range: 1.0, sills: [[1, 0], [0, 1]]}]
      )
    end

    assert_raise ArgumentError, ~r/\{points, values\}/, fn ->
      Cokriging.new([:a], structures: @structures)
    end

    assert_raise ArgumentError, ~r/one per pair/, fn ->
      Cokriging.fit_lmc(
        %{
          {0, 0} => %{lags: [1.0, 2.0], gamma: [1.0, 2.0], counts: [5, 5]},
          {1, 1} => %{lags: [1.0, 2.0], gamma: [1.0, 2.0], counts: [5, 5]}
        },
        [%{model: :nugget}, %{model: :spherical, range: 3.0}]
      )
    end
  end
end
