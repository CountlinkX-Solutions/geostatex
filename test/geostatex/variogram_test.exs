defmodule Geostatex.VariogramTest do
  use ExUnit.Case, async: true

  alias Geostatex.Variogram

  doctest Geostatex.Variogram

  @points [{0, 0}, {1, 0}, {2, 0}, {3, 0}, {0, 1}, {1, 1}]
  @values [1, 2, 4, 7, 2, 3]

  test "points and values may be Nx tensors or Explorer series" do
    lists = Variogram.new(@points, @values, n_lags: 3)
    tensor = Nx.tensor(Enum.map(@points, &Tuple.to_list/1), type: :f64)
    assert Variogram.new(tensor, Nx.tensor(@values, type: :f64), n_lags: 3) == lists
    assert Variogram.new(@points, Explorer.Series.from_list(@values), n_lags: 3) == lists
  end

  test "the result records the resolved options" do
    v =
      Variogram.new(@points, @values,
        maxlag: 0.5,
        estimator: :cressie,
        n_lags: 2,
        fit_method: nil
      )

    assert v.n == 6 and v.n_lags == 2 and v.estimator == :cressie
    assert_in_delta v.maxlag, 0.5 * :math.sqrt(10), 1.0e-12
    assert Variogram.new(@points, @values).maxlag == nil
  end

  test "options are validated" do
    for {opts, message} <- [
          {[estimator: :mean], ~r/:estimator must be one of/},
          {[n_lags: 0], ~r/:n_lags must be a positive integer/},
          {[bin_func: :auto], ~r/:bin_func must be one of/},
          {[bin_func: [3, 1]], ~r/ascending/},
          {[maxlag: -1], ~r/:maxlag must be/},
          {[dist_func: :haversine], ~r/:dist_func must be one of/},
          {[percentile: 120], ~r/:percentile/},
          {[bogus: 1], ~r/unknown option/}
        ] do
      assert_raise ArgumentError, message, fn -> Variogram.new(@points, @values, opts) end
    end
  end

  test "fitting at construction, transform/2 and no fit with fit_method: nil" do
    points = for x <- 0..9, y <- 0..9, do: {x * 1.0, y * 1.0}
    values = for {x, y} <- points, do: :math.sin(x / 3) + :math.cos(y / 4)
    v = Variogram.new(points, values, model: :exponential, use_nugget: true)
    assert [_range, _sill, _nugget] = v.cof
    assert length(v.parameters) == 3 and v.rmse >= 0
    assert Variogram.transform(v, [0.0]) == [List.last(v.cof)]

    plain = Variogram.new(points, values, fit_method: nil)
    assert plain.cof == nil and plain.rmse == nil
    assert_raise ArgumentError, ~r/no fitted model/, fn -> Variogram.transform(plain, [1.0]) end

    manual = Variogram.new(points, values, fit_method: :manual, fit_range: 5, fit_sill: 1)
    assert manual.cof == [5.0, 1.0, 0.0]
  end

  test "fit options are validated" do
    for {opts, message} <- [
          {[model: :linear], ~r/:model must be one of/},
          {[fit_method: :other], ~r/:fit_method must be one of/},
          {[fit_sigma: :other], ~r/:fit_sigma/},
          {[use_nugget: 1], ~r/:use_nugget/},
          {[fit_method: :manual, fit_range: 3], ~r/needs :fit_range and :fit_sill/},
          {[fit_range: -1], ~r/:fit_range must be a non-negative number/}
        ] do
      assert_raise ArgumentError, message, fn -> Variogram.new(@points, @values, opts) end
    end
  end

  test "inputs are validated" do
    assert_raise ArgumentError, ~r/values/, fn -> Variogram.new(@points, [1, 2]) end
    assert_raise ArgumentError, ~r/at least 2 points/, fn -> Variogram.new([{0, 0}], [1]) end
  end

  if match?({:unix, :linux}, :os.type()) do
    test "a pair list too large for the memory raises instead of aborting the VM" do
      points = for i <- 1..300_000, do: {rem(i * 7, 1000) * 1.0, div(i, 1000) * 1.0}
      values = List.duplicate(1.0, 300_000)

      assert_raise ArgumentError, ~r/every pair of 300000 points needs about .* GB/, fn ->
        Variogram.new(points, values, fit_method: nil)
      end
    end
  end

  test "summed models are validated" do
    for {model, message} <- [
          {[:spherical], ~r/:model must be one of/},
          {[:spherical, :harmonize], ~r/components must be among/},
          {[:spherical, :linear], ~r/components must be among/}
        ] do
      assert_raise ArgumentError, message, fn -> Variogram.new(@points, @values, model: model) end
    end

    assert_raise ArgumentError, ~r/cannot be fitted manually/, fn ->
      Variogram.new(@points, @values,
        model: [:spherical, :gaussian],
        fit_method: :manual,
        fit_range: 1,
        fit_sill: 1
      )
    end

    assert_raise ArgumentError, ~r/needs 4 coefficients/, fn ->
      Variogram.model_values([:spherical, :gaussian], [1.0, 1.0, 1.0], [1.0])
    end

    v = Variogram.new(@points, @values, model: [:spherical, :gaussian], n_lags: 3)
    assert length(v.cof) == 4 and length(v.parameters) == 5
    assert Variogram.transform(v, v.bins) == v.model_at_bins
  end

  test "a harmonized variogram is monotonic and has no model coefficients" do
    v = Variogram.new(@points, @values, model: :harmonize, n_lags: 3)
    known = Enum.reject(v.model_at_bins, &(&1 == :nan))
    assert known == Enum.sort(known)
    assert Variogram.transform(v, [0.0]) == [:nan]

    assert_raise ArgumentError, ~r/use transform/, fn ->
      Variogram.model_values(:harmonize, v.cof, [1.0])
    end

    # Explicit edges, where scikit-gstat's harmonization fails.
    edges = Variogram.new(@points, @values, model: :harmonize, bin_func: [1.2, 2.5, 3.5])
    assert [_range, _sill, +0.0] = edges.cof
  end

  test "pair sampling draws and records a seed, and a seed repeats the pairs" do
    opts = [samples: 4, n_lags: 2, fit_method: nil]
    v = Variogram.new(@points, @values, opts)
    assert is_integer(v.seed) and v.seed >= 0
    assert v.samples == 4

    again = Variogram.new(@points, @values, [seed: v.seed] ++ opts)
    assert {again.bin_count, again.experimental} == {v.bin_count, v.experimental}
    assert Variogram.new(@points, @values, n_lags: 2, fit_method: nil).seed == nil
  end

  test "pair sampling options are validated" do
    for bad <- [0, -1, 1.5, 0.0, :all] do
      assert_raise ArgumentError, ~r/:samples must be/, fn ->
        Variogram.new(@points, @values, samples: bad)
      end
    end

    for bad <- [-1, 18_446_744_073_709_551_616, 1.0] do
      assert_raise ArgumentError, ~r/:seed must be/, fn ->
        Variogram.new(@points, @values, samples: 3, seed: bad)
      end
    end

    assert_raise ArgumentError, ~r/between 1 and 6/, fn ->
      Variogram.new(@points, @values, samples: 7, fit_method: nil)
    end

    assert_raise ArgumentError, ~r/euclidean/, fn ->
      Variogram.new(@points, @values, samples: 3, dist_func: :cityblock, fit_method: nil)
    end

    assert_raise ArgumentError, ~r/direction/, fn ->
      Variogram.new(@points, @values, samples: 3, direction: [], fit_method: nil)
    end
  end

  test "cross-variogram options are validated" do
    assert_raise ArgumentError, ~r/:cross needs :co_values/, fn ->
      Variogram.new(@points, @values, cross: :classical)
    end

    assert_raise ArgumentError, ~r/:cross must be/, fn ->
      Variogram.new(@points, @values, co_values: @values, cross: :other)
    end

    assert_raise ArgumentError, ~r/needs estimator: :matheron/, fn ->
      Variogram.new(@points, @values, co_values: @values, cross: :classical, estimator: :dowd)
    end

    assert_raise ArgumentError, ~r/co_values/, fn ->
      Variogram.new(@points, @values, co_values: [1, 2])
    end

    assert_raise ArgumentError, ~r/at least 2 variables/, fn ->
      Variogram.cross_variograms(@points, [@values])
    end

    assert_raise ArgumentError, ~r/not :co_values/, fn ->
      Variogram.cross_variograms(@points, [@values, @values], co_values: @values)
    end
  end

  test "a bounded fit refuses a negative classical cross-variogram" do
    opposite = Enum.map(@values, &(-&1))
    opts = [co_values: opposite, cross: :classical, n_lags: 2]

    assert_raise ArgumentError, ~r/positive semivariance/, fn ->
      Variogram.new(@points, @values, opts)
    end

    assert %Variogram{cross: :classical} =
             Variogram.new(@points, @values, [fit_method: nil] ++ opts)
  end

  test "mahalanobis needs coordinates with a non-singular covariance" do
    collinear = for x <- 0..5, do: {x * 1.0, 2.0 * x}

    assert_raise ArgumentError, ~r/non-singular covariance/, fn ->
      Variogram.new(collinear, Enum.to_list(0..5), dist_func: :mahalanobis, fit_method: nil)
    end

    v = Variogram.new(@points, @values, dist_func: :mahalanobis, fit_method: nil, n_lags: 2)
    assert v.dist_func == :mahalanobis
  end
end

defmodule Geostatex.VariogramDirectionTest do
  use ExUnit.Case, async: true

  alias Geostatex.Variogram

  @points for x <- 0..9, y <- 0..9, do: {x * 1.0, y * 1.0}
  # Values vary along x only: pairs along y have zero semivariance.
  @values Enum.map(@points, fn {x, _} -> x end)

  test "pairs across the trend have no semivariance" do
    north = Variogram.new(@points, @values, n_lags: 4, direction: [azimuth: 90, tolerance: 1.0])
    east = Variogram.new(@points, @values, n_lags: 4, direction: [azimuth: 0, tolerance: 1.0])
    assert Enum.all?(north.experimental, &(&1 == 0.0))
    assert Enum.all?(east.experimental, &(&1 > 0.0))
    assert north.direction.model == :triangle and is_float(north.direction.bandwidth)
  end

  test "a bandwidth beyond the largest distance has no effect" do
    opts = [n_lags: 5, fit_method: nil]
    wide = Variogram.new(@points, @values, opts ++ [direction: [bandwidth: 1.0e6]])
    compass = Variogram.new(@points, @values, opts ++ [direction: [model: :compass]])
    assert wide.bin_count == compass.bin_count
    assert wide.direction.bandwidth == 1.0e6
  end

  test "direction options are validated" do
    for dir <- [
          [azimuth: 181],
          [tolerance: -1],
          [bandwidth: -2],
          [bandwidth: {:percentile, 120}],
          [model: :circle],
          [bogus: 1],
          :east
        ] do
      assert_raise ArgumentError, fn -> Variogram.new(@points, @values, direction: dir) end
    end
  end
end

defmodule Geostatex.SpaceTimeVariogramTest do
  use ExUnit.Case, async: true

  alias Geostatex.SpaceTimeVariogram

  @points for x <- 0..5, y <- 0..4, do: {x * 10.0, y * 10.0}
  @values Enum.map(@points, fn {x, _} -> for t <- 0..5, do: 0.1 * x + :math.sin(t) end)

  test "the grid has one row per spatial lag and one column per time step lag" do
    st = SpaceTimeVariogram.new(@points, @values, x_lags: 4)
    assert {st.x_lags, st.t_lags} == {4, 5}
    assert st.tbins == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert length(st.experimental) == 4 and Enum.all?(st.experimental, &(length(&1) == 5))
    assert Enum.all?(st.cof, &(&1 >= 0)) and length(st.cof) == 3
    assert Map.keys(st.sills) |> Enum.sort() == [:Ct, :Cx]
  end

  test "the models combine the marginals" do
    st = SpaceTimeVariogram.new(@points, @values, x_lags: 4, model: :sum)
    [vx] = Geostatex.Variogram.transform(st.x_marginal, [15.0])
    [vt] = Geostatex.Variogram.transform(st.t_marginal, [2.0])
    assert SpaceTimeVariogram.transform(st, [{15.0, 2.0}]) == [vx + vt]
    assert st.sills == %{} and st.cof == []

    p = SpaceTimeVariogram.new(@points, @values, x_lags: 4, model: :product)
    %{Cx: cx, Ct: ct} = p.sills
    [vx] = Geostatex.Variogram.transform(p.x_marginal, [15.0])
    [vt] = Geostatex.Variogram.transform(p.t_marginal, [2.0])
    assert SpaceTimeVariogram.transform(p, [{15.0, 2.0}]) == [cx * vt + ct * vx - vx * vt]
  end

  test "inputs and options are validated" do
    assert_raise ArgumentError, fn ->
      SpaceTimeVariogram.new(@points, Enum.map(@values, &Enum.take(&1, 1)))
    end

    assert_raise ArgumentError, fn -> SpaceTimeVariogram.new(@points, tl(@values)) end

    for opts <- [[model: :x], [x_lags: 0], [t_lags: :min], [estimator: :entropy], [bogus: 1]] do
      assert_raise ArgumentError, fn -> SpaceTimeVariogram.new(@points, @values, opts) end
    end
  end
end
