defmodule Geostatex.KrigingTest do
  use ExUnit.Case, async: true

  alias Geostatex.Kriging

  doctest Geostatex.Kriging

  @points for x <- 0..5, y <- 0..5, do: {x * 1.0, y * 1.0}
  @values for {x, y} <- @points, do: :math.sin(x / 2) + y / 3

  test "a sill list is converted to the partial sill, as in PyKrige" do
    k =
      Kriging.ordinary(@points, @values,
        variogram_model: :spherical,
        variogram_parameters: [1.0, 4.0, 0.2]
      )

    assert k.variogram_model_parameters == [0.8, 4.0, 0.2]

    m =
      Kriging.ordinary(@points, @values,
        variogram_model: :spherical,
        variogram_parameters: %{sill: 1.0, range: 4.0, nugget: 0.2}
      )

    assert m.variogram_model_parameters == [0.8, 4.0, 0.2]
  end

  test "grid output has ny rows of nx values and masked cells are :nan" do
    k = Kriging.ordinary(@points, @values, variogram_model: :exponential)
    {z, s} = Kriging.execute(k, :grid, [0.5, 1.5, 2.5], [0.5, 1.5])
    assert length(z) == 2 and Enum.all?(z, &(length(&1) == 3)) and length(s) == 2

    {zm, _} =
      Kriging.execute(k, :masked, [0.5, 1.5, 2.5], [0.5, 1.5],
        mask: [[true, false, false], [false, false, true]]
      )

    assert hd(hd(zm)) == :nan and List.last(List.last(zm)) == :nan
    assert Enum.at(hd(zm), 1) == Enum.at(hd(z), 1)
  end

  test "repeated points need a pseudo-inverse" do
    points = [{0, 0}, {1, 0}, {1, 0}, {0, 1}]
    values = [1, 2, 2.5, 3]
    params = [variogram_model: :linear, variogram_parameters: [1.0, 0.0]]
    k = Kriging.ordinary(points, values, params)
    assert_raise ArgumentError, ~r/singular/, fn -> Kriging.execute(k, :points, [0.5], [0.5]) end
    pk = Kriging.ordinary(points, values, params ++ [pseudo_inv: true])
    assert {[z], [_]} = Kriging.execute(pk, :points, [0.5], [0.5])
    assert is_float(z)
  end

  test "options are validated" do
    for {opts, message} <- [
          {[variogram_model: :cubic], ~r/:variogram_model must be one of/},
          {[variogram_model: :power, variogram_parameters: [1, 2]],
           ~r/invalid variogram_parameters/},
          {[nlags: 0], ~r/:nlags/},
          {[coordinates_type: :polar], ~r/:coordinates_type/},
          {[pseudo_inv_type: :svd], ~r/:pseudo_inv_type/},
          {[anisotropy_scaling: 0], ~r/:anisotropy_scaling/},
          {[exact_values: 1], ~r/:exact_values/},
          {[bogus: 1], ~r/unknown option/}
        ] do
      assert_raise ArgumentError, message, fn -> Kriging.ordinary(@points, @values, opts) end
    end

    k = Kriging.ordinary(@points, @values)

    assert_raise ArgumentError, ~r/same length/, fn ->
      Kriging.execute(k, :points, [1, 2], [1])
    end

    assert_raise ArgumentError, ~r/mask must have/, fn ->
      Kriging.execute(k, :masked, [1, 2], [1], mask: [[true]])
    end

    assert_raise ArgumentError, ~r/n_closest_points/, fn ->
      Kriging.execute(k, :points, [1], [1], n_closest_points: 1)
    end

    assert_raise ArgumentError, ~r/style must be/, fn -> Kriging.execute(k, :cells, [1], [1]) end
  end
end

defmodule Geostatex.KrigingUniversalTest do
  use ExUnit.Case, async: true

  alias Geostatex.Kriging

  @points for x <- 0..5, y <- 0..5, do: {x * 1.0, y * 1.0}
  @values for {x, y} <- @points, do: 2 + 0.5 * x - 0.3 * y + :math.sin(x * y)
  @vario [variogram_model: :linear, variogram_parameters: [0.2, 0.05]]

  test "drift options are validated" do
    for {opts, message} <- [
          {[], ~r/:drift_terms must be a non-empty list/},
          {[drift_terms: [:quadratic]], ~r/:drift_terms must be/},
          {[drift_terms: [:point_log]], ~r/point_drift/},
          {[
             drift_terms: [:external_z],
             external_drift: [[1]],
             external_drift_x: [0, 1],
             external_drift_y: [0]
           ], ~r/external_drift rows/},
          {[drift_terms: [:specified], specified_drift: [[1, 2]]], ~r/one value per data point/},
          {[drift_terms: [:functional], functional_drift: [fn x -> x end]], ~r/2-arity/},
          {[drift_terms: [:regional_linear], coordinates_type: :geographic], ~r/Euclidean/}
        ] do
      assert_raise ArgumentError, message, fn ->
        Kriging.universal(@points, @values, @vario ++ opts)
      end
    end
  end

  test "specified drifts are needed at the targets and the moving window is refused" do
    xs = Enum.map(@points, &elem(&1, 0))

    k =
      Kriging.universal(
        @points,
        @values,
        @vario ++ [drift_terms: [:specified], specified_drift: [xs]]
      )

    assert_raise ArgumentError, ~r/specified_drift_arrays/, fn ->
      Kriging.execute(k, :points, [1.5], [1.5])
    end

    assert {[_], [_]} = Kriging.execute(k, :points, [1.5], [1.5], specified_drift_arrays: [[1.5]])

    r = Kriging.universal(@points, @values, @vario ++ [drift_terms: [:regional_linear]])

    assert_raise ArgumentError, ~r/n_closest_points is not available/, fn ->
      Kriging.execute(r, :points, [1.5], [1.5], n_closest_points: 4)
    end
  end

  test "a regional drift follows a planar trend outside the data" do
    plane = for {x, y} <- @points, do: 1 + 2 * x - y
    k = Kriging.universal(@points, plane, @vario ++ [drift_terms: [:regional_linear]])
    {[z], _} = Kriging.execute(k, :points, [8.0], [1.0])
    assert_in_delta z, 1 + 16 - 1, 1.0e-8
  end
end

defmodule Geostatex.Kriging3DTest do
  use ExUnit.Case, async: true

  alias Geostatex.Kriging

  @points for x <- 0..3, y <- 0..3, z <- 0..2, do: {x * 1.0, y * 1.0, z * 1.0}
  @values for {x, y, z} <- @points, do: x + 0.5 * y - z
  @vario [variogram_model: :linear, variogram_parameters: [1.0, 0.0]]

  test "3D grids come back as planes of rows" do
    k = Kriging.ordinary_3d(@points, @values, @vario)
    {z, s} = Kriging.execute_3d(k, :grid, [0.5, 1.5], [0.5, 1.5, 2.5], [0.5])
    assert length(z) == 1 and length(hd(z)) == 3 and length(hd(hd(z))) == 2
    assert length(s) == 1
  end

  test "2D and 3D models are not mixed and options are validated" do
    k3 = Kriging.ordinary_3d(@points, @values, @vario)
    assert_raise FunctionClauseError, fn -> Kriging.execute(k3, :points, [1.0], [1.0]) end
    k2 = Kriging.ordinary([{0, 0}, {1, 0}, {0, 1}], [1, 2, 3], @vario)

    assert_raise ArgumentError, ~r/execute_3d\/6 needs a model/, fn ->
      Kriging.execute_3d(k2, :points, [1], [1], [1])
    end

    assert_raise ArgumentError, ~r/unknown option/, fn ->
      Kriging.ordinary_3d(@points, @values, anisotropy_angle: 3)
    end

    assert_raise ArgumentError, ~r/3D anisotropy/, fn ->
      Kriging.ordinary_3d(@points, @values, anisotropy_scaling_z: 0)
    end

    assert_raise ArgumentError, ~r/3D points/, fn -> Kriging.ordinary_3d([{0, 0}], [1]) end

    assert_raise ArgumentError, ~r/3D drift terms/, fn ->
      Kriging.universal_3d(@points, @values, drift_terms: [:point_log])
    end
  end
end

defmodule Geostatex.KrigingValidationTest do
  use ExUnit.Case, async: true

  alias Geostatex.Kriging

  @points for x <- 0..5, y <- 0..5, do: {x * 1.0, y * 1.0}
  @values Enum.map(@points, fn {x, y} -> :math.sin(x) + 0.5 * y end)

  test "statistics standardise the sequential residuals" do
    k = Kriging.ordinary(@points, @values, variogram_model: :spherical)
    s = Kriging.statistics(k)
    assert length(s.delta) == length(s.sigma) and length(s.delta) <= length(@points) - 1
    assert Enum.zip_with(s.delta, s.sigma, &(&1 / &2)) == s.epsilon
    assert s.q1 >= 0 and s.q2 > 0 and s.cr > 0
  end

  test "leave-one-out uses the full model's variogram for every point" do
    cv = Kriging.cross_validate(:ordinary, @points, @values, variogram_model: :spherical)
    full = Kriging.ordinary(@points, @values, variogram_model: :spherical)
    assert cv.variogram_model_parameters == full.variogram_model_parameters
    assert length(cv.predictions) == 36 and Enum.all?(cv.variances, &(&1 > 0))
    assert_in_delta cv.rmse, :math.sqrt(cv.mse), 1.0e-12
    assert cv.mae <= cv.rmse
  end

  test "3D leave-one-out and errors" do
    pts = for {x, y} <- @points, do: {x, y, x * y / 5}
    cv = Kriging.cross_validate(:universal_3d, pts, @values, drift_terms: [:regional_linear])
    assert length(cv.residuals) == 36

    assert_raise FunctionClauseError, fn -> Kriging.cross_validate(:simple, @points, @values) end

    assert_raise ArgumentError, fn ->
      Kriging.cross_validate(:ordinary_3d, @points, @values)
    end
  end

  test "leave-one-out matches the fold-by-fold loop bit for bit" do
    pts3 = for {x, y} <- @points, do: {x, y, x * y / 5}
    drift = Enum.map(@points, fn {x, y} -> x - y end)

    for {kind, points, opts} <- [
          {:ordinary, @points, [variogram_model: :spherical]},
          {:ordinary, @points, [variogram_model: :exponential, n_closest_points: 7]},
          {:universal, @points,
           [variogram_model: :linear, drift_terms: [:specified], specified_drift: [drift]]},
          {:universal_3d, pts3, [drift_terms: [:regional_linear]]},
          {:ordinary, @points,
           [
             variogram_model: :custom,
             variogram_function: fn [s, r], d -> s * (1 - :math.exp(-d / r)) end,
             variogram_parameters: [1.0, 2.0]
           ]}
        ] do
      cv = Kriging.cross_validate(kind, points, @values, opts)
      {p, v} = sequential_cv(kind, points, @values, opts)
      assert bits(cv.predictions) == bits(p), "#{kind} #{inspect(opts)}: predictions"
      assert bits(cv.variances) == bits(v), "#{kind} #{inspect(opts)}: variances"
    end
  end

  test "global folds run only as many at a time as the memory holds" do
    cores = System.schedulers_online()
    assert Kriging.fold_concurrency(1_000, nil) == cores
    assert Kriging.fold_concurrency(4_000, 10_000) == min(2, cores)
    assert Kriging.fold_concurrency(10 ** 12, 10 ** 9) == 1
    assert Kriging.fold_concurrency(0, 10) == cores
  end

  test "an error in a fold is raised as it is" do
    opts = [
      variogram_model: :custom,
      variogram_function: fn _, _ -> raise RuntimeError, "bad variogram" end,
      variogram_parameters: [1.0]
    ]

    assert_raise RuntimeError, "bad variogram", fn ->
      Kriging.cross_validate(:ordinary, @points, @values, opts)
    end
  end

  defp bits(xs), do: for(x <- xs, into: <<>>, do: <<x::float-64>>)

  # The leave-one-out loop as it ran before the folds ran in parallel, from the public API: the
  # reference the cross-validation must match bit for bit.
  defp sequential_cv(kind, points, values, opts) do
    {n_closest, opts} = Keyword.pop(opts, :n_closest_points)
    full = apply(Kriging, kind, [points, values, opts])
    opts = Keyword.put(opts, :variogram_parameters, parameters(full))
    specified = opts[:specified_drift] || []

    0..(length(points) - 1)
    |> Enum.map(fn i ->
      drifts = Enum.map(specified, &List.delete_at(&1, i))
      opts_i = if specified == [], do: opts, else: Keyword.put(opts, :specified_drift, drifts)
      k = apply(Kriging, kind, [List.delete_at(points, i), List.delete_at(values, i), opts_i])

      exec =
        Enum.reject(
          [
            n_closest_points: n_closest,
            specified_drift_arrays: Enum.map(specified, &[Enum.at(&1, i)])
          ],
          fn {_, v} -> v in [nil, []] end
        )

      case Enum.at(points, i) do
        {x, y, z} ->
          {[p], [s]} = Kriging.execute_3d(k, :points, [x], [y], [z], exec)
          {p, s}

        {x, y} ->
          {[p], [s]} = Kriging.execute(k, :points, [x], [y], exec)
          {p, s}
      end
    end)
    |> Enum.unzip()
  end

  defp parameters(%Kriging{variogram_model: :custom, variogram_model_parameters: p}), do: p

  defp parameters(%Kriging{variogram_model: :linear, variogram_model_parameters: [s, n]}),
    do: %{slope: s, nugget: n}

  defp parameters(%Kriging{variogram_model_parameters: [p, r, n]}),
    do: %{psill: p, range: r, nugget: n}
end

defmodule Geostatex.KrigingSimpleTest do
  use ExUnit.Case, async: true

  alias Geostatex.Kriging

  @points for x <- 0..4, y <- 0..3, do: {x * 10.0, y * 10.0}
  @values Enum.map(@points, fn {x, y} -> 1.0 + x / 40 - y / 30 end)
  @opts [
    variogram_model: :exponential,
    variogram_parameters: %{psill: 0.5, range: 30.0, nugget: 0.1},
    mean: 1.2
  ]

  test "simple kriging reverts to the known mean far from the data" do
    k = Kriging.simple(@points, @values, @opts)
    assert k.mean == 1.2
    {[near, far], [s_near, s_far]} = Kriging.execute(k, :points, [10.0, 1.0e5], [10.0, 1.0e5])
    assert_in_delta near, Enum.at(@values, 5), 1.0e-12
    assert s_near < 1.0e-12
    assert_in_delta far, 1.2, 1.0e-12
    assert_in_delta s_far, 0.6, 1.0e-12

    k3 = Kriging.simple_3d(Enum.map(@points, fn {x, y} -> {x, y, x / 10} end), @values, @opts)
    {[[[z3]]], _} = Kriging.execute_3d(k3, :grid, [1.0e5], [1.0e5], [0.0])
    assert_in_delta z3, 1.2, 1.0e-12
  end

  test "simple kriging options are validated" do
    assert Kriging.simple(@points, @values, Keyword.delete(@opts, :mean)).mean == 0.0

    assert_raise ArgumentError, ~r/model with a sill/, fn ->
      Kriging.simple(@points, @values,
        variogram_model: :linear,
        variogram_parameters: %{slope: 1.0, nugget: 0.0}
      )
    end

    assert_raise ArgumentError, ~r/:mean must be a number/, fn ->
      Kriging.simple(@points, @values, Keyword.put(@opts, :mean, :zero))
    end

    assert_raise ArgumentError, ~r/euclidean/, fn ->
      Kriging.simple(@points, @values, Keyword.put(@opts, :coordinates_type, :geographic))
    end

    k = Kriging.simple(@points, @values, @opts)

    assert_raise ArgumentError, ~r/simple kriging/, fn ->
      Kriging.execute(k, :points, [1.0], [1.0], n_closest_points: 3)
    end
  end
end

defmodule Geostatex.KrigingCustomTest do
  use ExUnit.Case, async: true

  alias Geostatex.Kriging

  @points for x <- 0..5, y <- 0..3, do: {x * 1.0, y * 1.0}
  @values Enum.map(@points, fn {x, y} -> x + 0.5 * y end)
  @exponential &__MODULE__.exponential/2
  @opts [
    variogram_model: :custom,
    variogram_function: @exponential,
    variogram_parameters: [2.0, 3.0, 0.1]
  ]

  def exponential([s, r, n], d), do: s * (1 - :math.exp(-d / r)) + n

  test "a custom variogram cross-validates with its fixed parameters" do
    cv = Kriging.cross_validate(:ordinary, @points, @values, @opts)
    assert is_float(cv.rmse) and cv.rmse > 0

    # The same function as a PyKrige exponential model (range / 3, partial sill).
    builtin = [
      variogram_model: :exponential,
      variogram_parameters: %{psill: 2.0, range: 9.0, nugget: 0.1}
    ]

    assert_in_delta cv.rmse,
                    Kriging.cross_validate(:ordinary, @points, @values, builtin).rmse,
                    1.0e-12
  end

  test "custom variogram options are validated" do
    for {opts, message} <- [
          {[variogram_model: :custom, variogram_parameters: [1.0]],
           ~r/needs :variogram_function/},
          {[variogram_model: :custom, variogram_function: @exponential],
           ~r/needs :variogram_parameters/},
          {[variogram_function: @exponential], ~r/needs variogram_model: :custom/},
          {Keyword.put(@opts, :variogram_function, fn _p, _d -> :oops end),
           ~r/must return a number/}
        ] do
      assert_raise ArgumentError, message, fn ->
        @points |> Kriging.ordinary(@values, opts) |> Kriging.execute(:points, [0.5], [0.5])
      end
    end

    k = Kriging.ordinary(@points, @values, @opts)
    assert_raise ArgumentError, ~r/no objective/, fn -> Kriging.variogram_objective(k, [1.0]) end
  end
end
