defmodule Geostatex.RegressionKrigingTest do
  use ExUnit.Case, async: true

  alias Geostatex.RegressionKriging

  doctest Geostatex.RegressionKriging

  @points for x <- 0..5, y <- 0..4, do: {x * 1.0, y * 1.0}
  @predictors for {x, y} <- @points, do: [x + y, x * y / 5]
  @values for {x, y} <- @points, do: 2 + 0.5 * (x + y) - 0.3 * (x * y / 5) + :math.sin(x)

  test "a {fit, predict} regression is used as given" do
    # The linear regression rebuilt as functions gives the built-in one's predictions.
    linear = RegressionKriging.fit(@predictors, @points, @values)
    fit = fn _rows, _values -> {linear.coef, linear.intercept} end

    predict = fn {coef, b}, rows ->
      Enum.map(rows, fn r -> b + Enum.sum(Enum.zip_with(r, coef, &(&1 * &2))) end)
    end

    custom = RegressionKriging.fit(@predictors, @points, @values, regression: {fit, predict})
    assert custom.coef == nil
    new = [{0.5, 0.5}, {4.5, 3.5}]
    rows = [[1.0, 0.05], [8.0, 3.15]]

    for {a, b} <-
          Enum.zip(
            RegressionKriging.predict(custom, rows, new),
            RegressionKriging.predict(linear, rows, new)
          ),
        do: assert_in_delta(a, b, 1.0e-12)
  end

  test "the score is 1 at the fitted points with exact kriging" do
    rk = RegressionKriging.fit(@predictors, @points, @values, n_closest_points: nil)
    assert_in_delta RegressionKriging.score(rk, @predictors, @points, @values), 1.0, 1.0e-9
  end

  test "options and inputs are validated" do
    for {args, message} <- [
          {{@predictors, @points, @values, [method: :simple]}, ~r/:method must be/},
          {{@predictors, @points, @values, [n_closest_points: 1]}, ~r/:n_closest_points/},
          {{@predictors, @points, @values, [regression: :svr]}, ~r/:regression must be/},
          {{Enum.drop(@predictors, 1), @points, @values, []}, ~r/one row per value/},
          {{Enum.map(@predictors, fn [a, _] -> [a, 2 * a] end), @points, @values, []},
           ~r/linearly dependent/},
          {{[[1.0], [1.0, 2.0]], [{0, 0}, {1, 1}], [1, 2], []}, ~r/same number of values/},
          {{@predictors, @points, @values, [regression: {fn _, _ -> :m end, fn _, _ -> [] end}]},
           ~r/one number per row/}
        ] do
      {p, x, z, opts} = args
      assert_raise ArgumentError, message, fn -> RegressionKriging.fit(p, x, z, opts) end
    end
  end
end
