defmodule Geostatex.ClassificationKrigingTest do
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias Geostatex.ClassificationKriging

  doctest Geostatex.ClassificationKriging

  @points for x <- 0..4, y <- 0..3, do: {x * 10.0, y * 10.0}
  @predictors for {x, y} <- @points, do: [x + y]
  @labels for {x, y} <- @points, do: if(x + y < 30, do: :low, else: :high)

  # A classifier by thresholds on the predictor: the share of `:high` grows with it.
  defp classifier do
    share = fn s -> min(max(s / 70, 0.05), 0.95) end

    {fn _p, _labels -> :model end,
     fn :model, rows -> for [s] <- rows, do: [share.(s), 1 - share.(s)] end}
  end

  test "ilr of two parts is log(b / a) / sqrt(2), and the inverse closes the composition" do
    [[z]] = ClassificationKriging.ilr([[0.2, 0.8]])
    assert_in_delta z, :math.log(0.8 / 0.2) / :math.sqrt(2), 1.0e-15
    [[a, b]] = ClassificationKriging.inverse_ilr([[z]])
    assert_in_delta a, 0.2, 1.0e-15
    assert_in_delta b, 0.8, 1.0e-15
    [uniform] = ClassificationKriging.ilr([[0.25, 0.25, 0.25, 0.25]])
    for v <- uniform, do: assert_in_delta(v, 0.0, 1.0e-15)
  end

  property "inverse_ilr undoes ilr on compositions" do
    check all(parts <- list_of(float(min: 0.01, max: 10.0), min_length: 2, max_length: 5)) do
      total = Enum.sum(parts)
      composition = Enum.map(parts, &(&1 / total))
      [back] = ClassificationKriging.inverse_ilr(ClassificationKriging.ilr([composition]))
      for {a, b} <- Enum.zip(back, composition), do: assert_in_delta(a, b, 1.0e-12)
    end
  end

  test "classes are sorted, predictions are classes, the score is the accuracy" do
    ck =
      ClassificationKriging.fit(@predictors, @points, @labels,
        classification: classifier(),
        variogram_model: :spherical,
        variogram_parameters: %{psill: 1.0, range: 30.0, nugget: 0.1}
      )

    assert ck.classes == [:high, :low]
    predicted = ClassificationKriging.predict(ck, @predictors, @points)
    assert Enum.all?(predicted, &(&1 in [:high, :low]))
    proba = ClassificationKriging.predict_proba(ck, @predictors, @points)
    assert Enum.all?(proba, &(abs(Enum.sum(&1) - 1) < 1.0e-12))
    # At the data points the kriged residuals restore the observed classes.
    assert predicted == @labels
    assert ClassificationKriging.score(ck, @predictors, @points, @labels) == 1.0
  end

  test "options are validated" do
    assert_raise ArgumentError, ~r/:classification/, fn ->
      ClassificationKriging.fit(@predictors, @points, @labels, variogram_model: :spherical)
    end

    bad = {fn _p, _l -> :m end, fn :m, rows -> for _ <- rows, do: [1.0] end}

    assert_raise ArgumentError, ~r/one probability per class/, fn ->
      ClassificationKriging.fit(@predictors, @points, @labels, classification: bad)
    end

    assert_raise ArgumentError, ~r/two classes/, fn ->
      ClassificationKriging.fit(@predictors, @points, Enum.map(@labels, fn _ -> :one end),
        classification: classifier()
      )
    end
  end
end
