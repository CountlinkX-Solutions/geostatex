defmodule Geostatex.ClassificationKrigingOracleTest do
  use ExUnit.Case, async: true

  alias Geostatex.{ClassificationKriging, Reference}

  @fixture Reference.load!("kriging/classification.json")
  @train @fixture["train"]
  @test @fixture["test"]

  # The fixtures name models with strings converted by String.to_existing_atom/1; the atoms are
  # defined by Geostatex.Kriging, which another test may not have loaded yet.
  setup_all do
    Code.ensure_loaded!(ClassificationKriging)
    Code.ensure_loaded!(Geostatex.Kriging)
    :ok
  end

  # A classifier replaying scikit-learn's probabilities for each predictor row.
  defp classifier(c) do
    table =
      Map.new(
        Enum.zip(@train["predictors"], c["ml_train"]) ++
          Enum.zip(@test["predictors"], c["ml_test"])
      )

    {fn _predictors, _labels -> :replay end,
     fn :replay, rows -> Enum.map(rows, &Map.fetch!(table, &1)) end}
  end

  defp parameters(model, p) do
    case {model, p} do
      {"linear", [slope, nugget]} -> %{slope: slope, nugget: nugget}
      {_, [psill, range, nugget]} -> %{psill: psill, range: range, nugget: nugget}
    end
  end

  defp assert_rows(got, want, name) do
    assert length(got) == length(want), name

    for {{a, b}, i} <- got |> List.flatten() |> Enum.zip(List.flatten(want)) |> Enum.with_index() do
      assert_in_delta a, b, 1.0e-8, "#{name} [#{i}]: #{a} vs #{b}"
    end
  end

  for c <- @fixture["cases"] do
    @case c
    test "pykrige ClassificationKriging: #{c["name"]}" do
      c = @case
      model = Map.get(c["options"], "variogram_model", "linear")

      window =
        if Map.has_key?(c["options"], "n_closest_points"),
          do: [n_closest_points: c["options"]["n_closest_points"]],
          else: []

      ck =
        ClassificationKriging.fit(
          @train["predictors"],
          Enum.zip(@train["x"], @train["y"]),
          @train["labels"],
          [
            classification: classifier(c),
            variogram_model: String.to_existing_atom(model),
            variogram_parameters: Enum.map(c["fitted"], &parameters(model, &1))
          ] ++ window
        )

      points = Enum.zip(@test["x"], @test["y"])
      assert ck.classes == @fixture["classes"]
      assert_rows(ClassificationKriging.ilr(c["ml_test"]), c["ilr_ml_test"], "#{c["name"]} ilr")

      assert_rows(
        ClassificationKriging.krige_residual(ck, points),
        c["residual"],
        "#{c["name"]} residual"
      )

      assert_rows(
        ClassificationKriging.predict_proba(ck, @test["predictors"], points),
        c["proba"],
        "#{c["name"]} proba"
      )

      # PyKrige returns the class indices; Geostatex the classes.
      assert ClassificationKriging.predict(ck, @test["predictors"], points) ==
               Enum.map(c["indices"], &Enum.at(@fixture["classes"], &1))
    end
  end
end
