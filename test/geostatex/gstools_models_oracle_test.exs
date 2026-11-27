defmodule Geostatex.GstoolsModelsOracleTest do
  use ExUnit.Case, async: true

  alias Geostatex.{Kriging, Reference, Variogram}

  @fixture Reference.load!("models/gstools.json")
  @data @fixture["data"]

  setup_all do
    Code.ensure_loaded!(Kriging)
    Code.ensure_loaded!(Variogram)
    :ok
  end

  defp model(%{"model" => "hyperspherical"}), do: :superspherical
  defp model(c), do: String.to_existing_atom(c["model"])

  defp shapes(c) do
    case model(c) do
      :tpl_stable -> [c["shape"]["hurst"], c["shape"]["alpha"]]
      :circular -> []
      _ -> c["shape"] |> Map.values()
    end
  end

  # Variogram.model_values/3 takes scikit-gstat's layout: [range, sill, (shape), nugget].
  for c <- @fixture["values"], c["model"] != "tpl_stable" do
    @case c
    test "gstools variogram: #{c["model"]} #{inspect(c["shape"])}" do
      c = @case
      cof = [c["len_scale"], c["var"]] ++ shapes(c) ++ [c["nugget"]]
      got = Variogram.model_values(model(c), cof, @fixture["distances"])

      for {{a, b}, i} <- got |> Enum.zip(c["variogram"]) |> Enum.with_index() do
        assert_in_delta a, b, 1.0e-10, "#{c["model"]} [#{i}]: #{a} vs #{b}"
      end
    end
  end

  for c <- @fixture["kriging"] do
    @case c
    test "gstools Ordinary with #{c["model"]}" do
      c = @case
      shape = Map.new(c["shape"], fn {k, v} -> {String.to_existing_atom(k), v} end)

      k =
        Kriging.ordinary(Enum.zip(@data["x"], @data["y"]), @data["values"],
          variogram_model: model(c),
          variogram_parameters:
            Map.merge(%{psill: c["var"], range: c["len_scale"], nugget: c["nugget"]}, shape)
        )

      {z, s} = Kriging.execute(k, :points, @data["px"], @data["py"])

      for {got, want, what} <- [{z, c["z"], "estimate"}, {s, c["variance"], "variance"}],
          {{a, b}, i} <- got |> Enum.zip(want) |> Enum.with_index() do
        assert_in_delta a, b, 1.0e-8, "#{c["model"]} #{what} [#{i}]: #{a} vs #{b}"
      end
    end
  end

  test "GSTools models need their parameters in kriging, and fit in variograms" do
    points = Enum.zip(@data["x"], @data["y"])

    assert_raise ArgumentError, ~r/:variogram_parameters/, fn ->
      Kriging.ordinary(points, @data["values"], variogram_model: :rational)
    end

    assert_raise ArgumentError, ~r/:alpha/, fn ->
      Kriging.ordinary(points, @data["values"],
        variogram_model: :rational,
        variogram_parameters: %{psill: 1.0, range: 10.0, nugget: 0.0}
      )
    end

    v = Variogram.new(points, @data["values"], model: :rational)
    assert Enum.all?(v.parameters, &is_float/1) and length(v.parameters) == 4

    assert_raise ArgumentError, ~r/:model/, fn ->
      Variogram.new(points, @data["values"], model: :tpl_stable)
    end
  end
end
