defmodule Geostatex.KrigingBlockOracleTest do
  use ExUnit.Case, async: true

  alias Geostatex.{Kriging, Reference}

  @fixture Reference.load!("kriging/block.json")
  @data @fixture["data"]
  @targets @fixture["targets"]

  setup_all do
    Code.ensure_loaded!(Kriging)
    :ok
  end

  defp model(c) do
    [psill, range, nugget] = c["parameters"]

    Kriging.ordinary(Enum.zip(@data["x"], @data["y"]), @data["values"],
      variogram_model: String.to_existing_atom(c["variogram_model"]),
      variogram_parameters: %{psill: psill, range: range, nugget: nugget}
    )
  end

  defp block_opts(%{"block" => %{"size" => [dx, dy]}}), do: [block: {dx, dy}]

  defp block_opts(%{"block" => %{"points" => points}}),
    do: [block_points: Enum.map(points, &List.to_tuple/1)]

  defp block_opts(_point), do: []

  for c <- @fixture["cases"] do
    @case c
    test "gstat block kriging: #{c["name"]}" do
      c = @case
      {z, s} = Kriging.execute(model(c), :points, @targets["x"], @targets["y"], block_opts(c))

      for {got, want, what} <- [{z, c["z"], "estimate"}, {s, c["variance"], "variance"}],
          {{a, b}, i} <- got |> Enum.zip(want) |> Enum.with_index() do
        assert_in_delta a, b, 1.0e-8, "#{c["name"]} #{what} [#{i}]: #{a} vs #{b} (#{a - b})"
      end
    end
  end

  test "a block's estimate is the weighted mean of the point estimates at its discretisation" do
    c = Enum.find(@fixture["cases"], &(&1["name"] == "rectangle"))
    k = model(c)
    [x0, y0] = [hd(@targets["x"]), hd(@targets["y"])]
    {[zb], _} = Kriging.execute(k, :points, [x0], [y0], block: {100, 30})

    # gstat's 4 x 4 Gauss-Legendre points of a 100 x 30 block and their exact weights. gstat
    # keeps the weights in single precision, so they sum to 1 - 7.5e-9 and the block estimate
    # moves by that share of the mean estimate: about 5e-8 here.
    nodes = [
      {-0.8611363115940526, 0.3478548451374538},
      {-0.3399810435848563, 0.6521451548625461},
      {0.3399810435848563, 0.6521451548625461},
      {0.8611363115940526, 0.3478548451374538}
    ]

    points = for {u, wu} <- nodes, {v, wv} <- nodes, do: {x0 + 50 * u, y0 + 15 * v, wu * wv / 4}
    {xs, ys, ws} = points |> Enum.map(&Tuple.to_list/1) |> Enum.zip_with(& &1) |> List.to_tuple()
    {zp, _} = Kriging.execute(k, :points, xs, ys)
    assert_in_delta zb, Enum.sum(Enum.zip_with(zp, ws, &(&1 * &2))), 1.0e-7
  end

  test "block options are validated" do
    k = model(hd(@fixture["cases"]))

    assert_raise ArgumentError, ~r/n_closest_points/, fn ->
      Kriging.execute(k, :points, [180_000.0], [331_000.0], block: {40, 40}, n_closest_points: 10)
    end

    assert_raise ArgumentError, ~r/:block/, fn ->
      Kriging.execute(k, :points, [180_000.0], [331_000.0], block: {40})
    end

    simple =
      Kriging.simple(Enum.zip(@data["x"], @data["y"]), @data["values"],
        variogram_model: :spherical,
        variogram_parameters: %{psill: 0.59, range: 897.0, nugget: 0.05},
        mean: 6.0
      )

    assert_raise ArgumentError, ~r/ordinary kriging only/, fn ->
      Kriging.execute(simple, :points, [180_000.0], [331_000.0], block: {40, 40})
    end
  end
end
