defmodule Geostatex.CokrigingOracleTest do
  use ExUnit.Case, async: true

  alias Geostatex.{Cokriging, Reference}

  @fixture Reference.load!("kriging/cokriging.json")
  @data @fixture["data"]
  @targets @fixture["targets"]

  setup_all do
    Code.ensure_loaded!(Cokriging)
    :ok
  end

  defp structures(c) do
    for s <- c["structures"] do
      %{model: String.to_existing_atom(s["model"]), range: s["range"], sills: s["sills"]}
    end
  end

  defp variable(%{"name" => name, "rows" => rows}) do
    {Enum.map(rows, &{Enum.at(@data["x"], &1), Enum.at(@data["y"], &1)}),
     Enum.map(rows, &Enum.at(@data[name], &1))}
  end

  defp assert_close(got, want, name) do
    assert length(got) == length(want), name

    for {{a, b}, i} <- got |> Enum.zip(want) |> Enum.with_index() do
      assert_in_delta a, b, 1.0e-8, "#{name} [#{i}]: #{a} vs #{b} (#{a - b})"
    end
  end

  for c <- @fixture["cases"] do
    @case c
    test "gstat ordinary cokriging: #{c["name"]}" do
      c = @case
      ck = Cokriging.new(Enum.map(c["variables"], &variable/1), structures: structures(c))
      r = Cokriging.execute(ck, :points, @targets["x"], @targets["y"])

      for {k, got, want} <-
            Enum.zip([0..(length(c["variables"]) - 1), r.estimates, c["estimates"]]),
          do: assert_close(got, want, "#{c["name"]} estimate #{k}")

      for {k, got, want} <-
            Enum.zip([0..(length(c["variables"]) - 1), r.variances, c["variances"]]),
          do: assert_close(got, want, "#{c["name"]} variance #{k}")

      for cov <- c["covariances"],
          do: assert_close(r.covariances[{cov["i"], cov["j"]}], cov["values"], "#{c["name"]} cov")
    end
  end

  for f <- @fixture["fits"] do
    @fit f
    test "gstat fit.lmc: #{f["name"]}" do
      f = @fit
      index = f["names"] |> Enum.with_index() |> Map.new()

      pair = fn id ->
        case String.split(id, ".") do
          [a] -> {index[a], index[a]}
          [a, b] -> {index[a], index[b]}
        end
      end

      samples =
        Map.new(f["sample"], fn s ->
          {pair.(s["id"]), %{lags: s["dist"], gamma: s["gamma"], counts: s["np"]}}
        end)

      fitted =
        Cokriging.fit_lmc(samples, [
          %{model: :nugget},
          %{model: :spherical, range: f["model"]["range"]}
        ])

      [nugget, spherical] = Enum.map(fitted, & &1.sills)

      for want <- f["fitted"] do
        {i, j} = pair.(want["id"])
        assert_in_delta nugget |> Enum.at(i) |> Enum.at(j), want["nugget"], 1.0e-8, want["id"]
        assert_in_delta spherical |> Enum.at(i) |> Enum.at(j), want["psill"], 1.0e-8, want["id"]
      end
    end
  end
end
