# Large data sets benchmark (dev only, not run in CI).
#
#     MIX_ENV=prod mix run bench/large_data.exs
#     MIX_ENV=prod mix run bench/large_data.exs --sizes 1000,5000 --cases kriging_window
#
# Run it in the prod environment: Rustler builds the NIF in debug mode in dev and test, and a
# debug build is many times slower than the release build users get.
#
# Each case runs on a seeded synthetic field (a smooth surface plus noise on a 50 km square) and
# reports the wall time and the peak resident memory it added (Linux: the high-water mark of the
# BEAM process, reset before each case; "n/a" elsewhere). A case runs once, or three times (best
# time) when one run takes under a second. Each case has a largest size, so a run stays within
# about 15 GB of memory and a few minutes; `--max` raises every limit to the given size.
#
# Options:
#   --sizes   comma-separated point counts (default 500,1000,2000,5000,10000)
#   --cases   comma-separated case names (default: all, listed in @cases below)
#   --targets prediction points for the kriging cases (default 1000)
#   --max     override every case's largest size

defmodule Geostatex.Bench do
  alias Geostatex.{Kriging, Variogram}

  # {name, largest size, description}
  @cases [
    {"variogram", 10_000, "Variogram.new/3, every pair, spherical fit"},
    {"variogram_samples", 50_000, "Variogram.new/3, samples: 2000 (or n), spherical fit"},
    {"kriging_build", 10_000, "Kriging.ordinary/3, automatic spherical variogram"},
    {"kriging_global", 10_000, "Kriging.execute/5 at the targets, every point"},
    {"kriging_window", 10_000, "Kriging.execute/5 at the targets, n_closest_points: 20"},
    {"statistics", 2_000, "Kriging.statistics/1"},
    {"cross_validate", 1_000, "Kriging.cross_validate/4, every point"},
    {"cross_validate_window", 2_000, "Kriging.cross_validate/4, n_closest_points: 20"}
  ]

  def main(argv) do
    {opts, _, _} =
      OptionParser.parse(argv,
        strict: [sizes: :string, cases: :string, targets: :integer, max: :integer]
      )

    sizes = list(opts[:sizes], [500, 1000, 2000, 5000, 10_000], &String.to_integer/1)
    names = list(opts[:cases], Enum.map(@cases, &elem(&1, 0)), & &1)
    m = opts[:targets] || 1000
    cases = Enum.filter(@cases, fn {name, _, _} -> name in names end)

    IO.puts(
      "Geostatex #{Mix.Project.config()[:version]}, #{System.schedulers_online()} schedulers, " <>
        "#{m} targets, #{System.otp_release()} / Elixir #{System.version()}\n"
    )

    IO.puts("| case | n | time (s) | peak memory (MB) |\n|---|---:|---:|---:|")

    for n <- sizes, {name, limit, _} <- cases do
      if n > (opts[:max] || limit) do
        IO.puts("| #{name} | #{n} | skipped (limit #{limit}) | |")
      else
        {points, values} = field(n, 1)
        {seconds, peak} = measure(prepare(name, points, values, m))
        IO.puts("| #{name} | #{n} | #{Float.round(seconds, 3)} | #{peak} |")
      end
    end

    IO.puts("\nCases:")
    for {name, limit, desc} <- cases, do: IO.puts("- #{name} (up to #{limit}): #{desc}")
  end

  defp list(nil, default, _), do: default
  defp list(s, _, f), do: s |> String.split(",", trim: true) |> Enum.map(f)

  # Each case returns the function timed; the kriging cases build the model (and the targets)
  # beforehand, so they time the prediction only.
  defp prepare("variogram", p, v, _), do: fn -> Variogram.new(p, v, model: :spherical) end

  defp prepare("variogram_samples", p, v, _),
    do: fn -> Variogram.new(p, v, model: :spherical, samples: min(2000, length(p)), seed: 7) end

  defp prepare("kriging_build", p, v, _),
    do: fn -> Kriging.ordinary(p, v, variogram_model: :spherical) end

  defp prepare("kriging_global", p, v, m), do: predict(model(p, v), m, [])
  defp prepare("kriging_window", p, v, m), do: predict(model(p, v), m, n_closest_points: 20)

  defp prepare("statistics", p, v, _) do
    k = model(p, v)
    fn -> Kriging.statistics(k) end
  end

  defp prepare("cross_validate", p, v, _),
    do: fn -> Kriging.cross_validate(:ordinary, p, v, variogram_model: :spherical) end

  defp prepare("cross_validate_window", p, v, _) do
    fn ->
      Kriging.cross_validate(:ordinary, p, v, variogram_model: :spherical, n_closest_points: 20)
    end
  end

  # One model per size, shared by the kriging cases.
  defp model(p, v) do
    key = {:model, length(p)}

    with nil <- Process.get(key) do
      k = Kriging.ordinary(p, v, variogram_model: :spherical)
      Process.put(key, k)
      k
    end
  end

  defp predict(k, m, opts) do
    {targets, _} = field(m, 2)
    {xs, ys} = Enum.unzip(targets)
    fn -> Kriging.execute(k, :points, xs, ys, opts) end
  end

  # A seeded smooth surface plus noise on a 50 km square.
  defp field(n, seed) do
    :rand.seed(:exsss, {seed, n, 12_345})

    points = for _ <- 1..n, do: {:rand.uniform() * 50_000.0, :rand.uniform() * 50_000.0}

    values =
      for {x, y} <- points,
          do: :math.sin(x / 7000) + :math.cos(y / 9000) + 0.1 * :rand.normal()

    {points, values}
  end

  defp measure(f) do
    {first, peak} = timed(f)

    seconds =
      if first < 1.0, do: Enum.min([first | for(_ <- 1..2, do: elem(timed(f), 0))]), else: first

    {seconds, peak}
  end

  defp timed(f) do
    :erlang.garbage_collect()
    base = rss()
    reset_peak()
    {us, _} = :timer.tc(f)

    peak =
      case {base, hwm()} do
        {b, h} when is_integer(b) and is_integer(h) -> div(max(h - b, 0), 1024)
        _ -> "n/a"
      end

    {us / 1.0e6, peak}
  end

  # Linux only: /proc/self/clear_refs with "5" resets the peak RSS (VmHWM).
  defp reset_peak, do: File.write("/proc/self/clear_refs", "5")
  defp rss, do: status_kb("VmRSS")
  defp hwm, do: status_kb("VmHWM")

  defp status_kb(field) do
    with {:ok, s} <- File.read("/proc/self/status"),
         [_, kb] <- Regex.run(~r/#{field}:\s+(\d+) kB/, s) do
      String.to_integer(kb)
    else
      _ -> nil
    end
  end
end

Geostatex.Bench.main(System.argv())
