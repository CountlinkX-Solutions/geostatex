defmodule Geostatex.ReleaseTest do
  # Guards the release configuration: the precompiled NIF build (release workflow, crate
  # features) must match what Geostatex.Native declares, and the Hex package must ship what a
  # build needs. Nothing here builds or publishes anything.
  use ExUnit.Case, async: true

  @root Path.expand("../..", __DIR__)

  describe "Hex package" do
    setup do
      {:ok, files: Mix.Project.config()[:package][:files]}
    end

    test "ships the native sources, lockfile, changelog and legal files", %{files: files} do
      for entry <-
            ~w(lib native/geostatex_core/src native/geostatex_native/src Cargo.toml Cargo.lock
               CHANGELOG.md LICENSE COPYING LICENSES NOTICE README.md mix.exs) do
        assert entry in files, "#{entry} missing from package files"
      end

      refute Enum.any?(files, &String.contains?(&1, "priv"))
    end

    test "declares the LGPL in the package, the crates and the licence files" do
      # The GSTools-derived code makes the whole library LGPL-3.0-only (NOTICE).
      assert Mix.Project.config()[:package][:licenses] == ["LGPL-3.0-only"]
      assert File.read!(Path.join(@root, "Cargo.toml")) =~ ~s(license = "LGPL-3.0-only")
      assert File.read!(Path.join(@root, "LICENSE")) =~ "GNU LESSER GENERAL PUBLIC LICENSE"
      assert File.read!(Path.join(@root, "COPYING")) =~ "GNU GENERAL PUBLIC LICENSE"
    end

    test "ships the licence text of every library whose code it follows" do
      for name <- ~w(scikit-gstat PyKrige numpy scipy scikit-learn) do
        assert Path.wildcard(Path.join(@root, "LICENSES/#{name}*")) != [],
               "no licence text for #{name} in LICENSES"

        assert File.read!(Path.join(@root, "NOTICE")) =~ name
      end
    end

    test "every package entry exists, so mix hex.build cannot report missing files", %{
      files: files
    } do
      for entry <- files do
        assert File.exists?(Path.join(@root, entry)), "#{entry} does not exist"
      end
    end

    test "lists the checksum files that exist in the project root", %{files: files} do
      packaged = Enum.filter(files, &String.starts_with?(&1, "checksum-"))
      assert packaged == Geostatex.MixProject.checksum_files(@root)
    end
  end

  describe "checksum files" do
    @tag :tmp_dir
    test "are found by name once a release produced them", %{tmp_dir: dir} do
      assert Geostatex.MixProject.checksum_files(dir) == []

      File.write!(Path.join(dir, "checksum-Elixir.Geostatex.Native.exs"), "%{}\n")
      File.write!(Path.join(dir, "other.exs"), "")

      assert Geostatex.MixProject.checksum_files(dir) == ["checksum-Elixir.Geostatex.Native.exs"]
    end
  end

  describe "docs" do
    test "link their sources to the release tag of this version" do
      config = Mix.Project.config()
      assert config[:docs][:source_ref] == "v" <> config[:version]
    end
  end

  describe "changelog" do
    test "a non-pre-release version has its own changelog section" do
      version = Version.parse!(Mix.Project.config()[:version])

      if version.pre == [] do
        changelog = File.read!(Path.join(@root, "CHANGELOG.md"))
        assert changelog =~ "## [#{version}] - "
      end
    end
  end

  describe "release workflow and NIF features" do
    setup do
      native = File.read!(Path.join(@root, "lib/geostatex/native.ex"))
      workflow = File.read!(Path.join(@root, ".github/workflows/release.yml"))
      cargo = File.read!(Path.join(@root, "native/geostatex_native/Cargo.toml"))

      [declared_nifs] = Regex.run(~r/nif_versions: \[([^\]]*)\]/, native, capture: :all_but_first)
      [declared_targets] = Regex.run(~r/targets: ~w\(([^)]*)\)/s, native, capture: :all_but_first)

      {:ok,
       workflow: workflow,
       cargo: cargo,
       nifs: Regex.scan(~r/\d+\.\d+/, declared_nifs) |> List.flatten() |> Enum.sort(),
       targets: declared_targets |> String.split() |> Enum.sort()}
    end

    test "Cargo.toml declares a rustler nif_version feature for every precompiled NIF version", %{
      cargo: cargo,
      nifs: nifs
    } do
      assert nifs != []

      for nif <- nifs do
        feature = "nif_version_" <> String.replace(nif, ".", "_")

        assert cargo =~ ~s(#{feature} = ["rustler/#{feature}"]),
               "#{feature} feature missing from native/geostatex_native/Cargo.toml"
      end
    end

    test "the release matrix builds exactly the NIF versions and targets of Geostatex.Native", %{
      workflow: workflow,
      nifs: nifs,
      targets: targets
    } do
      [matrix_nifs] = Regex.run(~r/^\s+nif: \[([^\]]*)\]/m, workflow, capture: :all_but_first)
      matrix_targets = Regex.scan(~r/target: ([\w-]+),/, workflow, capture: :all_but_first)

      assert matrix_nifs |> then(&Regex.scan(~r/\d+\.\d+/, &1)) |> List.flatten() |> Enum.sort() ==
               nifs

      assert matrix_targets |> List.flatten() |> Enum.sort() == targets
    end

    test "the workflow builds the geostatex_native crate", %{workflow: workflow} do
      assert workflow =~ "project-name: geostatex_native"
    end

    test "a tag push fails unless the tag is v plus the mix.exs version", %{workflow: workflow} do
      assert workflow =~ ~s("v$VERSION" != "${GITHUB_REF_NAME}")
    end

    test "the workflow does not coerce use-cross with || false", %{workflow: workflow} do
      assert workflow =~ "use-cross: ${{ matrix.job.use-cross }}"
      refute workflow =~ ~r/use-cross:[^\n]*\|\|/
    end
  end
end
