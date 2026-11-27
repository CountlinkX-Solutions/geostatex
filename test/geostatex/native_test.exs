defmodule Geostatex.NativeTest do
  use ExUnit.Case, async: true

  test "the NIF library loads and reports the ABI version the Elixir layer expects" do
    assert Geostatex.Native.nif_version() == 1
  end

  test "the core decodes the f64 wire format" do
    {bin, 3} = Geostatex.Input.to_f64_binary([1.5, -2, 4])
    assert Geostatex.Native.sum_f64(bin) == {:ok, 3.5}
    assert {:error, message} = Geostatex.Native.sum_f64(<<0::72>>)
    assert message =~ "multiple of 8"
  end
end
