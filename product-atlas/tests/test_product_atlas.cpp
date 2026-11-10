#include "product_atlas/product_atlas.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

}  // namespace

int main() {
  using product_atlas::float_to_bf16;

  const std::vector<std::uint16_t> activations = {
      float_to_bf16(-1.5f), float_to_bf16(0.5f), float_to_bf16(2.0f)};
  const std::vector<std::uint16_t> weights = {
      float_to_bf16(-2.0f), float_to_bf16(0.25f), float_to_bf16(3.0f)};
  product_atlas::ProductAtlas atlas(activations, weights);

  require(atlas.activation_count() == 3, "activation domain size mismatch");
  require(atlas.weight_count() == 3, "weight domain size mismatch");
  require(atlas.lookup(1, 2) == 1.5f, "lookup value mismatch");

  const std::vector<std::uint16_t> activation_ids = {0, 1, 2, 1, 0};
  const std::vector<std::uint16_t> weight_ids = {2, 1, 0, 2, 1};
  const float native = atlas.dot_native(activation_ids, weight_ids);
  require(atlas.dot_lookup(activation_ids, weight_ids) == native,
          "scalar lookup is not exact");
  require(atlas.dot_lookup_block4(activation_ids, weight_ids) == native,
          "blocked lookup is not exact");

  const std::array<std::uint32_t, 8> parameters = {1, 2, 3, 4, 5, 6, 7, 8};
  const std::array<std::uint8_t, 8> widths = {4, 4, 4, 4, 4, 4, 4, 4};
  const auto address = product_atlas::direct_block_address(parameters, widths);
  require(address.slot == (address.key & 63u), "slot mapping mismatch");
  require(address.block_number == (address.key >> 6),
          "block mapping mismatch");
  require(address.byte_offset == address.block_number * 4096u,
          "byte offset mismatch");

  bool rejected = false;
  try {
    product_atlas::direct_block_address(parameters,
                                        {9, 9, 9, 9, 9, 9, 9, 9});
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "oversized key width was not rejected");

  std::cout << "all tests passed\n";
  return 0;
}
