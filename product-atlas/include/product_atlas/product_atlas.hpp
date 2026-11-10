#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace product_atlas {

// Converts between an IEEE-754 float and the upper 16-bit BF16 encoding.
// float_to_bf16 uses round-to-nearest-even.
std::uint16_t float_to_bf16(float value);
float bf16_to_float(std::uint16_t bits);

class ProductAtlas {
 public:
  ProductAtlas(std::vector<std::uint16_t> activations,
               std::vector<std::uint16_t> weights);

  std::size_t activation_count() const noexcept;
  std::size_t weight_count() const noexcept;
  std::size_t bytes() const noexcept;

  float lookup(std::uint16_t activation_id,
               std::uint16_t weight_id) const;

  float dot_lookup(const std::vector<std::uint16_t>& activation_ids,
                   const std::vector<std::uint16_t>& weight_ids) const;

  float dot_lookup_block4(
      const std::vector<std::uint16_t>& activation_ids,
      const std::vector<std::uint16_t>& weight_ids) const;

  float dot_native(const std::vector<std::uint16_t>& activation_ids,
                   const std::vector<std::uint16_t>& weight_ids) const;

 private:
  void validate_ids(const std::vector<std::uint16_t>& activation_ids,
                    const std::vector<std::uint16_t>& weight_ids) const;

  std::vector<std::uint16_t> activations_;
  std::vector<std::uint16_t> weights_;
  std::vector<float> products_;
};

struct BlockAddress {
  std::uint64_t key;
  std::uint64_t block_number;
  std::uint32_t slot;
  std::uint64_t byte_offset;
};

// Packs eight bounded parameters into a dense key. widths must total <= 64.
// For 64-byte records in a 4 KiB page, the low six key bits select the slot.
BlockAddress direct_block_address(
    const std::array<std::uint32_t, 8>& parameters,
    const std::array<std::uint8_t, 8>& widths);

}  // namespace product_atlas
