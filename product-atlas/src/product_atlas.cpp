#include "product_atlas/product_atlas.hpp"

#include <cstring>
#include <limits>
#include <stdexcept>

namespace product_atlas {

std::uint16_t float_to_bf16(float value) {
  std::uint32_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));

  const std::uint32_t exponent = bits & 0x7f800000u;
  const std::uint32_t mantissa = bits & 0x007fffffu;
  if (exponent == 0x7f800000u && mantissa != 0) {
    return static_cast<std::uint16_t>((bits >> 16) | 0x0040u);
  }

  const std::uint32_t tie = (bits >> 16) & 1u;
  bits += 0x7fffu + tie;
  return static_cast<std::uint16_t>(bits >> 16);
}

float bf16_to_float(std::uint16_t bits) {
  const std::uint32_t expanded = static_cast<std::uint32_t>(bits) << 16;
  float value = 0.0f;
  std::memcpy(&value, &expanded, sizeof(value));
  return value;
}

ProductAtlas::ProductAtlas(std::vector<std::uint16_t> activations,
                           std::vector<std::uint16_t> weights)
    : activations_(std::move(activations)), weights_(std::move(weights)) {
  if (activations_.empty() || weights_.empty()) {
    throw std::invalid_argument("atlas domains must not be empty");
  }
  if (activations_.size() > std::numeric_limits<std::uint16_t>::max() ||
      weights_.size() > std::numeric_limits<std::uint16_t>::max()) {
    throw std::invalid_argument("atlas domains must fit in 16-bit IDs");
  }

  products_.resize(activations_.size() * weights_.size());
  for (std::size_t a = 0; a < activations_.size(); ++a) {
    const float activation = bf16_to_float(activations_[a]);
    for (std::size_t w = 0; w < weights_.size(); ++w) {
      products_[a * weights_.size() + w] =
          activation * bf16_to_float(weights_[w]);
    }
  }
}

std::size_t ProductAtlas::activation_count() const noexcept {
  return activations_.size();
}

std::size_t ProductAtlas::weight_count() const noexcept {
  return weights_.size();
}

std::size_t ProductAtlas::bytes() const noexcept {
  return products_.size() * sizeof(float);
}

float ProductAtlas::lookup(std::uint16_t activation_id,
                           std::uint16_t weight_id) const {
  if (activation_id >= activations_.size() || weight_id >= weights_.size()) {
    throw std::out_of_range("atlas ID out of range");
  }
  return products_[static_cast<std::size_t>(activation_id) * weights_.size() +
                   weight_id];
}

void ProductAtlas::validate_ids(
    const std::vector<std::uint16_t>& activation_ids,
    const std::vector<std::uint16_t>& weight_ids) const {
  if (activation_ids.size() != weight_ids.size()) {
    throw std::invalid_argument("dot-product inputs must have equal length");
  }
  for (std::size_t i = 0; i < activation_ids.size(); ++i) {
    if (activation_ids[i] >= activations_.size() ||
        weight_ids[i] >= weights_.size()) {
      throw std::out_of_range("atlas ID out of range");
    }
  }
}

float ProductAtlas::dot_lookup(
    const std::vector<std::uint16_t>& activation_ids,
    const std::vector<std::uint16_t>& weight_ids) const {
  validate_ids(activation_ids, weight_ids);
  float sum = 0.0f;
  for (std::size_t i = 0; i < activation_ids.size(); ++i) {
    sum += products_[static_cast<std::size_t>(activation_ids[i]) *
                         weights_.size() +
                     weight_ids[i]];
  }
  return sum;
}

float ProductAtlas::dot_lookup_block4(
    const std::vector<std::uint16_t>& activation_ids,
    const std::vector<std::uint16_t>& weight_ids) const {
  validate_ids(activation_ids, weight_ids);
  float sum = 0.0f;
  std::size_t i = 0;
  for (; i + 3 < activation_ids.size(); i += 4) {
    // Keep accumulation order identical to the scalar/native path.
    sum += products_[static_cast<std::size_t>(activation_ids[i]) *
                         weights_.size() + weight_ids[i]];
    sum += products_[static_cast<std::size_t>(activation_ids[i + 1]) *
                         weights_.size() + weight_ids[i + 1]];
    sum += products_[static_cast<std::size_t>(activation_ids[i + 2]) *
                         weights_.size() + weight_ids[i + 2]];
    sum += products_[static_cast<std::size_t>(activation_ids[i + 3]) *
                         weights_.size() + weight_ids[i + 3]];
  }
  for (; i < activation_ids.size(); ++i) {
    sum += products_[static_cast<std::size_t>(activation_ids[i]) *
                         weights_.size() + weight_ids[i]];
  }
  return sum;
}

float ProductAtlas::dot_native(
    const std::vector<std::uint16_t>& activation_ids,
    const std::vector<std::uint16_t>& weight_ids) const {
  validate_ids(activation_ids, weight_ids);
  float sum = 0.0f;
  for (std::size_t i = 0; i < activation_ids.size(); ++i) {
    sum += bf16_to_float(activations_[activation_ids[i]]) *
           bf16_to_float(weights_[weight_ids[i]]);
  }
  return sum;
}

BlockAddress direct_block_address(
    const std::array<std::uint32_t, 8>& parameters,
    const std::array<std::uint8_t, 8>& widths) {
  std::uint64_t key = 0;
  std::uint32_t shift = 0;
  for (std::size_t i = 0; i < parameters.size(); ++i) {
    const std::uint8_t width = widths[i];
    if (width > 32 || shift + width > 64) {
      throw std::invalid_argument("parameter widths must total at most 64 bits");
    }
    const std::uint64_t limit =
        width == 32 ? (std::uint64_t{1} << 32)
                    : (std::uint64_t{1} << width);
    if (parameters[i] >= limit) {
      throw std::out_of_range("parameter does not fit declared width");
    }
    key |= static_cast<std::uint64_t>(parameters[i]) << shift;
    shift += width;
  }

  return BlockAddress{key, key >> 6, static_cast<std::uint32_t>(key & 63u),
                      (key >> 6) << 12};
}

}  // namespace product_atlas
