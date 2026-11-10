#include "product_atlas/product_atlas.hpp"

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

std::vector<std::uint16_t> make_domain(std::size_t size, std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> values(-2.0f, 2.0f);
  std::vector<std::uint16_t> domain(size);
  for (auto& value : domain) {
    value = product_atlas::float_to_bf16(values(rng));
  }
  return domain;
}

template <typename Function>
double time_ms(Function&& function, std::size_t iterations, float& checksum) {
  const auto start = Clock::now();
  for (std::size_t i = 0; i < iterations; ++i) {
    checksum += function();
  }
  const auto stop = Clock::now();
  return std::chrono::duration<double, std::milli>(stop - start).count();
}

}  // namespace

int main(int argc, char** argv) {
  const std::size_t vector_size =
      argc > 1 ? static_cast<std::size_t>(std::stoull(argv[1])) : 4096;
  const std::size_t iterations =
      argc > 2 ? static_cast<std::size_t>(std::stoull(argv[2])) : 2000;
  const std::size_t domain_size = 512;

  auto activations = make_domain(domain_size, 7);
  auto weights = make_domain(domain_size, 11);
  product_atlas::ProductAtlas atlas(std::move(activations), std::move(weights));

  std::mt19937 rng(19);
  std::uniform_int_distribution<std::uint16_t> ids(
      0, static_cast<std::uint16_t>(domain_size - 1));
  std::vector<std::uint16_t> activation_ids(vector_size);
  std::vector<std::uint16_t> weight_ids(vector_size);
  for (std::size_t i = 0; i < vector_size; ++i) {
    activation_ids[i] = ids(rng);
    weight_ids[i] = ids(rng);
  }

  const float expected = atlas.dot_native(activation_ids, weight_ids);
  if (atlas.dot_lookup(activation_ids, weight_ids) != expected ||
      atlas.dot_lookup_block4(activation_ids, weight_ids) != expected) {
    std::cerr << "exactness check failed\n";
    return 1;
  }

  float checksum = 0.0f;
  const double native_ms = time_ms(
      [&] { return atlas.dot_native(activation_ids, weight_ids); }, iterations,
      checksum);
  const double lookup_ms = time_ms(
      [&] { return atlas.dot_lookup(activation_ids, weight_ids); }, iterations,
      checksum);
  const double block4_ms = time_ms(
      [&] { return atlas.dot_lookup_block4(activation_ids, weight_ids); },
      iterations, checksum);

  std::cout << std::fixed << std::setprecision(3)
            << "Product Atlas synthetic benchmark\n"
            << "vector_size=" << vector_size << " iterations=" << iterations
            << " atlas_mib=" << (atlas.bytes() / 1048576.0) << "\n"
            << "native_ms=" << native_ms << "\n"
            << "lookup_scalar_ms=" << lookup_ms << "\n"
            << "lookup_block4_ms=" << block4_ms << "\n"
            << "scalar_vs_native=" << (lookup_ms / native_ms) << "x\n"
            << "block4_vs_native=" << (block4_ms / native_ms) << "x\n"
            << "checksum=" << checksum << "\n";
  return 0;
}
