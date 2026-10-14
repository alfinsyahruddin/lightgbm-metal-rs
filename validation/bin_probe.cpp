// Test-only instrumentation of the unchanged upstream numeric bin mapper.
#include <LightGBM/bin.h>
#include <LightGBM/utils/log.h>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

int main() {
  LightGBM::Log::ResetLogLevel(LightGBM::LogLevel::Fatal);
  int rows, max_bin;
  if (!(std::cin >> rows >> max_bin) || rows < 1) return 1;
  std::vector<double> original, sampled;
  for (int i = 0; i < rows; ++i) {
    std::string token;
    if (!(std::cin >> token)) return 2;
    double value = std::stod(token);
    original.push_back(value);
    if (std::isnan(value) || std::fabs(value) > LightGBM::kZeroThreshold) sampled.push_back(value);
  }
  LightGBM::BinMapper mapper;
  mapper.FindBin(sampled.data(), static_cast<int>(sampled.size()), rows, max_bin,
                 3, 1, false, LightGBM::BinType::NumericalBin, true, false, {});
  std::cout << std::setprecision(17) << "{\"missing\":"
            << (mapper.missing_type() == LightGBM::MissingType::NaN ? "true" : "false")
            << ",\"upper_bounds\":[";
  for (int i = 0; i < mapper.num_bin(); ++i) {
    if (i) std::cout << ',';
    double value = mapper.BinToValue(i);
    if (std::isfinite(value)) std::cout << value;
    else std::cout << "null";
  }
  std::cout << "],\"bins\":[";
  for (int i = 0; i < rows; ++i) {
    if (i) std::cout << ',';
    std::cout << mapper.ValueToBin(original[i]);
  }
  std::cout << "]}\n";
}
