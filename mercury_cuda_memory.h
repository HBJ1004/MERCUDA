#pragma once
#include <cstddef>
#include <vector>

// All upload callers, including compact encounters, use the same five-slot ABI.
constexpr std::size_t MercuryForceComponents = 5;
inline std::vector<double> mercury_zero_force_parameters(std::size_t bodies) {
    return std::vector<double>(MercuryForceComponents*bodies,0.0);
}
