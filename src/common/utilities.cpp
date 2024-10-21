#include "utilities.h"

#include <algorithm>

std::string Utilities::intToString(int number, size_t width)
{
    const std::string numToStr = std::to_string(number);
    return std::string(width - std::min(width, numToStr.length()), '0') + numToStr;
}
