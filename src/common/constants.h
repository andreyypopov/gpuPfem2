#ifndef CONSTANTS_H
#define CONSTANTS_H

struct CONSTANTS {
    static constexpr double DOUBLE_MIN = 2e-6;
    static constexpr double ONE_THIRD = 0.3333333333333333;                 //!< Math constant \f$\frac13\f$
    static constexpr double PI = 3.14159265358979323846;                    //!< Math constant \f$\pi\f$
    static constexpr double TWO_PI = 6.28318530717958647692;                //!< Math constant \f$2\pi\f$

    static constexpr int MAX_SIMPLE_NEIGHBORS_PER_CELL = 12;
    static constexpr double MEMORY_REALLOCATION_COEFFICIENT = 1.1;
    static constexpr int MAX_GAUSS_POINTS = 13;
    static constexpr int MAX_CELL_DIVISION_LEVEL = 4;
    static constexpr int MAX_PARTICLES_PER_CELL = MAX_CELL_DIVISION_LEVEL * MAX_CELL_DIVISION_LEVEL;
};

#endif // CONSTANTS_H
