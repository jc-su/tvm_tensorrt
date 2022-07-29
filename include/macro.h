// header guard
#ifndef MACRO_H
#define MACRO_H

#define CHECK(status)                                                                              \
    do {                                                                                           \
        auto ret = (status);                                                                       \
        if (ret != 0) {                                                                            \
            std::cerr << "Cuda failure: " << ret << std::endl;                                     \
            abort();                                                                               \
        }                                                                                          \
    } while (0)

#endif // MACRO_H