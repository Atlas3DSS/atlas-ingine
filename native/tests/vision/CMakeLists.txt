# Vision tower: kernels against FP64 oracles and the reference processor (GPU, small).
ingine_add_test(NAME vision_kernels SOURCES test_vision_kernels.cpp LIBS ingine_kernels_vision
                LABELS vision GPU ORACLE TIMEOUT 600)
