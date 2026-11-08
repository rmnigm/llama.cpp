set(TARGET llama-kv-control)
add_executable(${TARGET} kv-control.cpp)
target_link_libraries(${TARGET} PRIVATE llama-common ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
