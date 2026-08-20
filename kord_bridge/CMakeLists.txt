cmake_minimum_required(VERSION 3.17)
project(kord_bridge CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── kord-api ──────────────────────────────────────────────────────────────────
# Defaults to the kord-api submodule; override with -DKORD_API_DIR=/path/to/kord-api
# to build against an existing checkout.
set(KORD_API_DIR "${CMAKE_CURRENT_SOURCE_DIR}/kord-api" CACHE PATH
    "Path to the kord-api checkout")
if(NOT EXISTS "${KORD_API_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
        "kord-api not found at ${KORD_API_DIR}. Initialise the submodule with "
        "`git submodule update --init kord_bridge/kord-api`, or pass "
        "-DKORD_API_DIR=/path/to/kord-api.")
endif()

# Disable kord-api's own install rules so they don't pollute our prefix.
set(INSTALL_KORD_API      OFF CACHE BOOL "" FORCE)
set(INSTALL_KORD_PROTOCOL OFF CACHE BOOL "" FORCE)
set(KORD_WITH_EXAMPLES    OFF CACHE BOOL "" FORCE)
set(KORD_WITH_TESTS       OFF CACHE BOOL "" FORCE)

add_subdirectory(${KORD_API_DIR} ${CMAKE_CURRENT_BINARY_DIR}/kord-api)

# ── nanobind ──────────────────────────────────────────────────────────────────
# Prerequisite: pip install nanobind
find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE NB_DIR
    RESULT_VARIABLE NB_RESULT)
if(NOT NB_RESULT EQUAL 0)
    message(FATAL_ERROR "nanobind not found. Install it with: pip install nanobind")
endif()
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

# ── bridge static lib ─────────────────────────────────────────────────────────
add_library(kord_bridge_lib STATIC src/kord_bridge.cpp)
target_include_directories(kord_bridge_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
# kord-api links Threads::Threads internally; we inherit it via PUBLIC linkage.
target_link_libraries(kord_bridge_lib PUBLIC kord-api)
# Required: static lib is linked into a shared object (_kord_bridge.so).
set_property(TARGET kord_bridge_lib PROPERTY POSITION_INDEPENDENT_CODE ON)

# ── Python extension _kord_bridge.so ─────────────────────────────────────────
# Drop STABLE_ABI: version-specific .so is simpler and works on all Python 3.10+.
nanobind_add_module(_kord_bridge NB_STATIC src/kord_bindings.cpp)
target_link_libraries(_kord_bridge PRIVATE kord_bridge_lib)

# kord-api is a shared library, so _kord_bridge.so has a runtime dependency on
# libkord-api.so.  Embed the build-tree path as RPATH so the .so is self-contained
# after install without requiring LD_LIBRARY_PATH.
set_target_properties(_kord_bridge PROPERTIES
    BUILD_RPATH  "${CMAKE_CURRENT_BINARY_DIR}/kord-api;${CMAKE_CURRENT_BINARY_DIR}/_deps/kord-protocol-build"
    INSTALL_RPATH "${CMAKE_CURRENT_BINARY_DIR}/kord-api;${CMAKE_CURRENT_BINARY_DIR}/_deps/kord-protocol-build"
    INSTALL_RPATH_USE_LINK_PATH ON
)

# Install into the active Python's site-packages so `import _kord_bridge` works
# regardless of working directory.  --component kord_bridge skips the
# kord-api/kord-protocol install rules that require root.
install(TARGETS _kord_bridge
    LIBRARY DESTINATION ${Python_SITEARCH}
    COMPONENT kord_bridge)
