cmake_minimum_required(VERSION 3.15...3.27)
project(fastbbox_nanobind LANGUAGES CXX)

# Detect Python with Development.Module component
if (CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()

find_package(Python 3.9 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

# Configure optimized release build by default
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Detect the installed nanobind package and import it into CMake
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

# Build bbox module
# Use NOMINSIZE since our binding layer contains significant computational logic
# that benefits from speed optimization rather than size optimization
nanobind_add_module(bbox NOMINSIZE ${CMAKE_CURRENT_SOURCE_DIR}/bbox_nb.cpp)

# Build obb_bbox module
nanobind_add_module(obb_bbox NOMINSIZE ${CMAKE_CURRENT_SOURCE_DIR}/obb_bbox_nb.cpp)

# Install the modules
install(TARGETS bbox obb_bbox LIBRARY DESTINATION fastbbox)
