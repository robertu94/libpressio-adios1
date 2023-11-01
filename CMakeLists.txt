## template for a CMake C++ Library
cmake_minimum_required(VERSION 3.14)
project(libpressio_adios1 VERSION "0.0.2" LANGUAGES CXX C)

#correct was to set a default build type
# https://blog.kitware.com/cmake-and-the-default-build-type/
set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "No build type was set. Setting build type to ${default_build_type}.")
  set(CMAKE_BUILD_TYPE ${default_build_type} CACHE 
    STRING "Choose the type to build" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")

include(CTest)
include(GNUInstallDirs)
include(CheckCXXCompilerFlag)

#compiler flags and standard conformance checks
check_cxx_compiler_flag("-fno-omit-frame-pointer" HAVE_NO_OMIT_FRAME_POINTER)
set(NO_OMIT_FRAME_POINTER_FLAG "")
if(HAVE_NO_OMIT_FRAME_POINTER)
  set(NO_OMIT_FRAME_POINTER_FLAG "-fno-omit-frame-pointer")
endif()

if(NOT USE_SANITIZER)
  set(USE_SANITIZER "" CACHE STRING "use a build sanitizer" FORCE)
endif()
set_property(CACHE USE_SANITIZER PROPERTY STRINGS "" "Address" "Thread" "Undefined" "Leak" "Memory")
if (USE_SANITIZER STREQUAL "Address")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
elseif (USE_SANITIZER STREQUAL "Thread")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=thread")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=thread")
elseif (USE_SANITIZER STREQUAL "Undefined")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=undefined")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=undefined")
elseif (USE_SANITIZER STREQUAL "Leak")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=leak")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=leak")
elseif (USE_SANITIZER STREQUAL "Memory")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=memory")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=memory")
endif()


set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BUILD_SHARED_LIBS "build libpressio_adios1 as a shared library" ON)

find_package(LibPressio REQUIRED)
find_package(std_compat REQUIRED)
find_package(MPI REQUIRED)
find_package(ADIOS REQUIRED)

add_library(libpressio_adios1
  ./src/libpressio_adios1.cc
  ./include/libpressio_adios1.h
  )
target_link_libraries(libpressio_adios1
    PUBLIC LibPressio::libpressio ${ADIOS_LIBRARIES} MPI::MPI_CXX
  )
target_include_directories(libpressio_adios1
    PUBLIC ${ADIOS_INCLUDE_DIRS}
  )
target_link_libraries(libpressio_adios1
  PRIVATE
  )
target_compile_features(libpressio_adios1 PUBLIC cxx_std_17)
target_include_directories(libpressio_adios1 
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> 
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/libpressio_adios1>
  )
target_compile_options(libpressio_adios1 PRIVATE 
  $<$<CONFIG:Debug>: -Wall  -Wextra -Wpedantic>
  $<$<CONFIG:RelWithDebInfo>: ${NO_OMIT_FRAME_POINTER_FLAG}>
  )

option(LIBPRESSIO_adios1_WERROR ON)
if(LIBPRESSIO_adios1_WERROR)
  target_compile_options(libpressio_adios1 PRIVATE 
    $<$<CONFIG:Debug>: -Werror>
    )
endif()

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/src/libpressio_adios1_version.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/libpressio_adios1_version.h
  )
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/libpressio_adios1.pc.in
  ${CMAKE_CURRENT_BINARY_DIR}/libpressio_adios1.pc
  @ONLY
  )

install(TARGETS libpressio_adios1 EXPORT libpressio_adios1Targets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  )
include(CMakePackageConfigHelpers)
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/libpressio_adios1Config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/libpressio_adios1Config.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/libpressio_adios1
)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/libpressio_adios1ConfigVersion.cmake"
  VERSION "${PROJECT_VERSION}"
  COMPATIBILITY AnyNewerVersion
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/libpressio_adios1Config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/libpressio_adios1ConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/libpressio_adios1
)
install(EXPORT libpressio_adios1Targets NAMESPACE libpressio_adios1:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/libpressio_adios1)
install(DIRECTORY include/ DESTINATION
  ${CMAKE_INSTALL_INCLUDEDIR}/libpressio_adios1)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/libpressio_adios1_version.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/libpressio_adios1)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libpressio_adios1.pc DESTINATION ${CMAKE_INSTALL_PREFIX}/share/pkgconfig)


if(BUILD_TESTING)
  add_subdirectory(test)
endif()

option(USE_CLANG_TIDY "include clang-tidy warnings in the build log" OFF)
if(USE_CLANG_TIDY)
  find_program(CLANG_TIDY clang-tidy)
  set_target_properties(libpressio_adios1 PROPERTIES CXX_CLANG_TIDY "${CLANG_TIDY}")
  set_target_properties(libpressio_adios1 PROPERTIES C_CLANG_TIDY "${CLANG_TIDY}")
endif()

option(USE_INCLUDE_WHAT_YOU_USE "include include-what-you-use warnings in the build log" OFF)
if(USE_INCLUDE_WHAT_YOU_USE)
  find_program(INCLUDE_WHAT_YOU_USE include-what-you-use)
  set_target_properties(libpressio_adios1 PROPERTIES CXX_INCLUDE_WHAT_YOU_USE "${INCLUDE_WHAT_YOU_USE}")
  set_target_properties(libpressio_adios1 PROPERTIES C_INCLUDE_WHAT_YOU_USE "${INCLUDE_WHAT_YOU_USE}")
endif()


option(BUILD_DOCS "build the documentation" OFF)
if(BUILD_DOCS)
  find_package(Doxygen REQUIRED dot)
  set(DOXYGEN_MAN_LINKS YES)
  set(DOXYGEN_GENERATE_MAN YES)
  set(DOXYGEN_GENERATE_HTML YES)
  set(DOXYGEN_EXTRACT_LOCAL_METHODS YES)
  set(DOXYGEN_EXTRACT_STATIC YES)
  set(DOXYGEN_MACRO_EXPANSION YES)
  set(DOXYGEN_USE_MDFILE_AS_MAINPAGE README.md)
  doxygen_add_docs(
    docs
    ${PROJECT_SOURCE_DIR}/README.md
    ${PROJECT_SOURCE_DIR}/include
    COMMENT "Generate Documenation"
  )
endif()
# vim: ft=cmake :
