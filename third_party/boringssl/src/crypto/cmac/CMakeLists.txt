include_directories(../../include)

add_library(
  cmac

  OBJECT

  cmac.c
)

add_executable(
  cmac_test

  cmac_test.cc

  $<TARGET_OBJECTS:test_support>
)

target_link_libraries(cmac_test crypto)
