include_directories(../../include)

add_library(
  bytestring

  OBJECT

  ber.c
  cbs.c
  cbb.c
)

add_executable(
  bytestring_test

  bytestring_test.cc

  $<TARGET_OBJECTS:test_support>
)

target_link_libraries(bytestring_test crypto)
