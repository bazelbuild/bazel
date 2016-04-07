include_directories(../../include)

add_library(
  pkcs8

  OBJECT

  pkcs8.c
  p8_pkey.c
  p5_pbe.c
  p5_pbev2.c
)

add_executable(
  pkcs12_test

  pkcs12_test.cc

  $<TARGET_OBJECTS:test_support>
)

add_executable(
  pkcs8_test

  pkcs8_test.cc
)

target_link_libraries(pkcs8_test crypto)
target_link_libraries(pkcs12_test crypto)
