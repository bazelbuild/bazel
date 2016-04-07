include_directories(../../include)

add_library(
  dh

  OBJECT

  dh.c
  dh_impl.c
  params.c
  check.c
  dh_asn1.c
)

add_executable(
  dh_test

  dh_test.cc

  $<TARGET_OBJECTS:test_support>
)

target_link_libraries(dh_test crypto)
