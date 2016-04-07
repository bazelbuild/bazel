include_directories(../../include)

add_library(
  rsa

  OBJECT

  rsa.c
  rsa_impl.c
  blinding.c
  padding.c
  rsa_asn1.c
)

add_executable(
  rsa_test

  rsa_test.cc

  $<TARGET_OBJECTS:test_support>
)

target_link_libraries(rsa_test crypto)
