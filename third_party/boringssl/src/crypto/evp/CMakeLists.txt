include_directories(../../include)

add_library(
  evp

  OBJECT

  algorithm.c
  digestsign.c
  evp.c
  evp_asn1.c
  evp_ctx.c
  p_dsa_asn1.c
  p_ec.c
  p_ec_asn1.c
  p_rsa.c
  p_rsa_asn1.c
  pbkdf.c
  sign.c
)


add_executable(
  evp_extra_test

  evp_extra_test.cc

  $<TARGET_OBJECTS:test_support>
)

add_executable(
  evp_test

  evp_test.cc

  $<TARGET_OBJECTS:test_support>
)

add_executable(
  pbkdf_test

  pbkdf_test.cc

  $<TARGET_OBJECTS:test_support>
)

target_link_libraries(evp_extra_test crypto)
target_link_libraries(evp_test crypto)
target_link_libraries(pbkdf_test crypto)
