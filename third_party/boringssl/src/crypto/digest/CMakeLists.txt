include_directories(../../include)

add_library(
  digest

  OBJECT

  digest.c
  digests.c
)

add_executable(
  digest_test

  digest_test.cc

  $<TARGET_OBJECTS:test_support>
)

target_link_libraries(digest_test crypto)
