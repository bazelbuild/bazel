include_directories(../../include)

add_executable(
  bssl_shim

  async_bio.cc
  bssl_shim.cc
  packeted_bio.cc
  test_config.cc

  $<TARGET_OBJECTS:test_support>
)

target_link_libraries(bssl_shim ssl crypto)
