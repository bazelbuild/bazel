add_subdirectory(bio)
add_subdirectory(blowfish)
add_subdirectory(cast)
add_subdirectory(des)
add_subdirectory(rsa)
add_subdirectory(xts)

add_library(
  decrepit

  $<TARGET_OBJECTS:bio_decrepit>
  $<TARGET_OBJECTS:blowfish>
  $<TARGET_OBJECTS:cast>
  $<TARGET_OBJECTS:des_decrepit>
  $<TARGET_OBJECTS:rsa_decrepit>
  $<TARGET_OBJECTS:xts>
)
