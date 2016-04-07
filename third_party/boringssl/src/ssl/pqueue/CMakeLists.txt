include_directories(../../include)

add_library(
  pqueue

  OBJECT

  pqueue.c
)

add_executable(
  pqueue_test

  pqueue_test.c

  $<TARGET_OBJECTS:test_support>
)

target_link_libraries(pqueue_test ssl crypto)
