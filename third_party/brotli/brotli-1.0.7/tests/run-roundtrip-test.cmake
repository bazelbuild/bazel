set(ENV{QEMU_LD_PREFIX} "${BROTLI_WRAPPER_LD_PREFIX}")

execute_process(
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  COMMAND ${BROTLI_WRAPPER} ${BROTLI_CLI} --force --quality=${QUALITY} ${INPUT} --output=${OUTPUT}.br
  RESULT_VARIABLE result
  ERROR_VARIABLE result_stderr)
if(result)
  message(FATAL_ERROR "Compression failed: ${result_stderr}")
endif()

execute_process(
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  COMMAND ${BROTLI_WRAPPER} ${BROTLI_CLI} --force --decompress ${OUTPUT}.br --output=${OUTPUT}.unbr
  RESULT_VARIABLE result)
if(result)
  message(FATAL_ERROR "Decompression failed")
endif()

function(test_file_equality f1 f2)
  if(NOT CMAKE_VERSION VERSION_LESS 2.8.7)
    file(SHA512 "${f1}" f1_cs)
    file(SHA512 "${f2}" f2_cs)
    if(NOT "${f1_cs}" STREQUAL "${f2_cs}")
      message(FATAL_ERROR "Files do not match")
    endif()
  else()
    file(READ "${f1}" f1_contents)
    file(READ "${f2}" f2_contents)
    if(NOT "${f1_contents}" STREQUAL "${f2_contents}")
      message(FATAL_ERROR "Files do not match")
    endif()
  endif()
endfunction()

test_file_equality("${INPUT}" "${OUTPUT}.unbr")
