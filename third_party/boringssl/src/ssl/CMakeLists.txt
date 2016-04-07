include_directories(../include)

add_subdirectory(pqueue)

add_library(
  ssl

  custom_extensions.c
  d1_both.c
  d1_clnt.c
  d1_lib.c
  d1_meth.c
  d1_pkt.c
  d1_srtp.c
  d1_srvr.c
  dtls_record.c
  s3_both.c
  s3_clnt.c
  s3_enc.c
  s3_lib.c
  s3_meth.c
  s3_pkt.c
  s3_srvr.c
  ssl_aead_ctx.c
  ssl_asn1.c
  ssl_buffer.c
  ssl_cert.c
  ssl_cipher.c
  ssl_file.c
  ssl_lib.c
  ssl_rsa.c
  ssl_session.c
  ssl_stat.c
  ssl_txt.c
  t1_enc.c
  t1_lib.c
  tls_record.c

  $<TARGET_OBJECTS:pqueue>
)

add_executable(
  ssl_test

  ssl_test.cc

  $<TARGET_OBJECTS:test_support>
)

target_link_libraries(ssl_test ssl crypto)
