#include <fcntl.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/util/delimited_message_util.h>
#include <iostream>
#include "src/main/protobuf/remote_execution_log.pb.h"

int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cerr << "error: usage: " << argv[0] << " GRPC_LOG" << std::endl;
    return EXIT_FAILURE;
  }

  auto filepath = argv[1];
  auto fp = open(filepath, O_RDONLY);
  if (fp == -1) {
    std::cerr << "error: failed to open '" << filepath
              << "' for reading with code " << errno << std::endl;
    return EXIT_FAILURE;
  }

  remote_logging::LogEntry entry;
  google::protobuf::io::FileInputStream file_input_stream(fp);
  while (google::protobuf::util::ParseDelimitedFromZeroCopyStream(
      &entry, &file_input_stream, nullptr)) {
    std::cout << entry.DebugString() << std::endl;
  }

  return EXIT_SUCCESS;
}
