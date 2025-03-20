/*
 * Copyright 2019 The gRPC Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef NET_GRPC_COMPILER_JAVA_GENERATOR_H_
#define NET_GRPC_COMPILER_JAVA_GENERATOR_H_

#include <stdlib.h>  // for abort()
#include <iostream>
#include <string>

#include <google/protobuf/io/zero_copy_stream.h>
#include <google/protobuf/descriptor.h>

class LogHelper {
  std::ostream* os;

 public:
  LogHelper(std::ostream* os) : os(os) {}
  ~LogHelper() {
    *os << std::endl;
    ::abort();
  }
  std::ostream& get_os() {
    return *os;
  }
};

// Abort the program after logging the message if the given condition is not
// true. Otherwise, do nothing.
#define GRPC_CODEGEN_CHECK(x) !(x) && LogHelper(&std::cerr).get_os() \
                             << "CHECK FAILED: " << __FILE__ << ":" \
                             << __LINE__ << ": "

// Abort the program after logging the message.
#define GRPC_CODEGEN_FAIL GRPC_CODEGEN_CHECK(false)

namespace java_grpc_generator {

namespace impl {
namespace protobuf = google::protobuf;
} // namespace impl

enum ProtoFlavor {
  NORMAL, LITE
};

// Returns the package name of the gRPC services defined in the given file.
std::string ServiceJavaPackage(const impl::protobuf::FileDescriptor* file);

// Returns the name of the outer class that wraps in all the generated code for
// the given service.
std::string ServiceClassName(const impl::protobuf::ServiceDescriptor* service);

// Writes the generated service interface into the given ZeroCopyOutputStream
void GenerateService(const impl::protobuf::ServiceDescriptor* service,
                     impl::protobuf::io::ZeroCopyOutputStream* out,
                     ProtoFlavor flavor,
                     bool disable_version);

}  // namespace java_grpc_generator

#endif  // NET_GRPC_COMPILER_JAVA_GENERATOR_H_
