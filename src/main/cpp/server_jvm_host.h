// Copyright 2026 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef BAZEL_SRC_MAIN_CPP_SERVER_JVM_HOST_H_
#define BAZEL_SRC_MAIN_CPP_SERVER_JVM_HOST_H_

namespace blaze {

// On Windows, the client starts the server JVM by running bazel-server.exe
// from the install base instead of the JDK's java.exe. It calls into the JDK's
// jli.dll exactly like java.exe does, but its app manifest sets the active code
// page to UTF-8, which the JVM uses for file names and command line arguments.
//
// The client passes the server javabase in this environment variable and
// otherwise runs bazel-server.exe with the same command line as java.exe.
inline constexpr char kServerJvmHostJavabaseEnvVar[] =
    "BAZEL_SERVER_JVM_HOST_JAVABASE";

// The path of bazel-server.exe relative to the install base.
//
// The Java launcher first looks for the JDK in the parent directory of the
// directory containing the executable, and only then relative to jli.dll.
// bazel-server.exe is in its own directory so that the first lookup fails.
inline constexpr char kServerJvmHostPath[] = "jvm_host/bin/bazel-server.exe";

}  // namespace blaze

#endif  // BAZEL_SRC_MAIN_CPP_SERVER_JVM_HOST_H_
