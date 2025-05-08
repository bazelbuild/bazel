// Copyright 2023 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.util.io;

import com.google.protobuf.Any;

/**
 * Consumer of {@link Any} protos that sends messages to the build tool's gRPC client.
 *
 * <p>Instances of this interface must be <em>thread-safe</em>.
 */
@FunctionalInterface
public interface CommandExtensionReporter {

  /** Extension reporter that drops all extensions. */
  CommandExtensionReporter NO_OP_COMMAND_EXTENSION_REPORTER = (any) -> {};

  /** Writes the command extension to the client in a gRPC RunResponse. */
  void report(Any commandExtension);
}
