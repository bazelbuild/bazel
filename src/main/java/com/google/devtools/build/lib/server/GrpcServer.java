// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.server;

import com.google.devtools.build.lib.runtime.CommandExecutor;
import com.google.devtools.build.lib.util.Clock;

import java.io.IOException;

/**
 * Interface for the gRPC server.
 *
 * <p>This is necessary so that Bazel kind of works during bootstrapping, at which time the
 * gRPC server is not compiled on so that we don't need gRPC for bootstrapping.
 */
public interface GrpcServer {

  /**
   * Factory class.
   *
   * Present so that we don't need to invoke a constructor with multiple arguments by reflection.
   */
  interface Factory {
    GrpcServer create(CommandExecutor commandExecutor, Clock clock, int port,
      String outputBase);
  }

  void serve() throws IOException;
  void terminate();
}
