// Copyright 2014 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.clock.Clock;
import com.google.devtools.build.lib.runtime.BlazeCommandDispatcher;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;

/**
 * A gRPC server instance.
 */
public interface RPCServer {
  /**
   * Factory class for the gRPC server.
   *
   * Present so that we don't need to invoke a constructor with multiple arguments by reflection.
   */
  interface Factory {
    RPCServer create(
        BlazeCommandDispatcher dispatcher,
        Clock clock,
        int port,
        Path serverDirectory,
        int maxIdleSeconds,
        boolean shutdownOnLowSysMem,
        boolean idleServerTasks)
        throws IOException;
  }

  /**
   * Start serving and block until the a shutdown command is received.
   */
  void serve() throws IOException;

  /**
   * Called when the server receives a SIGINT.
   */
  void interrupt();

  /**
   * Prepares for the server shutting down prematurely.
   *
   * <p>Used in <code>clean --expunge</code> where the server state is deleted from the disk and
   * we need to make sure that everything works during such an drastic measure.
   */
  void prepareForAbruptShutdown();
}
