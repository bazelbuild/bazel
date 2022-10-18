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

import com.google.devtools.build.lib.util.AbruptExitException;

/**
 * A gRPC server instance.
 */
public interface RPCServer {

  /** Start serving and block until the shutdown command is received. */
  void serve() throws AbruptExitException;

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
