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

import com.google.devtools.build.lib.runtime.BlazeCommandDispatcher;
import com.google.devtools.build.lib.util.io.OutErr;
import java.util.List;

/**
 * The {@link RPCServer} calls an arbitrary command implementing this
 * interface.
 */
public interface ServerCommand {

  /**
   * Executes the request, writing any output or error messages into err.
   * Returns 0 on success; any other value or exception indicates an error.
   */
  int exec(List<String> args, OutErr outErr, BlazeCommandDispatcher.LockingMode lockingMode,
      String clientDescription, long firstContactTime) throws InterruptedException;

  /**
   * Whether the server needs to be shut down.
   */
  boolean shutdown();
}
