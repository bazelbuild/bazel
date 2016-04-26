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
package com.google.devtools.build.lib.runtime;

import com.google.devtools.build.lib.runtime.BlazeCommandDispatcher.ShutdownMethod;
import com.google.devtools.build.lib.server.ServerCommand;
import com.google.devtools.build.lib.util.io.OutErr;

import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.List;
import java.util.logging.Logger;

/**
 * Executes a Blaze command.
 *
 * <p>This is the common execution path between the gRPC server and the legacy AF_UNIX server.
 */
public class CommandExecutor implements ServerCommand {
  private static final Logger LOG = Logger.getLogger(CommandExecutor.class.getName());

  private ShutdownMethod shutdown;
  private final BlazeRuntime runtime;
  private final BlazeCommandDispatcher dispatcher;

  CommandExecutor(BlazeRuntime runtime, BlazeCommandDispatcher dispatcher) {
    this.shutdown = ShutdownMethod.NONE;
    this.runtime = runtime;
    this.dispatcher = dispatcher;
  }

  @Override
  public int exec(List<String> args, OutErr outErr, BlazeCommandDispatcher.LockingMode lockingMode,
      String clientDescription, long firstContactTime) throws InterruptedException {
    LOG.info(BlazeRuntime.getRequestLogString(args));

    try {
      return dispatcher.exec(args, outErr, lockingMode, clientDescription, firstContactTime);
    } catch (BlazeCommandDispatcher.ShutdownBlazeServerException e) {
      if (e.getCause() != null) {
        StringWriter message = new StringWriter();
        message.write("Shutting down due to exception:\n");
        PrintWriter writer = new PrintWriter(message, true);
        e.printStackTrace(writer);
        writer.flush();
        LOG.severe(message.toString());
      }
      shutdown = e.getMethod();
      runtime.shutdown();
      dispatcher.shutdown();
      return e.getExitStatus();
    }
  }

  @Override
  public ShutdownMethod shutdown() {
    return shutdown;
  }
}
