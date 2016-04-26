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

import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.runtime.BlazeCommandDispatcher.LockingMode;
import com.google.devtools.build.lib.runtime.BlazeCommandDispatcher.ShutdownMethod;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.util.io.OutErr;

import java.util.List;
import java.util.logging.Logger;

/**
 * An RPCService is a Java object that can process RPC requests.  Requests may
 * be of the form:
 * <pre>
 *   blaze <blaze-arguments>
 * </pre>
 * Requests are delegated to the ServerCommand instance provided
 * to the constructor.
 */
public final class RPCService {

  private static final Logger LOG = Logger.getLogger(RPCService.class.getName());
  private final ServerCommand appCommand;
  private ShutdownMethod shutdown = ShutdownMethod.NONE;

  public RPCService(ServerCommand appCommand) {
    this.appCommand = appCommand;
  }

  /**
   * The {@link #executeRequest(List, OutErr, long)} method may
   * throw this exception if a command is unknown to the RPC service.
   */
  public static class UnknownCommandException extends Exception {
    private static final long serialVersionUID = 1L;
    UnknownCommandException(String command) {
      super("Unknown command: " + command);
    }
  }

  /**
   * Executes the request; returns Unix like return codes (0 means success). May
   * also throw arbitrary exceptions.
   */
  public int executeRequest(List<String> request,
                            OutErr outErr,
                            long firstContactTime) throws Exception {
    if (shutdown != ShutdownMethod.NONE) {
      throw new IllegalStateException("Received request after shutdown.");
    }
    String command = Iterables.getFirst(request, "");
    if (appCommand != null && command.equals("blaze")) { // an application request
      // Blocking is done in the client for AF_UNIX communications, so if blockForLock would block,
      // something went wrong
      int result = appCommand.exec(
          request.subList(1, request.size()), outErr, LockingMode.ERROR_OUT, "AF_UNIX client",
          firstContactTime);
      ShutdownMethod commandShutdown = appCommand.shutdown();
      if (commandShutdown != ShutdownMethod.NONE) {  // an application shutdown request
        shutdown(commandShutdown);
      }
      return result;
    } else {
      throw new UnknownCommandException(command);
    }
  }

  /**
   * After executing this function, further requests will fail, and
   * {@link #getShutdown()} will the shutdown method passed in.
   */
  public void shutdown(ShutdownMethod method) {
    Preconditions.checkState(method != ShutdownMethod.NONE);
    if (shutdown != ShutdownMethod.NONE) {
      return;
    }
    LOG.info("RPC Service: shutting down ...");
    shutdown = method;
  }

  /**
   * Has this service been shutdown. If so, any call to
   * {@link #executeRequest(List, OutErr, long)} will result in an
   * {@link IllegalStateException}
   */
  public ShutdownMethod getShutdown() {
    return shutdown;
  }

}
