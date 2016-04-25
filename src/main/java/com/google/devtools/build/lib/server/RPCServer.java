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

import com.google.devtools.build.lib.runtime.CommandExecutor;
import com.google.devtools.build.lib.util.Clock;
import com.google.devtools.build.lib.vfs.Path;

import java.io.IOException;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.logging.Logger;

/**
 * A server instance. Can either an AF_UNIX or a gRPC one.
 */
public abstract class RPCServer {
  private static final Logger LOG = Logger.getLogger(RPCServer.class.getName());

  /**
   * Factory class for the gRPC server.
   *
   * Present so that we don't need to invoke a constructor with multiple arguments by reflection.
   */
  public interface Factory {
    RPCServer create(CommandExecutor commandExecutor, Clock clock, int port, Path serverDirectory,
        int maxIdleSeconds) throws IOException;
  }

  protected RPCServer(Path serverDirectory) throws IOException {
    // server.pid was written in the C++ launcher after fork() but before exec() .
    // The client only accesses the pid file after connecting to the socket
    // which ensures that it gets the correct pid value.
    Path pidFile = serverDirectory.getRelative("server.pid");
    RPCServer.deleteAtExit(pidFile, /*deleteParent=*/ false);
  }

  /**
   * Schedule the specified file for (attempted) deletion at JVM exit.
   */
  protected static void deleteAtExit(final Path socketFile, final boolean deleteParent) {
    Runtime.getRuntime().addShutdownHook(new Thread() {
        @Override
        public void run() {
          try {
            socketFile.delete();
            if (deleteParent) {
              socketFile.getParentDirectory().delete();
            }
          } catch (IOException e) {
            printStack(e);
          }
        }
      });
  }

  static void printStack(IOException e) {
    /*
     * Hopefully this never happens. It's not very nice to just write this
     * to the user's console, but I'm not sure what better choice we have.
     */
    StringWriter err = new StringWriter();
    PrintWriter printErr = new PrintWriter(err);
    printErr.println("=======[BLAZE SERVER: ENCOUNTERED IO EXCEPTION]=======");
    e.printStackTrace(printErr);
    printErr.println("=====================================================");
    LOG.severe(err.toString());
  }

  /**
   * Start serving and block until the a shutdown command is received.
   */
  public abstract void serve() throws IOException;

  /**
   * Called when the server receives a SIGINT.
   */
  public abstract void interrupt();
}
