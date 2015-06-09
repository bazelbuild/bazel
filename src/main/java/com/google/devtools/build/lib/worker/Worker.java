// Copyright 2015 Google Inc. All rights reserved.
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
package com.google.devtools.build.lib.worker;

import com.google.common.base.Preconditions;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.lang.ProcessBuilder.Redirect;

/**
 * Interface to a worker process running as a child process.
 *
 * <p>A worker process must follow this protocol to be usable via this class: The worker process is
 * spawned on demand. The worker process is free to exit whenever necessary, as new instances will
 * be relaunched automatically. Communication happens via the WorkerProtocol protobuf, sent to and
 * received from the worker process via stdin / stdout.
 *
 * <p>Other code in Blaze can talk to the worker process via input / output streams provided by this
 * class.
 */
final class Worker {
  private final Process process;
  private final Thread shutdownHook;

  private Worker(Process process, Thread shutdownHook) {
    this.process = process;
    this.shutdownHook = shutdownHook;
  }

  static Worker create(WorkerKey key) throws IOException {
    Preconditions.checkNotNull(key);
    ProcessBuilder processBuilder = new ProcessBuilder(key.getArgs().toArray(new String[0]))
        .directory(key.getWorkDir().getPathFile())
        .redirectError(Redirect.INHERIT);
    processBuilder.environment().putAll(key.getEnv());

    final Process process = processBuilder.start();

    Thread shutdownHook = new Thread() {
      @Override
      public void run() {
        process.destroy();
      }
    };
    Runtime.getRuntime().addShutdownHook(shutdownHook);

    return new Worker(process, shutdownHook);
  }

  void destroy() {
    Runtime.getRuntime().removeShutdownHook(shutdownHook);
    process.destroy();
  }

  boolean isAlive() {
    // This is horrible, but Process.isAlive() is only available from Java 8 on and this is the
    // best we can do prior to that.
    try {
      process.exitValue();
      return false;
    } catch (IllegalThreadStateException e) {
      return true;
    }
  }

  InputStream getInputStream() {
    return process.getInputStream();
  }

  OutputStream getOutputStream() {
    return process.getOutputStream();
  }
}
