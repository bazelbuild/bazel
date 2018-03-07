// Copyright 2017 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.buildeventstream.transports;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.devtools.build.lib.buildeventstream.BuildEventTransport;
import com.google.devtools.build.lib.util.io.AsynchronousFileOutputStream;
import com.google.protobuf.Message;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Non-blocking file transport.
 *
 * <p>Implementors of this class need to implement {@code #sendBuildEvent(BuildEvent)} which
 * serializes the build event and writes it to file using {@link #writeData(byte[])}.
 */
abstract class FileTransport implements BuildEventTransport {

  private static final Logger logger = Logger.getLogger(FileTransport.class.getName());
  @VisibleForTesting
  final AsynchronousFileOutputStream out;

  FileTransport(String path) {
    try {
      out = new AsynchronousFileOutputStream(path);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  // Silent wrappers to AsynchronousFileOutputStream methods.

  protected void write(Message m) {
    try {
      out.write(m);
    } catch (Exception e) {
      logger.log(Level.SEVERE, e.getMessage(), e);
    }
  }

  protected void write(String s) {
    try {
      out.write(s);
    } catch (Exception e) {
      logger.log(Level.SEVERE, e.getMessage(), e);
    }
  }


  @Override
  public synchronized ListenableFuture<Void> close() {
    return Futures.catching(
        out.closeAsync(),
        Throwable.class,
        (t) -> {
          logger.log(Level.SEVERE, t.getMessage(), t);
          return null;
        },
        MoreExecutors.directExecutor());
  }

  @Override
  public void closeNow() {
    out.closeNow();
  }
}
