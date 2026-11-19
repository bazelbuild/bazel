// Copyright 2020 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.remote.common;

import build.bazel.remote.execution.v2.ExecuteRequest;
import build.bazel.remote.execution.v2.ExecuteResponse;
import build.bazel.remote.execution.v2.ServerCapabilities;
import com.google.common.util.concurrent.ListenableFuture;
import java.io.IOException;
import java.io.OutputStream;

/**
 * An interface for a remote execution protocol.
 *
 * <p>Implementations must be thread-safe.
 */
public interface RemoteExecutionClient {

  /** Returns the cache capabilities of the remote execution server */
  ServerCapabilities getServerCapabilities() throws IOException;

  /** Execute an action remotely using Remote Execution API. */
  ExecuteResponse executeRemotely(
      RemoteActionExecutionContext context, ExecuteRequest request, OperationObserver observer)
      throws IOException, InterruptedException;

  /**
   * Reads a live output stream of an executing action, as advertised in the {@code
   * stdout_stream_name} or {@code stderr_stream_name} of its {@link
   * build.bazel.remote.execution.v2.ExecuteOperationMetadata}, writing the output to {@code out} as
   * it arrives.
   *
   * <p>The returned future completes when the server ends the stream and fails if the stream can't
   * be read. Cancelling it stops the read.
   */
  ListenableFuture<Void> readOutputStream(
      RemoteActionExecutionContext context, String resourceName, OutputStream out);

  /** Close resources associated with the remote execution client. */
  void close();
}
