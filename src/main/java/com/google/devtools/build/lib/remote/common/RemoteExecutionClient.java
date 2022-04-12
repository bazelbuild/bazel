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
import java.io.IOException;

/**
 * An interface for a remote execution protocol.
 *
 * <p>Implementations must be thread-safe.
 */
public interface RemoteExecutionClient {

  /** Execute an action remotely using Remote Execution API. */
  ExecuteResponse executeRemotely(
      RemoteActionExecutionContext context, ExecuteRequest request, OperationObserver observer)
      throws IOException, InterruptedException;

  /** Close resources associated with the remote execution client. */
  void close();
}
