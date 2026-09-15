// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.remote.logging;

import build.bazel.remote.execution.v2.WaitExecutionRequest;
import com.google.devtools.build.lib.remote.logging.RemoteExecutionLog.RpcCallDetails;
import com.google.devtools.build.lib.remote.logging.RemoteExecutionLog.WaitExecutionDetails;
import com.google.longrunning.Operation;

/** LoggingHandler for {@link build.bazel.remote.execution.v2.WaitExecution} gRPC call. */
public class WaitExecutionHandler implements LoggingHandler<WaitExecutionRequest, Operation> {
  private final WaitExecutionDetails.Builder builder = WaitExecutionDetails.newBuilder();

  @Override
  public void handleReq(WaitExecutionRequest message) {
    builder.setRequest(message);
  }

  @Override
  public void handleResp(Operation message) {
    builder.addResponses(message);
  }

  @Override
  public RpcCallDetails getDetails() {
    return RpcCallDetails.newBuilder().setWaitExecution(builder).build();
  }
}
