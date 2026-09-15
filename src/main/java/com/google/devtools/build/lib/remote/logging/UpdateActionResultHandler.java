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

import build.bazel.remote.execution.v2.ActionResult;
import build.bazel.remote.execution.v2.UpdateActionResultRequest;
import com.google.devtools.build.lib.remote.logging.RemoteExecutionLog.RpcCallDetails;
import com.google.devtools.build.lib.remote.logging.RemoteExecutionLog.UpdateActionResultDetails;

/**
 * LoggingHandler for {@link google.devtools.remoteexecution.v1test.ActionCache.UpdateActionResult}
 * gRPC call.
 */
public class UpdateActionResultHandler
    implements LoggingHandler<UpdateActionResultRequest, ActionResult> {

  private final UpdateActionResultDetails.Builder updateActiontResultDetailsBuilder =
      UpdateActionResultDetails.newBuilder();

  @Override
  public void handleReq(UpdateActionResultRequest message) {
    updateActiontResultDetailsBuilder.setRequest(message);
  }

  @Override
  public void handleResp(ActionResult message) {
    updateActiontResultDetailsBuilder.setResponse(message);
  }

  @Override
  public RpcCallDetails getDetails() {
    return RpcCallDetails.newBuilder()
        .setUpdateActionResult(updateActiontResultDetailsBuilder)
        .build();
  }
}
