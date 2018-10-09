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

import build.bazel.remote.execution.v2.GetCapabilitiesRequest;
import build.bazel.remote.execution.v2.ServerCapabilities;
import com.google.devtools.build.lib.remote.logging.RemoteExecutionLog.GetCapabilitiesDetails;
import com.google.devtools.build.lib.remote.logging.RemoteExecutionLog.RpcCallDetails;

/**
 * LoggingHandler for {@link google.devtools.remoteexecution.v1test.ActionCache.GetCapabilities}
 * gRPC call.
 */
public class GetCapabilitiesHandler
    implements LoggingHandler<GetCapabilitiesRequest, ServerCapabilities> {

  private final GetCapabilitiesDetails.Builder builder = GetCapabilitiesDetails.newBuilder();

  @Override
  public void handleReq(GetCapabilitiesRequest message) {
    builder.setRequest(message);
  }

  @Override
  public void handleResp(ServerCapabilities message) {
    builder.setResponse(message);
  }

  @Override
  public RpcCallDetails getDetails() {
    return RpcCallDetails.newBuilder().setGetCapabilities(builder).build();
  }
}
