// Copyright 2026 The Bazel Authors. All rights reserved.
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

import build.bazel.remote.execution.v2.SpliceBlobRequest;
import build.bazel.remote.execution.v2.SpliceBlobResponse;
import com.google.devtools.build.lib.remote.logging.RemoteExecutionLog.SpliceBlobDetails;
import com.google.devtools.build.lib.remote.logging.RemoteExecutionLog.RpcCallDetails;

/**
 * LoggingHandler for {@link
 * build.bazel.remote.execution.v2.ContentAddressableStorage.SpliceBlob} gRPC call.
 */
public class SpliceBlobHandler implements LoggingHandler<SpliceBlobRequest, SpliceBlobResponse> {

  private final SpliceBlobDetails.Builder builder = SpliceBlobDetails.newBuilder();

  @Override
  public void handleReq(SpliceBlobRequest message) {
    builder.setRequest(message);
  }

  @Override
  public void handleResp(SpliceBlobResponse message) {
    builder.setResponse(message);
  }

  @Override
  public RpcCallDetails getDetails() {
    return RpcCallDetails.newBuilder().setSpliceBlob(builder).build();
  }
}
