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

import build.bazel.remote.execution.v2.FindMissingBlobsRequest;
import build.bazel.remote.execution.v2.FindMissingBlobsResponse;
import com.google.devtools.build.lib.remote.logging.RemoteExecutionLog.FindMissingBlobsDetails;
import com.google.devtools.build.lib.remote.logging.RemoteExecutionLog.RpcCallDetails;

/**
 * LoggingHandler for {@link
 * google.devtools.remoteexecution.v1test.ContentAddressableStorage.FindMissingBlobs} gRPC call.
 */
public class FindMissingBlobsHandler
    implements LoggingHandler<FindMissingBlobsRequest, FindMissingBlobsResponse> {

  private final FindMissingBlobsDetails.Builder builder = FindMissingBlobsDetails.newBuilder();

  @Override
  public void handleReq(FindMissingBlobsRequest message) {
    builder.setRequest(message);
  }

  @Override
  public void handleResp(FindMissingBlobsResponse message) {
    builder.setResponse(message);
  }

  @Override
  public RpcCallDetails getDetails() {
    return RpcCallDetails.newBuilder().setFindMissingBlobs(builder).build();
  }
}
