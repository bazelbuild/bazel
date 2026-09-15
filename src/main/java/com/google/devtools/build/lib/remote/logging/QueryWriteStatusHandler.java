// Copyright 2021 The Bazel Authors. All rights reserved.
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

import com.google.bytestream.ByteStreamProto.QueryWriteStatusRequest;
import com.google.bytestream.ByteStreamProto.QueryWriteStatusResponse;
import com.google.devtools.build.lib.remote.logging.RemoteExecutionLog.QueryWriteStatusDetails;
import com.google.devtools.build.lib.remote.logging.RemoteExecutionLog.RpcCallDetails;

/** LoggingHandler for {@link google.bytestream.QueryWriteStatus} gRPC call. */
public class QueryWriteStatusHandler
    implements LoggingHandler<QueryWriteStatusRequest, QueryWriteStatusResponse> {
  private final QueryWriteStatusDetails.Builder builder = QueryWriteStatusDetails.newBuilder();

  @Override
  public void handleReq(QueryWriteStatusRequest message) {
    builder.setRequest(message);
  }

  @Override
  public void handleResp(QueryWriteStatusResponse message) {
    builder.setResponse(message);
  }

  @Override
  public RpcCallDetails getDetails() {
    return RpcCallDetails.newBuilder().setQueryWriteStatus(builder).build();
  }
}
