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

import com.google.bytestream.ByteStreamProto.WriteRequest;
import com.google.bytestream.ByteStreamProto.WriteResponse;
import com.google.devtools.build.lib.remote.logging.RemoteExecutionLog.RpcCallDetails;
import com.google.devtools.build.lib.remote.logging.RemoteExecutionLog.WriteDetails;
import java.util.LinkedHashSet;
import java.util.Set;

/** LoggingHandler for {@link google.bytestream.Write} gRPC call. */
public class WriteHandler implements LoggingHandler<WriteRequest, WriteResponse> {
  private final WriteDetails.Builder builder = WriteDetails.newBuilder();
  private final Set<String> resources = new LinkedHashSet<>();
  private long numWrites = 0;
  private long bytesSent = 0;

  @Override
  public void handleReq(WriteRequest message) {
    resources.add(message.getResourceName());

    numWrites++;
    bytesSent += message.getData().size();
  }

  @Override
  public void handleResp(WriteResponse message) {
    builder.setResponse(message);
  }

  @Override
  public RpcCallDetails getDetails() {
    builder.addAllResourceNames(resources);
    builder.setNumWrites(numWrites);
    builder.setBytesSent(bytesSent);
    return RpcCallDetails.newBuilder().setWrite(builder).build();
  }
}
