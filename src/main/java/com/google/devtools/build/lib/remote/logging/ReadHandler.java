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

import com.google.bytestream.ByteStreamProto.ReadRequest;
import com.google.bytestream.ByteStreamProto.ReadResponse;
import com.google.devtools.build.lib.remote.logging.RemoteExecutionLog.ReadDetails;
import com.google.devtools.build.lib.remote.logging.RemoteExecutionLog.RpcCallDetails;

/** LoggingHandler for {@link google.bytestream.Read} gRPC call. */
public class ReadHandler implements LoggingHandler<ReadRequest, ReadResponse> {
  private final ReadDetails.Builder builder = ReadDetails.newBuilder();
  private long numReads = 0;
  private long bytesRead = 0;

  @Override
  public void handleReq(ReadRequest message) {
    builder.setRequest(message);
  }

  @Override
  public void handleResp(ReadResponse message) {
    numReads++;
    bytesRead += message.getData().size();
  }

  @Override
  public RpcCallDetails getDetails() {
    builder.setNumReads(numReads);
    builder.setBytesRead(bytesRead);
    return RpcCallDetails.newBuilder().setRead(builder).build();
  }
}
