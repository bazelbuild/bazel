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

import com.google.devtools.build.lib.remote.logging.RemoteExecutionLog.RpcCallDetails;
import com.google.devtools.build.lib.remote.logging.RemoteExecutionLog.WatchDetails;
import com.google.watcher.v1.ChangeBatch;
import com.google.watcher.v1.Request;

/** LoggingHandler for {@link google.watcher.v1.Watch} gRPC call. */
public class WatchHandler implements LoggingHandler<Request, ChangeBatch> {
  private final WatchDetails.Builder builder = WatchDetails.newBuilder();

  @Override
  public void handleReq(Request message) {
    builder.setRequest(message);
  }

  @Override
  public void handleResp(ChangeBatch message) {
    builder.addResponses(message);
  }

  @Override
  public RpcCallDetails getDetails() {
    return RpcCallDetails.newBuilder().setWatch(builder).build();
  }
}
