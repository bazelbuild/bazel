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

import com.google.devtools.build.lib.remote.logging.RpcLogEntry.LogEntry;
import com.google.devtools.build.lib.remote.logging.RpcLogEntry.LogEntry.ExecuteEntry;
import com.google.devtools.remoteexecution.v1test.ExecuteRequest;
import com.google.longrunning.Operation;

public class ExecuteHandler<ReqT, RespT> implements LoggingHandler<ReqT, RespT> {
  ExecuteEntry.Builder executeEntryBuilder = ExecuteEntry.newBuilder();

  @Override
  public void handleReq(ReqT message) {
    executeEntryBuilder.setRequest((ExecuteRequest) message);
  }

  @Override
  public void handleResp(RespT message) {
    executeEntryBuilder.setOperation((Operation) message);
  }

  @Override
  public LogEntry getEntry() {
    return LogEntry.newBuilder().setExecuteEntry(executeEntryBuilder).build();
  }
}
