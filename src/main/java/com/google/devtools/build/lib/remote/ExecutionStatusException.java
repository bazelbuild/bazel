// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.remote;

import build.bazel.remote.execution.v2.ExecuteResponse;
import com.google.rpc.Status;
import io.grpc.Status.Code;
import io.grpc.StatusRuntimeException;
import io.grpc.protobuf.StatusProto;
import javax.annotation.Nullable;

/**
 * Exception to signal that a remote execution has failed with a certain status received from the
 * server, and other details, such as the action result and the server logs. The exception may be
 * retriable or not, depending on the status/details.
 */
public class ExecutionStatusException extends StatusRuntimeException {
  private final Status status;
  private final ExecuteResponse response;

  ExecutionStatusException(
      StatusRuntimeException e, Status original, @Nullable ExecuteResponse response) {
    super(e.getStatus(), e.getTrailers());
    this.status = original;
    this.response = response;
  }

  public ExecutionStatusException(Status status, @Nullable ExecuteResponse response) {
    this(StatusProto.toStatusRuntimeException(convertStatus(status, response)), status, response);
  }

  private static Status convertStatus(Status status, @Nullable ExecuteResponse response) {
    Status.Builder result = status.toBuilder();
    if (isExecutionTimeout(status, response)) {
      // Hack: convert to non-retriable exception on timeouts.
      result.setCode(Code.FAILED_PRECONDITION.value());
    }
    return result.build();
  }

  private static boolean isExecutionTimeout(Status status, @Nullable ExecuteResponse response) {
    return response != null
        && response.getStatus().equals(status)
        && status.getCode() == Code.DEADLINE_EXCEEDED.value();
  }

  public boolean isExecutionTimeout() {
    return isExecutionTimeout(status, response);
  }

  @Nullable
  public ExecuteResponse getResponse() {
    return response;
  }
}
