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
import io.grpc.Metadata;
import io.grpc.Status.Code;
import io.grpc.StatusRuntimeException;
import io.grpc.protobuf.lite.ProtoLiteUtils;
import javax.annotation.Nullable;

/**
 * Exception to signal that a remote execution has failed with a certain status received from the
 * server, and other details, such as the action result and the server logs. The exception may be
 * retriable or not, depending on the status/details.
 */
public class ExecutionStatusException extends StatusRuntimeException {
  private final Status status;
  private final ExecuteResponse response;

  private static final Metadata.Key<com.google.rpc.Status> STATUS_DETAILS_KEY =
      Metadata.Key.of(
          "grpc-status-details-bin",
          ProtoLiteUtils.metadataMarshaller(com.google.rpc.Status.getDefaultInstance()));

  private static final Metadata getMetadata(Status status) {
    Metadata metadata = new Metadata();
    metadata.put(STATUS_DETAILS_KEY, status);
    return metadata;
  }

  public ExecutionStatusException(Status status, @Nullable ExecuteResponse response) {
    super(convertStatus(status, response), getMetadata(status));
    this.status = status;
    this.response = response;
  }

  private static io.grpc.Status convertStatus(Status status, @Nullable ExecuteResponse response) {
    io.grpc.Status result =
        io.grpc.Status.fromCodeValue(
            // Hack: convert to non-retriable exception on timeouts.
            isExecutionTimeout(status, response)
                ? Code.FAILED_PRECONDITION.value()
                : status.getCode());
    return result.withDescription(status.getMessage());
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
