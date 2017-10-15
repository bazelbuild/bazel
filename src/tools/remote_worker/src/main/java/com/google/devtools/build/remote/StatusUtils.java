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

package com.google.devtools.build.remote;

import com.google.devtools.remoteexecution.v1test.Digest;
import com.google.protobuf.Any;
import com.google.rpc.BadRequest;
import com.google.rpc.BadRequest.FieldViolation;
import com.google.rpc.Code;
import com.google.rpc.Status;
import io.grpc.StatusException;
import io.grpc.protobuf.StatusProto;

/** Some utility methods to convert exceptions to Status results. */
final class StatusUtils {
  private StatusUtils() {}

  static StatusException internalError(Exception e) {
    return StatusProto.toStatusException(internalErrorStatus(e));
  }

  static Status internalErrorStatus(Exception e) {
    // StatusProto.fromThrowable returns null on non-status errors or errors with no trailers,
    // unlike Status.fromThrowable which returns the UNKNOWN code for these.
    Status st = StatusProto.fromThrowable(e);
    return st != null
        ? st
        : Status.newBuilder().setCode(Code.INTERNAL.getNumber()).setMessage(e.getMessage()).build();
  }

  static StatusException notFoundError(Digest digest) {
    return StatusProto.toStatusException(notFoundStatus(digest));
  }

  static com.google.rpc.Status notFoundStatus(Digest digest) {
    return Status.newBuilder()
        .setCode(Code.NOT_FOUND.getNumber())
        .setMessage("Digest not found:" + digest)
        .build();
  }

  static StatusException interruptedError(Digest digest) {
    return StatusProto.toStatusException(interruptedStatus(digest));
  }

  static com.google.rpc.Status interruptedStatus(Digest digest) {
    return Status.newBuilder()
        .setCode(Code.CANCELLED.getNumber())
        .setMessage("Server operation was interrupted for " + digest)
        .build();
  }

  static StatusException invalidArgumentError(String field, String desc) {
    return StatusProto.toStatusException(invalidArgumentStatus(field, desc));
  }

  static com.google.rpc.Status invalidArgumentStatus(String field, String desc) {
    FieldViolation v = FieldViolation.newBuilder().setField(field).setDescription(desc).build();
    return Status.newBuilder()
        .setCode(Code.INVALID_ARGUMENT.getNumber())
        .setMessage("invalid argument(s): " + field + ": " + desc)
        .addDetails(Any.pack(BadRequest.newBuilder().addFieldViolations(v).build()))
        .build();
  }
}
