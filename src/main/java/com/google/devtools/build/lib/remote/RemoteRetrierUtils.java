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

import io.grpc.Status;
import io.grpc.StatusException;
import io.grpc.StatusRuntimeException;

/** Methods useful when using the {@link RemoteRetrier}. */
public final class RemoteRetrierUtils {

  public static boolean causedByStatus(Throwable e, Status.Code expected) {
    if (e instanceof StatusRuntimeException) {
      return ((StatusRuntimeException) e).getStatus().getCode() == expected;
    } else if (e instanceof StatusException) {
      return ((StatusException) e).getStatus().getCode() == expected;
    } else if (e.getCause() != null) {
      return causedByStatus(e.getCause(), expected);
    }
    return false;
  }

  public static boolean causedByExecTimeout(Throwable e) {
    if (e instanceof ExecutionStatusException) {
      return ((ExecutionStatusException) e).isExecutionTimeout();
    } else if (e.getCause() != null) {
      return causedByExecTimeout(e.getCause());
    }
    return false;
  }
}
