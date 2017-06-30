// Copyright 2016 The Bazel Authors. All rights reserved.
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

import io.grpc.Status.Code;
import io.grpc.StatusRuntimeException;
import java.io.IOException;

/** An exception to indicate failed retry attempts. */
public final class RetryException extends IOException {
  private final int attempts;

  RetryException(Throwable cause, int retryAttempts) {
    super(cause);
    this.attempts = retryAttempts + 1;
  }

  public int getAttempts() {
    return attempts;
  }

  public boolean causedByStatusCode(Code code) {
    return getCause() instanceof StatusRuntimeException
          && ((StatusRuntimeException) getCause()).getStatus().getCode() == code;
  }

  @Override
  public String toString() {
    return String.format("after %d attempts: %s", attempts, getCause());
  }
}
