// Copyright 2020 The Bazel Authors. All rights reserved.
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

import com.google.common.util.concurrent.ListeningScheduledExecutorService;
import com.google.devtools.build.lib.remote.common.BulkTransferException;
import com.google.protobuf.Any;
import com.google.protobuf.InvalidProtocolBufferException;
import com.google.protobuf.util.Durations;
import com.google.rpc.DebugInfo;
import com.google.rpc.Help;
import com.google.rpc.LocalizedMessage;
import com.google.rpc.PreconditionFailure;
import com.google.rpc.PreconditionFailure.Violation;
import com.google.rpc.RequestInfo;
import com.google.rpc.ResourceInfo;
import com.google.rpc.RetryInfo;
import com.google.rpc.Status;
import io.grpc.Status.Code;
import io.grpc.protobuf.StatusProto;

/** Specific retry logic for execute request with gapi Status. */
class ExecuteRetrier extends RemoteRetrier {

  private static final String VIOLATION_TYPE_MISSING = "MISSING";

  private static class RetryInfoBackoff implements Backoff {
    private final int maxRetryAttempts;
    int retries = 0;

    RetryInfoBackoff(int maxRetryAttempts) {
      this.maxRetryAttempts = maxRetryAttempts;
    }

    @Override
    public long nextDelayMillis(Exception e) {
      if (retries >= maxRetryAttempts) {
        return -1;
      }
      RetryInfo retryInfo = getRetryInfo(e);
      retries++;
      return Durations.toMillis(retryInfo.getRetryDelay());
    }

    RetryInfo getRetryInfo(Exception e) {
      RetryInfo retryInfo = RetryInfo.getDefaultInstance();
      Status status = StatusProto.fromThrowable(e);
      if (status != null) {
        for (Any detail : status.getDetailsList()) {
          if (detail.is(RetryInfo.class)) {
            try {
              retryInfo = detail.unpack(RetryInfo.class);
            } catch (InvalidProtocolBufferException protoEx) {
              // really shouldn't happen, ignore
            }
          }
        }
      }
      return retryInfo;
    }

    @Override
    public int getRetryAttempts() {
      return retries;
    }
  }

  ExecuteRetrier(
      int maxRetryAttempts,
      ListeningScheduledExecutorService retryService,
      CircuitBreaker circuitBreaker) {
    super(
        () -> maxRetryAttempts > 0 ? new RetryInfoBackoff(maxRetryAttempts) : RETRIES_DISABLED,
        ExecuteRetrier::shouldRetry,
        retryService,
        circuitBreaker);
  }

  private static boolean shouldRetry(Exception e) {
    if (BulkTransferException.allCausedByCacheNotFoundException(e)) {
      return true;
    }
    Status status = StatusProto.fromThrowable(e);
    if (status == null || status.getDetailsCount() == 0) {
      return false;
    }
    boolean failedPrecondition = status.getCode() == Code.FAILED_PRECONDITION.value();
    for (Any detail : status.getDetailsList()) {
      if (detail.is(RetryInfo.class)) {
        // server says we can retry, regardless of other details
        return true;
      } else if (failedPrecondition) {
        if (detail.is(PreconditionFailure.class)) {
          try {
            PreconditionFailure f = detail.unpack(PreconditionFailure.class);
            if (f.getViolationsCount() == 0) {
              failedPrecondition = false;
            }
            for (Violation v : f.getViolationsList()) {
              if (!v.getType().equals(VIOLATION_TYPE_MISSING)) {
                failedPrecondition = false;
              }
            }
            // if *all* > 0 precondition failure violations have type MISSING, failedPrecondition
            // remains true
          } catch (InvalidProtocolBufferException protoEx) {
            // really shouldn't happen
            return false;
          }
        } else if (!(detail.is(DebugInfo.class)
            || detail.is(Help.class)
            || detail.is(LocalizedMessage.class)
            || detail.is(RequestInfo.class)
            || detail.is(ResourceInfo.class))) { // ignore benign details
          // consider all other details as failures
          failedPrecondition = false;
        }
      }
    }
    return failedPrecondition;
  }
}
