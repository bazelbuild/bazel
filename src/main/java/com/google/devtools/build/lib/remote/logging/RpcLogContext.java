// Copyright 2024 The Bazel Authors. All rights reserved.
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

import io.grpc.Context;

/**
 * Immutable per-attempt identity propagated one-way from the retrier to {@link LoggingInterceptor}
 * via the gRPC {@link Context}, so that interleaved attempt entries can be correlated into a single
 * logical RPC.
 *
 * <p>The retrier creates one instance per attempt — the {@code rpcId} is constant across a logical
 * call while {@code attemptNumber} increases — attaches it to the {@link Context} around each
 * attempt, and {@link LoggingInterceptor} reads it and stamps {@code rpc_id}/{@code attempt_number}
 * onto each {@link RemoteExecutionLog.LogEntry}. The flow is strictly one-way: no logging data is
 * passed back to the retrier.
 */
public final class RpcLogContext {

  /** Context key used to pass the per-attempt identity from the retrier to the interceptor. */
  public static final Context.Key<RpcLogContext> KEY = Context.key("remote-grpc-rpc-log-context");

  private final String rpcId;
  private final int attemptNumber;

  public RpcLogContext(String rpcId, int attemptNumber) {
    this.rpcId = rpcId;
    this.attemptNumber = attemptNumber;
  }

  /** Returns the id shared by all attempts of this logical call. */
  public String getRpcId() {
    return rpcId;
  }

  /** Returns the 1-based index of this attempt within the logical call. */
  public int getAttemptNumber() {
    return attemptNumber;
  }
}
