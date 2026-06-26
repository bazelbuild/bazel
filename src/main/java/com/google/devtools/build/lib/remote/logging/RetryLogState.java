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

import com.google.devtools.build.lib.remote.logging.RemoteExecutionLog.LogEntry;
import com.google.devtools.build.lib.util.io.MessageOutputStream;
import io.grpc.Context;
import javax.annotation.Nullable;

/**
 * Mutable, per-logical-call state shared between {@code Retrier} and {@link LoggingInterceptor} via
 * {@link io.grpc.Context}, used to emit an explicit terminal log entry when a remote RPC fails after
 * all retry attempts are exhausted.
 *
 * <p>The retrier creates one instance per logical call and attaches it to the gRPC {@link Context}
 * around each attempt. On each attempt {@link LoggingInterceptor} records the just-written {@link
 * LogEntry} (and the sink it was written to) here. If the retrier later exhausts its retries, it
 * reuses the captured entry to emit a synthetic terminal entry with {@link
 * RemoteExecutionLog.RetrySummary} set.
 *
 * <p>This bridges two layers that each hold only half of the information: the interceptor knows the
 * gRPC method and target but not that retries were ultimately exhausted, while the retrier knows
 * about exhaustion but not the gRPC method/target.
 */
public final class RetryLogState {

  /** Context key used to pass the per-logical-call state from the retrier to the interceptor. */
  public static final Context.Key<RetryLogState> KEY = Context.key("remote-grpc-retry-log-state");

  @Nullable private volatile LogEntry lastEntry;
  @Nullable private volatile MessageOutputStream<LogEntry> sink;

  /** Records the most recent attempt's log entry and the sink it was written to. */
  public void recordAttempt(LogEntry entry, MessageOutputStream<LogEntry> sink) {
    this.lastEntry = entry;
    this.sink = sink;
  }

  /** Returns the log entry of the most recent attempt, or {@code null} if none was logged. */
  @Nullable
  public LogEntry getLastEntry() {
    return lastEntry;
  }

  /** Returns the sink the attempt entries were written to, or {@code null} if none was logged. */
  @Nullable
  public MessageOutputStream<LogEntry> getSink() {
    return sink;
  }
}
