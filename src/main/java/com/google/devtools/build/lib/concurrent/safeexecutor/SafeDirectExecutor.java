// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.concurrent.safeexecutor;

import static com.google.common.util.concurrent.MoreExecutors.directExecutor;

import com.google.common.base.Preconditions;
import com.google.common.util.concurrent.FutureCallback;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import java.util.concurrent.Executor;

/**
 * Direct execution implementation of {@link SafeExecutor}.
 *
 * <p>This exists for classes that reference an injected executor that uses inline callback
 * execution. Never triggers {@link RejectedExecutionException}.
 */
final class SafeDirectExecutor implements SafeExecutor {
  public static final SafeDirectExecutor INSTANCE = new SafeDirectExecutor();

  private SafeDirectExecutor() {}

  @Override
  public Executor getInternalUnsafeExecutor() {
    return directExecutor();
  }

  @Override
  public void execute(RejectionHandlingRunnable task) {
    task.run();
  }

  @Override
  public <T> void addCallback(ListenableFuture<T> future, FutureCallback<? super T> callback) {
    Preconditions.checkNotNull(future, "future");
    Preconditions.checkNotNull(callback, "callback");
    Futures.addCallback(future, callback, directExecutor());
  }
}
