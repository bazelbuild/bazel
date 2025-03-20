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
package com.google.devtools.build.lib.exec;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.ActionInputPrefetcher;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.LinkedHashSet;
import java.util.Set;

/**
 * Builder class to create an {@link com.google.devtools.build.lib.actions.Executor} instance. This
 * class is part of the module API, which allows modules to affect how the executor is initialized.
 */
public class ExecutorBuilder {
  private final Set<ExecutorLifecycleListener> executorLifecycleListeners = new LinkedHashSet<>();
  private ActionInputPrefetcher prefetcher;

  /** Returns all executor lifecycle listeners registered with this builder so far. */
  public ImmutableSet<ExecutorLifecycleListener> getExecutorLifecycleListeners() {
    return ImmutableSet.copyOf(executorLifecycleListeners);
  }

  public ActionInputPrefetcher getActionInputPrefetcher() {
    return prefetcher == null ? ActionInputPrefetcher.NONE : prefetcher;
  }

  /**
   * Sets the action input prefetcher. Only one module may set the prefetcher. If multiple modules
   * set it, this method will throw an {@link IllegalStateException}.
   */
  @CanIgnoreReturnValue
  public ExecutorBuilder setActionInputPrefetcher(ActionInputPrefetcher prefetcher) {
    Preconditions.checkState(this.prefetcher == null);
    this.prefetcher = Preconditions.checkNotNull(prefetcher);
    return this;
  }

  /**
   * Registers an executor lifecycle listener which will receive notifications throughout the
   * execution phase (if one occurs).
   *
   * @see ExecutorLifecycleListener for events that can be listened to
   */
  @CanIgnoreReturnValue
  public ExecutorBuilder addExecutorLifecycleListener(ExecutorLifecycleListener listener) {
    executorLifecycleListeners.add(listener);
    return this;
  }
}
