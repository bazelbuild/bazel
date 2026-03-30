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

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;

import com.google.common.base.Strings;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.ActionInputPrefetcher;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.LinkedHashSet;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * Builder class to create an {@link com.google.devtools.build.lib.actions.Executor} instance. This
 * class is part of the module API, which allows modules to affect how the executor is initialized.
 */
public class ExecutorBuilder {
  private final Set<ExecutorLifecycleListener> executorLifecycleListeners = new LinkedHashSet<>();
  private ActionInputPrefetcher prefetcher;
  @Nullable private String actionExecutionSalt;

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
    checkState(this.prefetcher == null, "prefetcher already set");
    this.prefetcher = checkNotNull(prefetcher, "cannot set prefetcher to null");
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

  /**
   * Returns the action execution salt previously set by {@link #setActionExecutionSalt}, or the
   * empty string if it was never set.
   */
  public String getActionExecutionSalt() {
    return Strings.nullToEmpty(actionExecutionSalt);
  }

  /**
   * Sets the action execution salt.
   *
   * <p>The salt is an opaque value (typically a digest) used by Skyframe and the persistent action
   * cache to invalidate prior action executions against a different value. It may be suitable for
   * communicating information about the action execution environment that is not already
   * incorporated in the action key.
   *
   * <p>At most one module may set the salt. If no module sets it, it defaults to the empty string.
   * If multiple modules set it, an {@link IllegalStateException} is thrown.
   */
  @CanIgnoreReturnValue
  public ExecutorBuilder setActionExecutionSalt(String actionExecutionSalt) {
    checkState(this.actionExecutionSalt == null, "salt already set");
    this.actionExecutionSalt = checkNotNull(actionExecutionSalt, "cannot set salt to null");
    return this;
  }
}
