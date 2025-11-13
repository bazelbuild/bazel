// Copyright 2025 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.profiler;

import com.google.common.util.concurrent.ListenableFuture;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.function.Consumer;

/**
 * A profiler that can be used to profile async operations.
 *
 * <p>This profiler is thread-compatible but not thread-safe. You should create one profiler per
 * task.
 */
public interface AsyncProfiler extends SilentCloseable {

  SilentCloseable profile(ProfilerTask type, String description);

  SilentCloseable profile(String description);

  <T> ListenableFuture<T> profileFuture(ListenableFuture<T> future, String description);

  @CanIgnoreReturnValue
  <T> ListenableFuture<T> profileFuture(
      ListenableFuture<T> future, ProfilerTask type, String description);

  Runnable profileCallback(Runnable runnable, String description);

  Runnable profileCallback(Runnable runnable, ProfilerTask type, String description);

  <T> Consumer<T> profileCallback(Consumer<T> consumer, String description);

  <T> Consumer<T> profileCallback(Consumer<T> consumer, ProfilerTask type, String description);
}
