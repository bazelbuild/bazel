// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skyframe.serialization;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;

/** Stateful class for providing additional context to a single serialization "session". */
// TODO(bazel-team): This class is just a shell, fill in.
public class SerializationContext {

  /**
   * This is a stub for context where it is less straightforward to thread from the top-level
   * invocation.
   *
   * <p>This is a bug waiting to happen because it is very easy to accidentally modify a codec to
   * use this context which won't contain any of the expected state.
   */
  // TODO(bazel-team): delete this and all references to it.
  public static final SerializationContext UNTHREADED_PLEASE_FIX =
      new SerializationContext(ImmutableMap.of());

  private final ImmutableMap<Class<?>, Object> dependencies;

  public SerializationContext(ImmutableMap<Class<?>, Object> dependencies) {
    this.dependencies = dependencies;
  }

  @SuppressWarnings("unchecked")
  public <T> T getDependency(Class<T> type) {
    Preconditions.checkNotNull(type);
    return (T) dependencies.get(type);
  }
}
