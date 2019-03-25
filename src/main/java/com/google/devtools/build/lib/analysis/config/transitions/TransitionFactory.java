// Copyright 2019 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis.config.transitions;

import com.google.devtools.build.lib.analysis.config.transitions.TransitionFactory.TransitionFactoryData;

/**
 * Helper for the types of transitions that are statically declared but must be instantiated for
 * each use.
 */
public interface TransitionFactory<T extends TransitionFactoryData> {

  /** Interface for types of data that a {@link TransitionFactory} can use. */
  interface TransitionFactoryData {}

  /** Returns a new {@link ConfigurationTransition}, based on the given data. */
  ConfigurationTransition create(T data);

  /** Returns {@code true} if the result of this {@link TransitionFactory} is a host transition. */
  default boolean isHost() {
    return false;
  }

  /** Returns {@code true} if the result of this {@link TransitionFactory} is a split transition. */
  default boolean isSplit() {
    return false;
  }

  /** Returns {@code true} if the result of this {@link TransitionFactory} is a final transition. */
  default boolean isFinal() {
    return false;
  }
}
