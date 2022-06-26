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
package com.google.devtools.build.lib.analysis.config;

import com.google.auto.value.AutoValue;
import com.google.devtools.build.lib.analysis.config.transitions.ConfigurationTransition;
import com.google.devtools.build.lib.analysis.config.transitions.NoTransition;
import com.google.devtools.build.lib.analysis.config.transitions.NullTransition;
import com.google.devtools.build.lib.analysis.config.transitions.SplitTransition;
import com.google.devtools.build.lib.analysis.config.transitions.TransitionFactory;

/** Useful implementations of {@link TransitionFactory}. */
// This class is in lib.analysis.config in order to access HostTransition, which is not visible to
// lib.analysis.config.transitions.
public final class TransitionFactories {
  // Don't instantiate this class.
  private TransitionFactories() {}

  /** Returns a {@link TransitionFactory} that wraps a static transition. */
  public static <T extends TransitionFactory.Data> TransitionFactory<T> of(
      ConfigurationTransition transition) {
    if (transition instanceof HostTransition) {
      return HostTransition.createFactory();
    } else if (transition instanceof NoTransition) {
      return NoTransition.createFactory();
    } else if (transition instanceof NullTransition) {
      return NullTransition.createFactory();
    } else if (transition instanceof SplitTransition) {
      return split((SplitTransition) transition);
    }
    return new AutoValue_TransitionFactories_IdentityFactory<T>(transition);
  }

  /** Returns a {@link TransitionFactory} that wraps a static split transition. */
  public static <T extends TransitionFactory.Data> TransitionFactory<T> split(
      SplitTransition splitTransition) {
    return new AutoValue_TransitionFactories_SplitTransitionFactory<T>(splitTransition);
  }

  /** A {@link TransitionFactory} implementation that wraps a static transition. */
  @AutoValue
  abstract static class IdentityFactory<T extends TransitionFactory.Data>
      implements TransitionFactory<T> {

    abstract ConfigurationTransition transition();

    @Override
    public ConfigurationTransition create(T data) {
      return transition();
    }
  }

  /** A {@link TransitionFactory} implementation that wraps a split transition. */
  @AutoValue
  abstract static class SplitTransitionFactory<T extends TransitionFactory.Data>
      implements TransitionFactory<T> {
    abstract SplitTransition splitTransition();

    @Override
    public SplitTransition create(T data) {
      return splitTransition();
    }

    @Override
    public boolean isSplit() {
      return true;
    }
  }
}
