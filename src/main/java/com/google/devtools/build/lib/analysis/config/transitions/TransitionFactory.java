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

/**
 * Factory interface for transitions that are created dynamically, instead of being created as
 * singletons.
 *
 * <p>This class allows for cases where the general <i>type</i> of a transition is known, but the
 * specifics of the transition itself cannot be determined until the target is configured. Examples
 * of this are transitions that depend on other (non-configured) attributes from the same target, or
 * transitions that depend on state determined during configuration, such as the execution platform
 * or resolved toolchains.
 *
 * <p>Implementations must override {@link Object#equals} and {@link Object#hashCode} unless
 * exclusively accessed as singletons.
 *
 * @param <T> the type of data object passed to the {@link #create} method, used to create the
 *     actual {@link ConfigurationTransition} instance
 */
public interface TransitionFactory<T extends TransitionFactory.Data> {

  /** Enum that describes what type of transition a TransitionFactory creates. */
  enum TransitionType {
    /** A transition that can be used for rules or attributes. */
    ANY,
    /** A transition that can be used for rules only. */
    RULE,
    /** A transition that can be used for attributes only. */
    ATTRIBUTE;

    public boolean isCompatibleWith(TransitionType other) {
      if (this == ANY) {
        return true;
      }
      if (other == ANY) {
        return true;
      }
      return this == other;
    }
  }

  /** A marker interface for classes that provide data to TransitionFactory instances. */
  interface Data {}

  /** Returns a new {@link ConfigurationTransition}, based on the given data. */
  ConfigurationTransition create(T data);

  /**
   * Returns a {@link TransitionType} to clarify what data (if any) the factory requires to create a
   * transation.
   */
  TransitionType transitionType();

  // TODO(https://github.com/bazelbuild/bazel/issues/7814): Once everything uses TransitionFactory,
  // remove these methods.

  /**
   * Returns {@code true} if the result of this {@link TransitionFactory} should be considered as
   * part of the tooling rather than a dependency of the original target.
   */
  default boolean isTool() {
    return false;
  }

  /** Returns {@code true} if the result of this {@link TransitionFactory} is a split transition. */
  default boolean isSplit() {
    return false;
  }

  /** Visit this trnsition factory with the given visitor. */
  default void visit(Visitor<T> visitor) {
    visitor.visit(this);
  }

  /** Interface used to progressively visit transitions. */
  interface Visitor<T extends TransitionFactory.Data> {
    void visit(TransitionFactory<T> factory);
  }
}
