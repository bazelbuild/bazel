// Copyright 2023 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis.producers;

import com.google.auto.value.AutoOneOf;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.packages.PackageGroup;
import com.google.devtools.build.lib.skyframe.BuildConfigurationKey;

@AutoOneOf(AttributeConfiguration.Kind.class)
abstract class AttributeConfiguration {
  enum Kind {
    /**
     * This is a visibility dependency.
     *
     * <p>Visibility dependencies have null configuration as they are not configurable. Once the
     * target is known, it should be verified to be a {@link PackageGroup}.
     */
    VISIBILITY,
    /**
     * The configuration is null.
     *
     * <p>This is only applied when the dependency is in the same package as the parent and it is
     * not configurable. The value in this case stores any transition keys.
     */
    NULL_TRANSITION_KEYS,
    /**
     * There is a single configuration.
     *
     * <p>This can be the result of a patch transition or no transition at all.
     */
    UNARY,
    /**
     * There is a split transition.
     *
     * <p>It's possible for there to be only one entry.
     */
    SPLIT
  }

  abstract Kind kind();

  abstract void visibility();

  abstract ImmutableList<String> nullTransitionKeys();

  abstract BuildConfigurationKey unary();

  abstract ImmutableMap<String, BuildConfigurationKey> split();

  public int count() {
    switch (kind()) {
      case VISIBILITY:
      case NULL_TRANSITION_KEYS:
      case UNARY:
        return 1;
      case SPLIT:
        return split().size();
    }
    throw new IllegalStateException("unreachable");
  }

  static AttributeConfiguration ofVisibility() {
    return AutoOneOf_AttributeConfiguration.visibility();
  }

  static AttributeConfiguration ofNullTransitionKeys(ImmutableList<String> transitionKeys) {
    return AutoOneOf_AttributeConfiguration.nullTransitionKeys(transitionKeys);
  }

  static AttributeConfiguration ofUnary(BuildConfigurationKey key) {
    return AutoOneOf_AttributeConfiguration.unary(key);
  }

  static AttributeConfiguration ofSplit(
      ImmutableMap<String, BuildConfigurationKey> configurations) {
    return AutoOneOf_AttributeConfiguration.split(configurations);
  }
}
