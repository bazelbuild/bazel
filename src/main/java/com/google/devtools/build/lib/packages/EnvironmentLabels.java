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

package com.google.devtools.build.lib.packages;

import com.google.common.base.Verify;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

/**
 * Parts of an {@link EnvironmentGroup} that are needed for analysis. Since {@link EnvironmentGroup}
 * keeps a reference to a {@link Package} object, it is too heavyweight to store in analysis.
 */
@AutoCodec
public class EnvironmentLabels {
  final Label label;
  final ImmutableSet<Label> environments;
  final ImmutableSet<Label> defaults;
  /**
   * Maps a member environment to the set of environments that directly fulfill it. Note that we
   * can't populate this map until all Target instances for member environments have been
   * initialized, which may occur after group instantiation (this makes the class mutable).
   */
  final Map<Label, NestedSet<Label>> fulfillersMap = new HashMap<>();

  EnvironmentLabels(Label label, Collection<Label> environments, Collection<Label> defaults) {
    this.label = label;
    this.environments = ImmutableSet.copyOf(environments);
    this.defaults = ImmutableSet.copyOf(defaults);
  }

  public Set<Label> getEnvironments() {
    return environments;
  }

  public Set<Label> getDefaults() {
    return defaults;
  }

  /**
   * Determines whether or not an environment is a default. Returns false if the environment doesn't
   * belong to this group.
   */
  public boolean isDefault(Label environment) {
    return defaults.contains(environment);
  }

  /**
   * Returns the set of environments that transitively fulfill the specified environment. The
   * environment must be a valid member of this group.
   *
   * <p>>For example, if the input is <code>":foo"</code> and <code>":bar"</code> fulfills <code>
   * ":foo"</code> and <code>":baz"</code> fulfills <code>":bar"</code>, this returns <code>
   * [":foo", ":bar", ":baz"]</code>.
   *
   * <p>If no environments fulfill the input, returns an empty set.
   */
  public Iterable<Label> getFulfillers(Label environment) {
    return Verify.verifyNotNull(fulfillersMap.get(environment));
  }

  public Label getLabel() {
    return label;
  }
}
