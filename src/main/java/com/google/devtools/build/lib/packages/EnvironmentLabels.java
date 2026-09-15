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

import com.google.common.base.MoreObjects;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.collect.ImmutableSortedSet;
import com.google.devtools.build.lib.cmdline.Label;
import java.util.Collection;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Parts of an {@link EnvironmentGroup} that are needed for analysis. Since {@link EnvironmentGroup}
 * keeps a reference to a {@link Package} object, it is too heavyweight to store in analysis.
 *
 * <p>Constructor should only be called by {@link EnvironmentGroup}, and this object must never be
 * accessed externally until after {@link EnvironmentGroup#processMemberEnvironments} is called. The
 * mutability of fulfillersMap means that we must take care to wait until it is set before doing
 * anything with this class.e
 */
public final class EnvironmentLabels {
  final Label label;
  final ImmutableSet<Label> environments;
  final ImmutableSet<Label> defaults;

  /**
   * Maps a member environment to the set of environments that directly fulfill it. Note that we
   * can't set this map until all Target instances for member environments have been initialized,
   * which occurs after group instantiation (this makes the class mutable).
   */
  private Map<Label, ImmutableSortedSet<Label>> fulfillersMap;

  EnvironmentLabels(Label label, Collection<Label> environments, Collection<Label> defaults) {
    this(label, environments, defaults, null);
  }

  /**
   * Only for use by serialization: the mutable fulfillersMap object is not properly initialized
   * otherwise during deserialization.
   */
  private EnvironmentLabels(
      Label label,
      Collection<Label> environments,
      Collection<Label> defaults,
      Map<Label, ImmutableSortedSet<Label>> fulfillersMap) {
    this.label = label;
    this.environments = ImmutableSortedSet.copyOf(environments);
    this.defaults = ImmutableSortedSet.copyOf(defaults);
    this.fulfillersMap = fulfillersMap == null ? null : ImmutableSortedMap.copyOf(fulfillersMap);
  }

  void assertNotInitialized() {
    Preconditions.checkState(fulfillersMap == null, this);
  }

  void checkInitialized() {
    Preconditions.checkNotNull(fulfillersMap, this);
  }

  void setFulfillersMap(Map<Label, ImmutableSortedSet<Label>> fulfillersMap) {
    Preconditions.checkState(this.fulfillersMap == null, this);
    this.fulfillersMap = ImmutableSortedMap.copyOf(fulfillersMap);
  }

  public Set<Label> getEnvironments() {
    checkInitialized();
    return environments;
  }

  public Set<Label> getDefaults() {
    checkInitialized();
    return defaults;
  }

  /**
   * Determines whether or not an environment is a default. Returns false if the environment doesn't
   * belong to this group.
   */
  public boolean isDefault(Label environment) {
    checkInitialized();
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
  public ImmutableSortedSet<Label> getFulfillers(Label environment) {
    checkInitialized();
    ImmutableSortedSet<Label> ans = fulfillersMap.get(environment);
    return ans == null ? ImmutableSortedSet.of() : ans;
  }

  public Label getLabel() {
    checkInitialized();
    return label;
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("label", label)
        .add("sizes", environments.size() + ", " + defaults.size() + ", " + fulfillersMap.size())
        .add("environments", environments)
        .add("defaults", defaults)
        .add("fulfillersMap", fulfillersMap)
        .toString();
  }

  @Override
  public int hashCode() {
    checkInitialized();
    return Objects.hash(label, environments, defaults, fulfillersMap.keySet());
  }

  @Override
  public boolean equals(Object o) {
    checkInitialized();
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }
    EnvironmentLabels that = (EnvironmentLabels) o;
    that.checkInitialized();
    return label.equals(that.label)
        && environments.equals(that.environments)
        && defaults.equals(that.defaults)
        && fulfillersMap.equals(that.fulfillersMap);
  }
}
