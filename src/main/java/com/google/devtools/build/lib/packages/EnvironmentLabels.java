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
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import java.util.Collection;
import java.util.Collections;
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
 * anything with this class.
 */
@AutoCodec
public class EnvironmentLabels {
  final Label label;
  final ImmutableSet<Label> environments;
  final ImmutableSet<Label> defaults;
  /**
   * Maps a member environment to the set of environments that directly fulfill it. Note that we
   * can't set this map until all Target instances for member environments have been initialized,
   * which occurs after group instantiation (this makes the class mutable).
   */
  private Map<Label, NestedSet<Label>> fulfillersMap;

  EnvironmentLabels(Label label, Collection<Label> environments, Collection<Label> defaults) {
    this(label, environments, defaults, null);
  }

  /**
   * Only for use by serialization: the mutable fulfillersMap object is not properly initialized
   * otherwise during deserialization.
   */
  @AutoCodec.VisibleForSerialization
  @AutoCodec.Instantiator
  EnvironmentLabels(
      Label label,
      Collection<Label> environments,
      Collection<Label> defaults,
      Map<Label, NestedSet<Label>> fulfillersMap) {
    this.label = label;
    this.environments = ImmutableSet.copyOf(environments);
    this.defaults = ImmutableSet.copyOf(defaults);
    this.fulfillersMap = fulfillersMap;
  }

  void assertNotInitialized() {
    Preconditions.checkState(fulfillersMap == null, this);
  }

  void checkInitialized() {
    Preconditions.checkNotNull(fulfillersMap, this);
  }

  void setFulfillersMap(Map<Label, NestedSet<Label>> fulfillersMap) {
    Preconditions.checkState(this.fulfillersMap == null, this);
    this.fulfillersMap = Collections.unmodifiableMap(fulfillersMap);
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
  public NestedSet<Label> getFulfillers(Label environment) {
    checkInitialized();
    NestedSet<Label> ans = fulfillersMap.get(environment);
    return ans == null ? NestedSetBuilder.emptySet(Order.STABLE_ORDER) : ans;
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

  /**
   * Compares {@code map1} and {@code map2} using deep equality for their values. Should be feasible
   * because comparison will usually only happen between == objects, so this is hit rarely. If
   * objects are equal, but have been deserialized separately so not ==, this should still be ok
   * because these nested sets are not particularly big, and there are very few EnvironmentGroups
   * (and therefore EnvironmentLabels) in any given build.
   *
   * <p>This will have to be revisited if it turns out to be noticeably expensive. It should be
   * sound to not compare the values of the fulfillerMaps at all, since they are determined from the
   * package each EnvironmentLabel is associated with, and so as long as EnvironmentLabels from
   * different source states but the same package are not compared, the values shouldn't be
   * necessary.
   */
  private static boolean fulfillerMapsEqual(
      Map<Label, NestedSet<Label>> map1, Map<Label, NestedSet<Label>> map2) {
    if (map1 == map2) {
      return true;
    }
    if (map1.size() != map2.size()) {
      return false;
    }
    for (Map.Entry<Label, NestedSet<Label>> entry : map1.entrySet()) {
      NestedSet<Label> secondValue = map2.get(entry.getKey());
      // Do shallowEquals check first for speed.
      if (secondValue == null
          || (!entry.getValue().shallowEquals(secondValue)
              && !entry.getValue().toList().equals(secondValue.toList()))) {
        return false;
      }
    }
    return true;
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
        && fulfillerMapsEqual(this.fulfillersMap, that.fulfillersMap);
  }
}
