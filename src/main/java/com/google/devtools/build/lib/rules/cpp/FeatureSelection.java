// Copyright 2015 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.cpp;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableMultimap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.ActionConfig;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.CollidingProvidesException;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.CrosstoolSelectable;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.Feature;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Queue;
import java.util.Set;

/**
 * Implements the feature selection algorithm.
 *
 * <p>Feature selection is done by first enabling all features reachable by an 'implies' edge, and
 * then iteratively pruning features that have unmet requirements.
 */
class FeatureSelection {

  /**
   * The selectables Bazel would like to enable; either because they are supported and generally
   * useful, or because the user required them (for example through the command line).
   */
  private final ImmutableSet<CrosstoolSelectable> requestedSelectables;

  /** All features requested by the user regardless of whether they exist in CROSSTOOLs or not. */
  private final ImmutableSet<String> requestedFeatures;

  /**
   * The currently enabled selectable; during feature selection, we first put all selectables
   * reachable via an 'implies' edge into the enabled selectable set, and than prune that set from
   * selectables that have unmet requirements.
   */
  private final Set<CrosstoolSelectable> enabled = new HashSet<>();
  /**
   * All features and action configs in the order in which they were specified in the configuration.
   *
   * <p>We guarantee the command line to be in the order in which the flags were specified in the
   * configuration.
   */
  private final ImmutableList<CrosstoolSelectable> selectables;

  /**
   * Maps an action's name to the ActionConfig.
   */
  private final ImmutableMap<String, ActionConfig> actionConfigsByActionName;

  /**
   * Maps from a selectable to a set of all the selectables it has a direct 'implies' edge to.
   */
  private final ImmutableMultimap<CrosstoolSelectable, CrosstoolSelectable> implies;

  /**
   * Maps from a selectable to all features that have an direct 'implies' edge to this
   * selectable.
   */
  private final ImmutableMultimap<CrosstoolSelectable, CrosstoolSelectable> impliedBy;

  /**
   * Maps from a selectable to a set of selecatable sets, where:
   * <ul>
   * <li>a selectable set satisfies the 'requires' condition, if all selectables in the
   *        selectable set are enabled</li>
   * <li>the 'requires' condition is satisfied, if at least one of the selectable sets satisfies
   *        the 'requires' condition.</li>
   * </ul>
   */
  private final ImmutableMultimap<CrosstoolSelectable, ImmutableSet<CrosstoolSelectable>>
      requires;

  /**
   * Maps from a string to the set of selectables that 'provide' it.
   */
  private final ImmutableMultimap<String, CrosstoolSelectable> provides;

  /**
   * Maps from a selectable to all selectables that have a requirement referencing it.
   *
   * <p>This will be used to determine which selectables need to be re-checked after a selectable
   * was disabled.
   */
  private final ImmutableMultimap<CrosstoolSelectable, CrosstoolSelectable> requiredBy;

  /** Location of the cc_toolchain in use. */
  private final PathFragment ccToolchainPath;

  FeatureSelection(
      ImmutableSet<String> requestedFeatures,
      ImmutableMap<String, CrosstoolSelectable> selectablesByName,
      ImmutableList<CrosstoolSelectable> selectables,
      ImmutableMultimap<String, CrosstoolSelectable> provides,
      ImmutableMultimap<CrosstoolSelectable, CrosstoolSelectable> implies,
      ImmutableMultimap<CrosstoolSelectable, CrosstoolSelectable> impliedBy,
      ImmutableMultimap<CrosstoolSelectable, ImmutableSet<CrosstoolSelectable>> requires,
      ImmutableMultimap<CrosstoolSelectable, CrosstoolSelectable> requiredBy,
      ImmutableMap<String, ActionConfig> actionConfigsByActionName,
      PathFragment ccToolchainPath) {
    this.requestedFeatures = requestedFeatures;
    ImmutableSet.Builder<CrosstoolSelectable> builder = ImmutableSet.builder();
    for (String name : requestedFeatures) {
      if (selectablesByName.containsKey(name)) {
        builder.add(selectablesByName.get(name));
      }
    }
    this.requestedSelectables = builder.build();
    this.selectables = selectables;
    this.provides = provides;
    this.implies = implies;
    this.impliedBy = impliedBy;
    this.requires = requires;
    this.requiredBy = requiredBy;
    this.actionConfigsByActionName = actionConfigsByActionName;
    this.ccToolchainPath = ccToolchainPath;
  }

  /**
   * @return a {@link FeatureConfiguration} that reflects the set of activated features and action
   *     configs.
   */
  FeatureConfiguration run() throws CollidingProvidesException {
    for (CrosstoolSelectable selectable : requestedSelectables) {
      enableAllImpliedBy(selectable);
    }

    disableUnsupportedActivatables();
    ImmutableList.Builder<CrosstoolSelectable> enabledActivatablesInOrderBuilder =
        ImmutableList.builder();
    for (CrosstoolSelectable selectable : selectables) {
      if (enabled.contains(selectable)) {
        enabledActivatablesInOrderBuilder.add(selectable);
      }
    }

    ImmutableList<CrosstoolSelectable> enabledActivatablesInOrder =
        enabledActivatablesInOrderBuilder.build();
    ImmutableList<Feature> enabledFeaturesInOrder =
        enabledActivatablesInOrder
            .stream()
            .filter(a -> a instanceof Feature)
            .map(f -> (Feature) f)
            .collect(ImmutableList.toImmutableList());
    Iterable<ActionConfig> enabledActionConfigsInOrder =
        Iterables.filter(enabledActivatablesInOrder, ActionConfig.class);

    for (String provided : provides.keys()) {
      List<String> conflicts = new ArrayList<>();
      for (CrosstoolSelectable selectableProvidingString : provides.get(provided)) {
        if (enabledActivatablesInOrder.contains(selectableProvidingString)) {
          conflicts.add(selectableProvidingString.getName());
        }
      }

      if (conflicts.size() > 1) {
        throw new CollidingProvidesException(
            String.format(
                CcToolchainFeatures.COLLIDING_PROVIDES_ERROR,
                provided,
                Joiner.on(" ").join(conflicts)));
      }
    }

    ImmutableSet.Builder<String> enabledActionConfigNames = ImmutableSet.builder();
    for (ActionConfig actionConfig : enabledActionConfigsInOrder) {
      enabledActionConfigNames.add(actionConfig.getActionName());
    }

    return new FeatureConfiguration(
        requestedFeatures,
        enabledFeaturesInOrder,
        enabledActionConfigNames.build(),
        actionConfigsByActionName,
        ccToolchainPath);
  }

  /**
   * Transitively and unconditionally enable all selectables implied by the given selectable and the
   * selectable itself to the enabled selectable set.
   */
  private void enableAllImpliedBy(CrosstoolSelectable selectable) {
    if (enabled.contains(selectable)) {
      return;
    }
    enabled.add(selectable);
    for (CrosstoolSelectable implied : implies.get(selectable)) {
      enableAllImpliedBy(implied);
    }
  }

  /** Remove all unsupported features from the enabled feature set. */
  private void disableUnsupportedActivatables() {
    Queue<CrosstoolSelectable> check = new ArrayDeque<>(enabled);
    while (!check.isEmpty()) {
      checkActivatable(check.poll());
    }
  }

  /**
   * Check if the given selectable is still satisfied within the set of currently enabled
   * selectables.
   *
   * <p>If it is not, remove the selectable from the set of enabled selectables, and re-check all
   * selectables that may now also become disabled.
   */
  private void checkActivatable(CrosstoolSelectable selectable) {
    if (!enabled.contains(selectable) || isSatisfied(selectable)) {
      return;
    }
    enabled.remove(selectable);

    // Once we disable a selectable, we have to re-check all selectables that can be affected
    // by that removal.
    // 1. A selectable that implied the current selectable is now going to be disabled.
    for (CrosstoolSelectable impliesCurrent : impliedBy.get(selectable)) {
      checkActivatable(impliesCurrent);
    }
    // 2. A selectable that required the current selectable may now be disabled, depending on
    // whether the requirement was optional.
    for (CrosstoolSelectable requiresCurrent : requiredBy.get(selectable)) {
      checkActivatable(requiresCurrent);
    }
    // 3. A selectable that this selectable implied may now be disabled if no other selectables
    // also implies it.
    for (CrosstoolSelectable implied : implies.get(selectable)) {
      checkActivatable(implied);
    }
  }

  /**
   * @return whether all requirements of the selectable are met in the set of currently enabled
   *     selectables.
   */
  private boolean isSatisfied(CrosstoolSelectable selectable) {
    return (requestedSelectables.contains(selectable) || isImpliedByEnabledActivatable(selectable))
        && allImplicationsEnabled(selectable)
        && allRequirementsMet(selectable);
  }

  /** @return whether a currently enabled selectable implies the given selectable. */
  private boolean isImpliedByEnabledActivatable(CrosstoolSelectable selectable) {
    return !Collections.disjoint(impliedBy.get(selectable), enabled);
  }

  /** @return whether all implications of the given feature are enabled. */
  private boolean allImplicationsEnabled(CrosstoolSelectable selectable) {
    for (CrosstoolSelectable implied : implies.get(selectable)) {
      if (!enabled.contains(implied)) {
        return false;
      }
    }
    return true;
  }

  /**
   * @return whether all requirements are enabled.
   *     <p>This implies that for any of the selectable sets all of the specified selectable are
   *     enabled.
   */
  private boolean allRequirementsMet(CrosstoolSelectable feature) {
    if (!requires.containsKey(feature)) {
      return true;
    }
    for (ImmutableSet<CrosstoolSelectable> requiresAllOf : requires.get(feature)) {
      boolean requirementMet = true;
      for (CrosstoolSelectable required : requiresAllOf) {
        if (!enabled.contains(required)) {
          requirementMet = false;
          break;
        }
      }
      if (requirementMet) {
        return true;
      }
    }
    return false;
  }
}
