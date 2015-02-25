// Copyright 2015 Google Inc. All rights reserved.
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

import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.syntax.Label;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Model for the "environment_group' rule: the piece of Bazel's rule constraint system that binds
 * thematically related environments together and determines which environments a rule supports
 * by default. See {@link com.google.devtools.build.lib.analysis.constraints.ConstraintSemantics}
 * for precise semantic details of how this information is used.
 *
 * <p>Note that "environment_group" is implemented as a loading-time function, not a rule. This is
 * to support proper discovery of defaults: Say rule A has no explicit constraints and depends
 * on rule B, which is explicitly constrained to environment ":bar". Since A declares nothing
 * explicitly, it's implicitly constrained to DEFAULTS (whatever that is). Therefore, the
 * dependency is only allowed if DEFAULTS doesn't include environments beyond ":bar". To figure
 * that out, we need to be able to look up the environment group for ":bar", which is what this
 * class provides.
 *
 * <p>If we implemented this as a rule, we'd have to provide that lookup via rule dependencies,
 * e.g. something like:
 *
 * <code>
 *   environment(
 *       name = 'bar',
 *       group = [':sample_environments'],
 *       is_default = 1
 *   )
 * </code>
 *
 * <p>But this won't work. This would let us find the environment group for ":bar", but the only way
 * to determine what other environments belong to the group is to have the group somehow reference
 * them. That would produce circular dependencies in the build graph, which is no good.
 */
@Immutable
public class EnvironmentGroup implements Target {
  private final Label label;
  private final Location location;
  private final Package containingPackage;
  private final Set<Label> environments;
  private final Set<Label> defaults;

  /**
   * Predicate that matches labels from a different package than the initialized package.
   */
  private static final class DifferentPackage implements Predicate<Label> {
    private final Package containingPackage;

    private DifferentPackage(Package containingPackage) {
      this.containingPackage = containingPackage;
    }

    @Override
    public boolean apply(Label environment) {
      return !environment.getPackageName().equals(containingPackage.getName());
    }
  }

  /**
   * Instantiates a new group without verifying the soundness of its contents. See the validation
   * methods below for appropriate checks.
   *
   * @param label the build label identifying this group
   * @param pkg the package this group belongs to
   * @param environments the set of environments that belong to this group
   * @param defaults the environments a rule implicitly supports unless otherwise specified
   * @param location location in the BUILD file of this group
   */
  EnvironmentGroup(Label label, Package pkg, final List<Label> environments, List<Label> defaults,
      Location location) {
    this.label = label;
    this.location = location;
    this.containingPackage = pkg;
    this.environments = ImmutableSet.copyOf(environments);
    this.defaults = ImmutableSet.copyOf(defaults);
  }

  /**
   * Checks that all environments declared by this group are in the same package as the group (so
   * we can perform an environment --> environment_group lookup and know the package is available)
   * and checks that all defaults are legitimate members of the group.
   *
   * <p>Does <b>not</b> check that the referenced environments exist (see
   * {@link #checkEnvironmentsExist).
   *
   * @return a list of validation errors that occurred
   */
  List<Event> validateMembership() {
    List<Event> events = new ArrayList<>();

    // All environments should belong to the same package as this group.
    for (Label environment :
        Iterables.filter(environments, new DifferentPackage(containingPackage))) {
      events.add(Event.error(location,
          environment + " is not in the same package as group " + label));
    }

    // The defaults must be a subset of the member environments.
    for (Label unknownDefault : Sets.difference(defaults, environments)) {
      events.add(Event.error(location, "default " + unknownDefault + " is not a "
          + "declared environment for group " + getLabel()));
    }

    return events;
  }

  /**
   * Given the set of targets in this group's package, checks that all of the group's declared
   * environments are part of that set (i.e. the group doesn't reference non-existant labels).
   *
   * @param pkgTargets mapping from label name to target instance for this group's package
   * @return a list of validation errors that occurred
   */
  List<Event> checkEnvironmentsExist(Map<String, Target> pkgTargets) {
    List<Event> events = new ArrayList<>();
    for (Label envName : environments) {
      Target env =  pkgTargets.get(envName.getName());
      if (env == null) {
        events.add(Event.error(location, "environment " + envName + " does not exist"));
      } else if (!env.getTargetKind().equals("environment rule")) {
        events.add(Event.error(location, env.getLabel() + " is not a valid environment"));
      }
    }
    return events;
  }

  /**
   * Returns the environments that belong to this group.
   */
  public Set<Label> getEnvironments() {
    return environments;
  }

  /**
   * Returns the environments a rule supports by default, i.e. if it has no explicit references to
   * environments in this group.
   */
  public Set<Label> getDefaults() {
    return defaults;
  }

  /**
   * Determines whether or not an environment is a default. Returns false if the environment
   * doesn't belong to this group.
   */
  public boolean isDefault(Label environment) {
    return defaults.contains(environment);
  }

  @Override
  public Label getLabel() {
    return label;
  }

  @Override
  public String getName() {
    return label.getName();
  }

  @Override
  public Package getPackage() {
    return containingPackage;
  }

  @Override
  public String getTargetKind() {
    return targetKind();
  }

  @Override
  public Rule getAssociatedRule() {
    return null;
  }

  @Override
  public License getLicense() {
    return License.NO_LICENSE;
  }

  @Override
  public Location getLocation() {
    return location;
  }

  @Override
  public String toString() {
   return targetKind() + " " + getLabel();
  }

  @Override
  public Set<License.DistributionType> getDistributions() {
    return Collections.emptySet();
  }

  @Override
  public RuleVisibility getVisibility() {
    return ConstantRuleVisibility.PRIVATE; // No rule should be referencing an environment_group.
  }

  public static String targetKind() {
    return "environment group";
  }

  @Override
  public boolean equals(Object o) {
    // In a distributed implementation these may not be the same object.
    if (o == this) {
      return true;
    } else if (!(o instanceof EnvironmentGroup)) {
      return false;
    } else {
      return ((EnvironmentGroup) o).getLabel().equals(getLabel());
    }
  }
}
