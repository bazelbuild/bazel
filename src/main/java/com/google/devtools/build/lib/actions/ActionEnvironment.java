// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.actions;

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Interner;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.util.Fingerprint;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;

/**
 * Environment variables for build or test actions.
 *
 * <p>The action environment consists of three parts.
 *
 * <ol>
 *   <li>All the environment variables with a fixed value, stored in a map.
 *   <li>All the environment variables inherited from the client environment, stored in a set.
 *   <li>All the environment variables that are explicitly unset, stored in a set. These are
 *       removed from the environment even if an earlier layer of the environment (e.g. the default
 *       test environment) set them.
 * </ol>
 *
 * <p>Inherited environment variables must be declared in the Action interface (see {@link
 * Action#getClientEnvironmentVariables}), so that the dependency on the client environment is known
 * to the execution framework for correct incremental builds.
 *
 * <p>By splitting the environment, we can handle environment variable changes more efficiently -
 * the dependency of the action on the environment variable are tracked in Skyframe (and in the
 * action cache), such that Bazel knows exactly which actions it needs to rerun, and does not have
 * to reanalyze the entire dependency graph.
 */
public abstract class ActionEnvironment {

  public static final ActionEnvironment EMPTY = new EmptyActionEnvironment();

  private static final Interner<ActionEnvironment> actionEnvironmentInterner =
      BlazeInterners.newWeakInterner();

  /** Convenience method for creating an {@link ActionEnvironment} with no inherited variables. */
  public static ActionEnvironment create(ImmutableMap<String, String> fixedEnv) {
    return create(fixedEnv, /* inheritedEnv= */ ImmutableSet.of());
  }

  /**
   * Creates a new {@link ActionEnvironment} with no unset variables.
   *
   * <p>If an environment variable is contained both as a key in {@code fixedEnv} and in {@code
   * inheritedEnv}, the result of {@link #resolve} will contain the value inherited from the client
   * environment.
   */
  public static ActionEnvironment create(
      ImmutableMap<String, String> fixedEnv, ImmutableSet<String> inheritedEnv) {
    return create(fixedEnv, inheritedEnv, /* unsetEnv= */ ImmutableSet.of());
  }

  /**
   * Creates a new {@link ActionEnvironment}.
   *
   * <p>If an environment variable is contained both as a key in {@code fixedEnv} and in {@code
   * inheritedEnv}, the result of {@link #resolve} will contain the value inherited from the client
   * environment.
   *
   * <p>Variables in {@code unsetEnv} are removed from the map passed to {@link #resolve}, even if
   * they were present in the map before the call or are contained in {@code fixedEnv} or {@code
   * inheritedEnv}. This allows an environment to remove variables that were set by an earlier
   * layer of the environment, e.g. the default test environment.
   */
  public static ActionEnvironment create(
      ImmutableMap<String, String> fixedEnv,
      ImmutableSet<String> inheritedEnv,
      ImmutableSet<String> unsetEnv) {
    if (fixedEnv.isEmpty() && inheritedEnv.isEmpty() && unsetEnv.isEmpty()) {
      return EMPTY;
    }
    return actionEnvironmentInterner.intern(
        new SimpleActionEnvironment(fixedEnv, inheritedEnv, unsetEnv));
  }

  /**
   * Splits the given map into a map of variables with a fixed value, and a set of variables that
   * should be inherited, the latter of which are identified by having a {@code null} value in the
   * given map. Returns these two parts as a new {@link ActionEnvironment} instance.
   */
  public static ActionEnvironment split(Map<String, String> env) {
    return split(env, /* unsetEnv= */ ImmutableSet.of());
  }

  /**
   * Splits the given map into a map of variables with a fixed value, and a set of variables that
   * should be inherited, the latter of which are identified by having a {@code null} value in the
   * given map. Returns these two parts together with the given set of variables to unset as a new
   * {@link ActionEnvironment} instance.
   */
  public static ActionEnvironment split(Map<String, String> env, Set<String> unsetEnv) {
    Map<String, String> fixedEnv = new TreeMap<>();
    Set<String> inheritedEnv = new TreeSet<>();
    for (Map.Entry<String, String> entry : env.entrySet()) {
      if (entry.getValue() != null) {
        fixedEnv.put(entry.getKey(), entry.getValue());
      } else {
        inheritedEnv.add(entry.getKey());
      }
    }
    return create(
        ImmutableMap.copyOf(fixedEnv),
        ImmutableSet.copyOf(inheritedEnv),
        ImmutableSet.copyOf(new TreeSet<>(unsetEnv)));
  }

  private ActionEnvironment() {}

  /**
   * Returns the 'fixed' part of the environment, i.e., those environment variables that are set to
   * fixed values and their values. This should only be used for testing and to compute the cache
   * keys of actions. Use {@link #resolve} instead to get the complete environment.
   */
  public abstract ImmutableMap<String, String> getFixedEnv();

  /**
   * Returns the 'inherited' part of the environment, i.e., those environment variables that are
   * inherited from the client environment and therefore have no fixed value here. This should only
   * be used for testing and to compute the cache keys of actions. Use {@link #resolve} instead to
   * get the complete environment.
   */
  public abstract ImmutableSet<String> getInheritedEnv();

  /**
   * Returns the 'unset' part of the environment, i.e., those environment variables that are
   * explicitly removed from the environment by {@link #resolve}. This should only be used for
   * testing and to compute the cache keys of actions. Use {@link #resolve} instead to get the
   * complete environment.
   */
  public abstract ImmutableSet<String> getUnsetEnv();

  /**
   * Returns an upper bound on the combined size of the fixed and inherited environments. A call to
   * {@link #resolve} may add fewer entries than this number if environment variables are contained
   * in both the fixed and the inherited environment.
   */
  public abstract int estimatedSize();

  /**
   * Resolves the action environment and adds the resulting entries to the given {@code result} map,
   * by looking up any inherited env variables in the given {@code clientEnv}. Afterwards, removes
   * all unset env variables from the map.
   *
   * <p>We pass in a map to mutate to avoid creating and merging intermediate maps.
   */
  public final void resolve(Map<String, String> result, Map<String, String> clientEnv) {
    checkNotNull(clientEnv);
    result.putAll(getFixedEnv());
    for (String var : getInheritedEnv()) {
      String value = clientEnv.get(var);
      if (value != null) {
        result.put(var, value);
      }
    }
    ImmutableSet<String> unsetEnv = getUnsetEnv();
    if (!unsetEnv.isEmpty()) {
      result.keySet().removeAll(unsetEnv);
    }
  }

  public final void addTo(Fingerprint f) {
    f.addStringMap(getFixedEnv());
    f.addStrings(getInheritedEnv());
    // Only add the unset variables if there are any so that the fingerprint of the common case of
    // an environment without unset variables remains unchanged.
    ImmutableSet<String> unsetEnv = getUnsetEnv();
    if (!unsetEnv.isEmpty()) {
      f.addStrings(unsetEnv);
    }
  }

  /**
   * Returns a copy of the environment with the given fixed variables added to it, <em>overwriting
   * any existing occurrences of those variables</em>, including any that were previously unset.
   */
  public final ActionEnvironment withAdditionalFixedVariables(Map<String, String> fixedVars) {
    if (fixedVars.isEmpty()) {
      return this;
    }
    if (this == EMPTY) {
      return actionEnvironmentInterner.intern(
          new SimpleActionEnvironment(
              ImmutableMap.copyOf(fixedVars), ImmutableSet.of(), ImmutableSet.of()));
    }
    return actionEnvironmentInterner.intern(
        new CompoundActionEnvironment(this, ImmutableMap.copyOf(fixedVars)));
  }

  private static final class EmptyActionEnvironment extends ActionEnvironment {

    @Override
    public ImmutableMap<String, String> getFixedEnv() {
      return ImmutableMap.of();
    }

    @Override
    public ImmutableSet<String> getInheritedEnv() {
      return ImmutableSet.of();
    }

    @Override
    public ImmutableSet<String> getUnsetEnv() {
      return ImmutableSet.of();
    }

    @Override
    public int estimatedSize() {
      return 0;
    }
  }

  private static final class SimpleActionEnvironment extends ActionEnvironment {
    private final ImmutableMap<String, String> fixedEnv;
    private final ImmutableSet<String> inheritedEnv;
    private final ImmutableSet<String> unsetEnv;

    SimpleActionEnvironment(
        ImmutableMap<String, String> fixedEnv,
        ImmutableSet<String> inheritedEnv,
        ImmutableSet<String> unsetEnv) {
      this.fixedEnv = fixedEnv;
      this.inheritedEnv = inheritedEnv;
      this.unsetEnv = unsetEnv;
    }

    @Override
    public ImmutableMap<String, String> getFixedEnv() {
      return fixedEnv;
    }

    @Override
    public ImmutableSet<String> getInheritedEnv() {
      return inheritedEnv;
    }

    @Override
    public ImmutableSet<String> getUnsetEnv() {
      return unsetEnv;
    }

    @Override
    public int estimatedSize() {
      return fixedEnv.size() + inheritedEnv.size();
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (!(o instanceof SimpleActionEnvironment that)) {
        return false;
      }
      return fixedEnv.equals(that.fixedEnv)
          && inheritedEnv.equals(that.inheritedEnv)
          && unsetEnv.equals(that.unsetEnv);
    }

    @Override
    public int hashCode() {
      return Objects.hash(fixedEnv, inheritedEnv, unsetEnv);
    }
  }

  private static final class CompoundActionEnvironment extends ActionEnvironment {
    private final ActionEnvironment base;
    private final ImmutableMap<String, String> fixedVars;

    private CompoundActionEnvironment(
        ActionEnvironment base, ImmutableMap<String, String> fixedVars) {
      this.base = base;
      this.fixedVars = fixedVars;
    }

    @Override
    public ImmutableMap<String, String> getFixedEnv() {
      return ImmutableMap.<String, String>builder()
          .putAll(base.getFixedEnv())
          .putAll(fixedVars)
          .buildKeepingLast();
    }

    @Override
    public ImmutableSet<String> getInheritedEnv() {
      return base.getInheritedEnv();
    }

    @Override
    public ImmutableSet<String> getUnsetEnv() {
      // The additional fixed variables override any unset variables of the base environment.
      ImmutableSet<String> baseUnsetEnv = base.getUnsetEnv();
      if (baseUnsetEnv.isEmpty()) {
        return baseUnsetEnv;
      }
      return Sets.difference(baseUnsetEnv, fixedVars.keySet()).immutableCopy();
    }

    @Override
    public int estimatedSize() {
      return base.estimatedSize() + fixedVars.size();
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (!(o instanceof CompoundActionEnvironment that)) {
        return false;
      }
      return base.equals(that.base) && fixedVars.equals(that.fixedVars);
    }

    @Override
    public int hashCode() {
      return Objects.hash(base, fixedVars);
    }
  }
}
