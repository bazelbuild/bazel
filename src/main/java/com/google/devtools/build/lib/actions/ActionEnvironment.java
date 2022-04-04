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
import com.google.devtools.build.lib.util.Fingerprint;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;

/**
 * Environment variables for build or test actions.
 *
 * <p>The action environment consists of two parts.
 *
 * <ol>
 *   <li>All the environment variables with a fixed value, stored in a map.
 *   <li>All the environment variables inherited from the client environment, stored in a set.
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
public final class ActionEnvironment {

  /**
   * A map of environment variables together with a list of variables to inherit from the shell
   * environment.
   */
  public interface EnvironmentVariables {

    /**
     * Returns the fixed environment variables as a map.
     *
     * <p>WARNING: this allocates additional objects if the underlying implementation is a {@link
     * CompoundEnvironmentVariables}; use sparingly.
     */
    ImmutableMap<String, String> getFixedEnvironment();

    /**
     * Returns the inherited environment variables as a set.
     *
     * <p>WARNING: this allocates additional objects if the underlying implementation is a {@link
     * CompoundEnvironmentVariables}; use sparingly.
     */
    ImmutableSet<String> getInheritedEnvironment();

    default boolean isEmpty() {
      return getFixedEnvironment().isEmpty() && getInheritedEnvironment().isEmpty();
    }

    default int size() {
      return getFixedEnvironment().size() + getInheritedEnvironment().size();
    }
  }

  /**
   * An {@link EnvironmentVariables} that combines variables from two different environments without
   * allocation a new map.
   */
  static class CompoundEnvironmentVariables implements EnvironmentVariables {

    static EnvironmentVariables create(
        Map<String, String> fixedVars, Set<String> inheritedVars, EnvironmentVariables base) {
      if (fixedVars.isEmpty() && inheritedVars.isEmpty() && base.isEmpty()) {
        return EMPTY_ENVIRONMENT_VARIABLES;
      }
      return new CompoundEnvironmentVariables(fixedVars, inheritedVars, base);
    }

    private final EnvironmentVariables current;
    private final EnvironmentVariables base;

    private CompoundEnvironmentVariables(
        Map<String, String> fixedVars, Set<String> inheritedVars, EnvironmentVariables base) {
      this.current = SimpleEnvironmentVariables.create(fixedVars, inheritedVars);
      this.base = base;
    }

    @Override
    public boolean isEmpty() {
      return current.isEmpty() && base.isEmpty();
    }

    @Override
    public ImmutableMap<String, String> getFixedEnvironment() {
      ImmutableMap.Builder<String, String> result = new ImmutableMap.Builder<>();
      result.putAll(base.getFixedEnvironment());
      result.putAll(current.getFixedEnvironment());
      return result.buildKeepingLast();
    }

    @Override
    public ImmutableSet<String> getInheritedEnvironment() {
      ImmutableSet.Builder<String> result = new ImmutableSet.Builder<>();
      result.addAll(base.getInheritedEnvironment());
      result.addAll(current.getInheritedEnvironment());
      return result.build();
    }
  }

  /** A simple {@link EnvironmentVariables}. */
  static class SimpleEnvironmentVariables implements EnvironmentVariables {

    static EnvironmentVariables create(Map<String, String> fixedVars, Set<String> inheritedVars) {
      if (fixedVars.isEmpty() && inheritedVars.isEmpty()) {
        return EMPTY_ENVIRONMENT_VARIABLES;
      }
      return new SimpleEnvironmentVariables(fixedVars, inheritedVars);
    }

    private final ImmutableMap<String, String> fixedVars;
    private final ImmutableSet<String> inheritedVars;

    private SimpleEnvironmentVariables(Map<String, String> fixedVars, Set<String> inheritedVars) {
      this.fixedVars = ImmutableMap.copyOf(fixedVars);
      this.inheritedVars = ImmutableSet.copyOf(inheritedVars);
    }

    @Override
    public ImmutableMap<String, String> getFixedEnvironment() {
      return fixedVars;
    }

    @Override
    public ImmutableSet<String> getInheritedEnvironment() {
      return inheritedVars;
    }
  }

  /** An empty {@link EnvironmentVariables}. */
  public static final EnvironmentVariables EMPTY_ENVIRONMENT_VARIABLES =
      new SimpleEnvironmentVariables(ImmutableMap.of(), ImmutableSet.of());

  /**
   * An empty environment, mainly for testing. Production code should never use this, but instead
   * get the proper environment from the current configuration.
   */
  // TODO(ulfjack): Migrate all production code to use the proper action environment, and then make
  // this @VisibleForTesting or rename it to clarify.
  public static final ActionEnvironment EMPTY = new ActionEnvironment(EMPTY_ENVIRONMENT_VARIABLES);

  /**
   * Splits the given map into a map of variables with a fixed value, and a set of variables that
   * should be inherited, the latter of which are identified by having a {@code null} value in the
   * given map. Returns these two parts as a new {@link ActionEnvironment} instance.
   */
  public static ActionEnvironment split(Map<String, String> env) {
    // Care needs to be taken that the two sets don't overlap - the order in which the two parts are
    // combined later is undefined.
    Map<String, String> fixedEnv = new TreeMap<>();
    Set<String> inheritedEnv = new TreeSet<>();
    for (Map.Entry<String, String> entry : env.entrySet()) {
      if (entry.getValue() != null) {
        fixedEnv.put(entry.getKey(), entry.getValue());
      } else {
        String key = entry.getKey();
        inheritedEnv.add(key);
      }
    }
    return create(SimpleEnvironmentVariables.create(fixedEnv, inheritedEnv));
  }

  private final EnvironmentVariables vars;

  private ActionEnvironment(EnvironmentVariables vars) {
    this.vars = vars;
  }

  /**
   * Creates a new action environment. The order in which the environments are combined is
   * undefined, so callers need to take care that the key set of the {@code fixedEnv} map and the
   * set of {@code inheritedEnv} elements are disjoint.
   */
  private static ActionEnvironment create(EnvironmentVariables vars) {
    if (vars.isEmpty()) {
      return EMPTY;
    }
    return new ActionEnvironment(vars);
  }

  public static ActionEnvironment create(
      Map<String, String> fixedEnv, ImmutableSet<String> inheritedEnv) {
    return ActionEnvironment.create(SimpleEnvironmentVariables.create(fixedEnv, inheritedEnv));
  }

  public static ActionEnvironment create(Map<String, String> fixedEnv) {
    return ActionEnvironment.create(SimpleEnvironmentVariables.create(fixedEnv, ImmutableSet.of()));
  }

  /**
   * Returns a copy of the environment with the given fixed variables added to it, <em>overwriting
   * any existing occurrences of those variables</em>.
   */
  public ActionEnvironment addFixedVariables(Map<String, String> fixedVars) {
    return ActionEnvironment.create(
        CompoundEnvironmentVariables.create(fixedVars, ImmutableSet.of(), vars));
  }

  /**
   * Returns a copy of the environment with the given fixed and inherited variables added to it,
   * <em>overwriting any existing occurrences of those variables</em>.
   */
  public ActionEnvironment addVariables(Map<String, String> fixedVars, Set<String> inheritedVars) {
    return ActionEnvironment.create(
        CompoundEnvironmentVariables.create(fixedVars, inheritedVars, vars));
  }

  /** Returns the combined size of the fixed and inherited environments. */
  public int size() {
    return vars.size();
  }

  /**
   * Returns the 'fixed' part of the environment, i.e., those environment variables that are set to
   * fixed values and their values. This should only be used for testing and to compute the cache
   * keys of actions. Use {@link #resolve} instead to get the complete environment.
   */
  public ImmutableMap<String, String> getFixedEnv() {
    return vars.getFixedEnvironment();
  }

  /**
   * Returns the 'inherited' part of the environment, i.e., those environment variables that are
   * inherited from the client environment and therefore have no fixed value here. This should only
   * be used for testing and to compute the cache keys of actions. Use {@link #resolve} instead to
   * get the complete environment.
   */
  public ImmutableSet<String> getInheritedEnv() {
    return vars.getInheritedEnvironment();
  }

  /**
   * Resolves the action environment and adds the resulting entries to the given {@code result} map,
   * by looking up any inherited env variables in the given {@code clientEnv}.
   *
   * <p>We pass in a map to mutate to avoid creating and merging intermediate maps.
   */
  public void resolve(Map<String, String> result, Map<String, String> clientEnv) {
    checkNotNull(clientEnv);
    result.putAll(vars.getFixedEnvironment());
    for (String var : vars.getInheritedEnvironment()) {
      String value = clientEnv.get(var);
      if (value != null) {
        result.put(var, value);
      }
    }
  }

  public void addTo(Fingerprint f) {
    f.addStringMap(vars.getFixedEnvironment());
    f.addStrings(vars.getInheritedEnvironment());
  }
}
