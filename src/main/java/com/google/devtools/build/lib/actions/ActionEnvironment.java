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
public abstract class ActionEnvironment {

  public static final ActionEnvironment EMPTY = new EmptyActionEnvironment();

  /** Convenience method for creating an {@link ActionEnvironment} with no inherited variables. */
  public static ActionEnvironment create(ImmutableMap<String, String> fixedEnv) {
    return create(fixedEnv, /* inheritedEnv= */ ImmutableSet.of());
  }

  /**
   * Creates a new {@link ActionEnvironment}.
   *
   * <p>If an environment variable is contained both as a key in {@code fixedEnv} and in {@code
   * inheritedEnv}, the result of {@link #resolve} will contain the value inherited from the client
   * environment.
   */
  public static ActionEnvironment create(
      ImmutableMap<String, String> fixedEnv, ImmutableSet<String> inheritedEnv) {
    if (fixedEnv.isEmpty() && inheritedEnv.isEmpty()) {
      return EMPTY;
    }
    return new SimpleActionEnvironment(fixedEnv, inheritedEnv);
  }

  /**
   * Splits the given map into a map of variables with a fixed value, and a set of variables that
   * should be inherited, the latter of which are identified by having a {@code null} value in the
   * given map. Returns these two parts as a new {@link ActionEnvironment} instance.
   */
  public static ActionEnvironment split(Map<String, String> env) {
    Map<String, String> fixedEnv = new TreeMap<>();
    Set<String> inheritedEnv = new TreeSet<>();
    for (Map.Entry<String, String> entry : env.entrySet()) {
      if (entry.getValue() != null) {
        fixedEnv.put(entry.getKey(), entry.getValue());
      } else {
        inheritedEnv.add(entry.getKey());
      }
    }
    return create(ImmutableMap.copyOf(fixedEnv), ImmutableSet.copyOf(inheritedEnv));
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
   * Returns an upper bound on the combined size of the fixed and inherited environments. A call to
   * {@link #resolve} may add fewer entries than this number if environment variables are contained
   * in both the fixed and the inherited environment.
   */
  public abstract int estimatedSize();

  /**
   * Resolves the action environment and adds the resulting entries to the given {@code result} map,
   * by looking up any inherited env variables in the given {@code clientEnv}.
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
  }

  public final void addTo(Fingerprint f) {
    f.addStringMap(getFixedEnv());
    f.addStrings(getInheritedEnv());
  }

  /**
   * Returns a copy of the environment with the given fixed variables added to it, <em>overwriting
   * any existing occurrences of those variables</em>.
   */
  public final ActionEnvironment withAdditionalFixedVariables(Map<String, String> fixedVars) {
    return withAdditionalVariables(fixedVars, ImmutableSet.of());
  }

  /**
   * Returns a copy of this environment with the given fixed and inherited variables added to it,
   * <em>overwriting any existing occurrences of those variables</em>.
   */
  public final ActionEnvironment withAdditionalVariables(
      Map<String, String> fixedVars, Set<String> inheritedVars) {
    if (fixedVars.isEmpty() && inheritedVars.isEmpty()) {
      return this;
    }
    if (this == EMPTY) {
      return new SimpleActionEnvironment(
          ImmutableMap.copyOf(fixedVars), ImmutableSet.copyOf(inheritedVars));
    }
    return new CompoundActionEnvironment(
        this, ImmutableMap.copyOf(fixedVars), ImmutableSet.copyOf(inheritedVars));
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
    public int estimatedSize() {
      return 0;
    }
  }

  private static final class SimpleActionEnvironment extends ActionEnvironment {
    private final ImmutableMap<String, String> fixedEnv;
    private final ImmutableSet<String> inheritedEnv;

    SimpleActionEnvironment(
        ImmutableMap<String, String> fixedEnv, ImmutableSet<String> inheritedEnv) {
      this.fixedEnv = fixedEnv;
      this.inheritedEnv = inheritedEnv;
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
    public int estimatedSize() {
      return fixedEnv.size() + inheritedEnv.size();
    }
  }

  private static final class CompoundActionEnvironment extends ActionEnvironment {
    private final ActionEnvironment base;
    private final ImmutableMap<String, String> fixedVars;
    private final ImmutableSet<String> inheritedVars;

    private CompoundActionEnvironment(
        ActionEnvironment base,
        ImmutableMap<String, String> fixedVars,
        ImmutableSet<String> inheritedVars) {
      this.base = base;
      this.fixedVars = fixedVars;
      this.inheritedVars = inheritedVars;
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
      return ImmutableSet.<String>builder()
          .addAll(base.getInheritedEnv())
          .addAll(inheritedVars)
          .build();
    }

    @Override
    public int estimatedSize() {
      return base.estimatedSize() + fixedVars.size() + inheritedVars.size();
    }
  }
}
