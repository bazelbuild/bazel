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
import com.google.common.collect.Maps;
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
 * <p>The action environment consists of two parts.
 *
 * <ol>
 *   <li>All the environment variables with a fixed value, stored in a map. The value of such a
 *       variable is either a {@link String}, which is used as is, or an {@link Artifact}, which is
 *       resolved to its exec path at execution time so that any applicable {@link PathMapper} can
 *       be applied to it.
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

  private static final Interner<ActionEnvironment> actionEnvironmentInterner =
      BlazeInterners.newWeakInterner();

  /** Convenience method for creating an {@link ActionEnvironment} with no inherited variables. */
  public static ActionEnvironment create(ImmutableMap<String, ?> fixedEnv) {
    return create(fixedEnv, /* inheritedEnv= */ ImmutableSet.of());
  }

  /**
   * Creates a new {@link ActionEnvironment}.
   *
   * <p>The values of {@code fixedEnv} must be of type {@link String} or {@link Artifact}.
   *
   * <p>If an environment variable is contained both as a key in {@code fixedEnv} and in {@code
   * inheritedEnv}, the result of {@link #resolve} will contain the value inherited from the client
   * environment.
   */
  public static ActionEnvironment create(
      ImmutableMap<String, ?> fixedEnv, ImmutableSet<String> inheritedEnv) {
    if (fixedEnv.isEmpty() && inheritedEnv.isEmpty()) {
      return EMPTY;
    }
    // copyOf returns the given map unchanged, with its value type widened to Object.
    return actionEnvironmentInterner.intern(
        new SimpleActionEnvironment(ImmutableMap.copyOf(fixedEnv), inheritedEnv));
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

  private static String maybeMapValue(Object value, PathMapper pathMapper) {
    return value instanceof Artifact artifact
        ? pathMapper.getMappedExecPathString(artifact)
        : (String) value;
  }

  private ActionEnvironment() {}

  /**
   * Returns the 'fixed' part of the environment, with {@link String} values used as is and {@link
   * Artifact} values resolved to their unmapped exec paths. This should only be used for testing
   * and analysis-time introspection (e.g. aquery), as the value of an artifact-valued variable seen
   * by the action may differ due to path mapping. Use {@link #resolve} instead to get the complete
   * environment.
   */
  public final ImmutableMap<String, String> getFixedEnv() {
    ImmutableMap<String, Object> fixedEnv = fixedEnv();
    if (!hasArtifactValues(fixedEnv)) {
      // Covariant cast of a map without Artifact values.
      @SuppressWarnings("unchecked")
      var stringEnv = (ImmutableMap<String, String>) (ImmutableMap<?, ?>) fixedEnv;
      return stringEnv;
    }
    return ImmutableMap.copyOf(
        Maps.transformValues(fixedEnv, value -> maybeMapValue(value, PathMapper.NOOP)));
  }

  private static boolean hasArtifactValues(ImmutableMap<String, Object> fixedEnv) {
    return fixedEnv.values().stream().anyMatch(value -> value instanceof Artifact);
  }

  /**
   * Returns the 'fixed' part of the environment, with values that are either of type {@link String}
   * or {@link Artifact}.
   */
  abstract ImmutableMap<String, /* String | Artifact */ Object> fixedEnv();

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
   * by looking up any inherited env variables in the given {@code clientEnv} and resolving any
   * artifact-valued env variables to their unmapped exec paths.
   *
   * <p>We pass in a map to mutate to avoid creating and merging intermediate maps.
   */
  public final void resolve(Map<String, String> result, Map<String, String> clientEnv) {
    checkNotNull(clientEnv);
    result.putAll(Maps.transformValues(fixedEnv(), value -> maybeMapValue(value, PathMapper.NOOP)));
    for (String var : getInheritedEnv()) {
      String value = clientEnv.get(var);
      if (value != null) {
        result.put(var, value);
      }
    }
  }

  /**
   * Like {@link #resolve}, but keeps {@link Artifact} values unresolved so that callers can later
   * resolve them with a {@link PathMapper} applied via {@link #resolveValues}.
   */
  public final void resolveKeepingArtifacts(
      Map<String, /* String | Artifact */ Object> result, Map<String, String> clientEnv) {
    checkNotNull(clientEnv);
    result.putAll(fixedEnv());
    for (String var : getInheritedEnv()) {
      String value = clientEnv.get(var);
      if (value != null) {
        result.put(var, value);
      }
    }
  }

  /**
   * Resolves the values of an environment as returned by {@link Action#getEffectiveEnvironment}:
   * {@link String} values are used as is and {@link Artifact} values are replaced by their exec
   * paths with the given {@link PathMapper} applied.
   *
   * <p>Actions that support path mapping should pass in the path mapper of the spawn being created
   * so that artifact-valued env variables reflect the paths seen by the action at execution time.
   */
  public static ImmutableMap<String, String> resolveValues(
      Map<String, /* String | Artifact */ Object> env, PathMapper pathMapper) {
    return ImmutableMap.copyOf(
        Maps.transformValues(env, value -> maybeMapValue(value, pathMapper)));
  }

  public final void addTo(Fingerprint f) {
    addTo(PathMapper.NOOP, f);
  }

  /**
   * Adds this environment to the given fingerprint, resolving artifact-valued env variables with
   * the given {@link PathMapper}.
   *
   * <p>Actions that support path mapping should pass in the path mapper returned by {@code
   * PathMapper.forActionKey} to ensure that the action key remains stable across configurations if
   * and only if the resolved environment does.
   */
  public final void addTo(PathMapper pathMapper, Fingerprint f) {
    f.addStringMap(Maps.transformValues(fixedEnv(), value -> maybeMapValue(value, pathMapper)));
    f.addStrings(getInheritedEnv());
  }

  /**
   * Returns a copy of the environment with the given fixed variables added to it, <em>overwriting
   * any existing occurrences of those variables</em>.
   *
   * <p>The values of {@code fixedVars} must be of type {@link String} or {@link Artifact}.
   */
  public final ActionEnvironment withAdditionalFixedVariables(Map<String, ?> fixedVars) {
    if (fixedVars.isEmpty()) {
      return this;
    }
    if (this == EMPTY) {
      return actionEnvironmentInterner.intern(
          new SimpleActionEnvironment(ImmutableMap.copyOf(fixedVars), ImmutableSet.of()));
    }
    return actionEnvironmentInterner.intern(
        new CompoundActionEnvironment(this, ImmutableMap.copyOf(fixedVars)));
  }

  private static final class EmptyActionEnvironment extends ActionEnvironment {

    @Override
    ImmutableMap<String, Object> fixedEnv() {
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
    private final ImmutableMap<String, /* String | Artifact */ Object> fixedEnv;
    private final ImmutableSet<String> inheritedEnv;

    SimpleActionEnvironment(
        ImmutableMap<String, Object> fixedEnv, ImmutableSet<String> inheritedEnv) {
      this.fixedEnv = fixedEnv;
      this.inheritedEnv = inheritedEnv;
    }

    @Override
    ImmutableMap<String, Object> fixedEnv() {
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

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (!(o instanceof SimpleActionEnvironment that)) {
        return false;
      }
      return fixedEnv.equals(that.fixedEnv) && inheritedEnv.equals(that.inheritedEnv);
    }

    @Override
    public int hashCode() {
      return Objects.hash(fixedEnv, inheritedEnv);
    }
  }

  private static final class CompoundActionEnvironment extends ActionEnvironment {
    private final ActionEnvironment base;
    private final ImmutableMap<String, /* String | Artifact */ Object> fixedVars;

    private CompoundActionEnvironment(
        ActionEnvironment base, ImmutableMap<String, Object> fixedVars) {
      this.base = base;
      this.fixedVars = fixedVars;
    }

    @Override
    ImmutableMap<String, Object> fixedEnv() {
      return ImmutableMap.<String, Object>builder()
          .putAll(base.fixedEnv())
          .putAll(fixedVars)
          .buildKeepingLast();
    }

    @Override
    public ImmutableSet<String> getInheritedEnv() {
      return base.getInheritedEnv();
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
