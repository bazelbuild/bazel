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
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.collect.Interner;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.analysis.config.CoreOptions.OutputPathsMode;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.skyframe.serialization.VisibleForSerialization;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.SerializationConstant;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.util.HashCodes;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;

/**
 * Environment variables for build or test actions.
 *
 * <p>The action environment consists of three parts.
 *
 * <ol>
 *   <li>All the environment variables with a fixed value, stored in a map. The value of such a
 *       variable is either a {@link String}, which is used as is, or an {@link Artifact}, which is
 *       resolved to its exec path in {@link #resolve(java.util.Map, java.util.Map,
 *       com.google.devtools.build.lib.actions.PathMapper)} only so that it can be path mapped.
 *   <li>All the environment variables inherited from the client environment, stored in a set.
 *   <li>All the environment variables that are explicitly unset, stored in a set. These are removed
 *       from the environment even if an earlier layer of the environment (e.g. the default test
 *       environment) set them.
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

  @SerializationConstant public static final ActionEnvironment EMPTY = new EmptyActionEnvironment();

  /** Convenience method for creating an {@link ActionEnvironment} with no inherited variables. */
  public static ActionEnvironment create(ImmutableMap<String, /* String | Artifact */ ?> fixedEnv) {
    return create(fixedEnv, /* inheritedEnv= */ ImmutableSet.of());
  }

  /**
   * Creates a new {@link ActionEnvironment} with no unset variables.
   *
   * <p>The values of {@code fixedEnv} must be of type {@link String} or {@link Artifact}.
   *
   * <p>If an environment variable is contained both as a key in {@code fixedEnv} and in {@code
   * inheritedEnv}, the result of {@link #resolve} will contain the value inherited from the client
   * environment.
   */
  public static ActionEnvironment create(
      ImmutableMap<String, /* String | Artifact */ ?> fixedEnv, ImmutableSet<String> inheritedEnv) {
    return create(fixedEnv, inheritedEnv, /* unsetEnv= */ ImmutableSet.of());
  }

  /**
   * Creates a new {@link ActionEnvironment}.
   *
   * <p>The values of {@code fixedEnv} must be of type {@link String} or {@link Artifact}.
   *
   * <p>If an environment variable is contained both as a key in {@code fixedEnv} and in {@code
   * inheritedEnv}, the result of {@link #resolve} will contain the value inherited from the client
   * environment.
   *
   * <p>Variables in {@code unsetEnv} are removed from the map passed to {@link #resolve}, even if
   * they were present in the map before the call or are contained in {@code fixedEnv} or {@code
   * inheritedEnv}. This allows an environment to remove variables that were set by an earlier layer
   * of the environment, e.g. the default test environment.
   */
  public static ActionEnvironment create(
      ImmutableMap<String, /* String | Artifact */ ?> fixedEnv,
      ImmutableSet<String> inheritedEnv,
      ImmutableSet<String> unsetEnv) {
    if (fixedEnv.isEmpty() && inheritedEnv.isEmpty() && unsetEnv.isEmpty()) {
      return EMPTY;
    }
    return SimpleActionEnvironment.of(ImmutableMap.copyOf(fixedEnv), inheritedEnv, unsetEnv);
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
        ImmutableSet.copyOf(unsetEnv));
  }

  private static String maybeMapValue(Object value, PathMapper pathMapper) {
    return switch (value) {
      case String s -> s;
      case Artifact artifact -> pathMapper.getMappedExecPathString(artifact);
      default ->
          throw new IllegalStateException(
              "Expected env value of type String or Artifact, but got: " + value);
    };
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
   * by looking up any inherited env variables in the given {@code clientEnv} and resolving any
   * artifact-valued env variables to their unmapped exec paths. Afterwards, removes all unset env
   * variables from the map.
   *
   * <p>We pass in a map to mutate to avoid creating and merging intermediate maps.
   *
   * <p>Note: Prefer {@link #resolve(Map, Map, PathMapper)} if a {@link PathMapper} is available,
   * i.e., during action execution.
   */
  public final void resolve(Map<String, String> result, Map<String, String> clientEnv) {
    resolve(result, clientEnv, PathMapper.NOOP);
  }

  /**
   * Resolves the action environment and adds the resulting entries to the given {@code result} map,
   * by looking up any inherited env variables in the given {@code clientEnv} and resolving any
   * artifact-valued env variables to their exec paths with the given {@link PathMapper} applied.
   * Afterwards, removes all unset env variables from the map.
   *
   * <p>Actions that support path mapping should pass in the path mapper of the spawn being created
   * so that artifact-valued env variables reflect the paths seen by the action at execution time.
   *
   * <p>We pass in a map to mutate to avoid creating and merging intermediate maps.
   */
  public final void resolve(
      Map<String, String> result, Map<String, String> clientEnv, PathMapper pathMapper) {
    checkNotNull(clientEnv);
    result.putAll(Maps.transformValues(fixedEnv(), value -> maybeMapValue(value, pathMapper)));
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

  /**
   * Adds this environment to the given fingerprint, resolving artifact-valued env variables with
   * the given {@link PathMapper}.
   */
  public final void addTo(OutputPathsMode effectiveOutputPathsMode, Fingerprint f) {
    f.addStringMap(
        Maps.transformValues(
            fixedEnv(),
            value -> maybeMapValue(value, PathMapper.forActionKey(effectiveOutputPathsMode))));
    f.addStrings(getInheritedEnv());
    f.addStrings(getUnsetEnv());
  }

  /**
   * Returns a copy of the environment with the given fixed variables added to it, <em>overwriting
   * any existing occurrences of those variables</em>, including any that were previously unset.
   *
   * <p>The values of {@code fixedVars} must be of type {@link String} or {@link Artifact}.
   */
  public final ActionEnvironment withAdditionalFixedVariables(
      Map<String, /* String | Artifact */ ?> fixedVars) {
    if (fixedVars.isEmpty()) {
      return this;
    }
    if (this == EMPTY) {
      return create(ImmutableMap.copyOf(fixedVars), ImmutableSet.of(), ImmutableSet.of());
    }
    return CompoundActionEnvironment.of(this, ImmutableMap.copyOf(fixedVars));
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
    public ImmutableSet<String> getUnsetEnv() {
      return ImmutableSet.of();
    }

    @Override
    public int estimatedSize() {
      return 0;
    }
  }

  @AutoCodec
  @VisibleForSerialization
  static final class SimpleActionEnvironment extends ActionEnvironment {
    private static final Interner<SimpleActionEnvironment> interner =
        BlazeInterners.newWeakInterner();

    private final ImmutableMap<String, /* String | Artifact */ Object> fixedEnv;
    private final ImmutableSet<String> inheritedEnv;
    private final ImmutableSet<String> unsetEnv;

    private SimpleActionEnvironment(
        ImmutableMap<String, Object> fixedEnv,
        ImmutableSet<String> inheritedEnv,
        ImmutableSet<String> unsetEnv) {
      this.fixedEnv = fixedEnv;
      this.inheritedEnv = inheritedEnv;
      this.unsetEnv = unsetEnv;
    }

    @AutoCodec.Instantiator
    static SimpleActionEnvironment of(
        ImmutableMap<String, Object> fixedEnv,
        ImmutableSet<String> inheritedEnv,
        ImmutableSet<String> unsetEnv) {
      return interner.intern(
          new SimpleActionEnvironment(fixedEnv, inheritedEnv, ImmutableSortedSet.copyOf(unsetEnv)));
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
      return HashCodes.hashObjects(fixedEnv, inheritedEnv, unsetEnv);
    }
  }

  @AutoCodec
  @VisibleForSerialization
  static final class CompoundActionEnvironment extends ActionEnvironment {
    private static final Interner<CompoundActionEnvironment> interner =
        BlazeInterners.newWeakInterner();

    private final ActionEnvironment base;
    private final ImmutableMap<String, /* String | Artifact */ Object> fixedVars;

    private CompoundActionEnvironment(
        ActionEnvironment base, ImmutableMap<String, Object> fixedVars) {
      this.base = base;
      this.fixedVars = fixedVars;
    }

    @AutoCodec.Instantiator
    static CompoundActionEnvironment of(
        ActionEnvironment base, ImmutableMap<String, Object> fixedVars) {
      return interner.intern(new CompoundActionEnvironment(base, fixedVars));
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
      return HashCodes.hashObjects(base, fixedVars);
    }
  }
}
