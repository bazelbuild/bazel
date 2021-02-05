// Copyright 2020 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableMap;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import net.starlark.java.eval.Starlark;

/**
 * This class encapsulates knowledge of how to set up the Starlark environment for BUILD, WORKSPACE,
 * and bzl file evaluation, including the top-level predeclared symbols, the {@code native} module,
 * and the special environment for {@code @_builtins} bzl evaluation.
 *
 * <p>The set of available symbols is determined by 1) gathering registered toplevels, rules,
 * extensions, etc., from the {@link ConfiguredRuleClassProvider} and {@link PackageFactory}, and
 * then 2) applying builtins injection (see {@link StarlarkBuiltinsFunction}, if applicable. The
 * result of (1) is cached by an instance of this class. (2) is obtained using helper methods on
 * this class, and cached in {@link StarlarkBuiltinsValue}.
 */
// TODO(#11437): To deal with StarlarkSemantics inspection via _builtins.flags, we can make it a
// function (not a struct field) that accesses the StarlarkThread. To deal with flag-guarded
// top-level symbols exposed under _builtins.toplevel, we can simply make them all accessible
// unconditionally, under the principle that the flag migrator should be responsible for deleting
// all uses from @_builtins code, and @_builtins code can be trusted to use flag-guarded features
// responsibly.
public final class BazelStarlarkEnvironment {

  // TODO(#11954): Eventually the BUILD and WORKSPACE bzl dialects should converge. Right now they
  // only differ on the "native" object.

  private final RuleClassProvider ruleClassProvider;
  private final ImmutableMap<String, ?> ruleFunctions;

  /** The "native" module fields for a BUILD-loaded bzl module, before builtins injection. */
  private final ImmutableMap<String, Object> uninjectedBuildBzlNativeBindings;
  /** The "native" module fields for a WORKSPACE-loaded bzl module. */
  private final ImmutableMap<String, Object> workspaceBzlNativeBindings;
  /** The top-level predeclared symbols for a BUILD-loaded bzl module, before builtins injection. */
  private final ImmutableMap<String, Object> uninjectedBuildBzlEnv;
  /** The top-level predeclared symbols for BUILD files, before builtins injection and prelude. */
  private final ImmutableMap<String, Object> uninjectedBuildEnv;
  /** The top-level predeclared symbols for a WORKSPACE-loaded bzl module. */
  private final ImmutableMap<String, Object> workspaceBzlEnv;
  /** The top-level predeclared symbols for a bzl module in the {@code @_builtins} pseudo-repo. */
  private final ImmutableMap<String, Object> builtinsBzlEnv;

  BazelStarlarkEnvironment(
      RuleClassProvider ruleClassProvider,
      ImmutableMap<String, ?> ruleFunctions,
      List<PackageFactory.EnvironmentExtension> environmentExtensions,
      Object packageFunction,
      String version) {
    this.ruleClassProvider = ruleClassProvider;
    this.ruleFunctions = ruleFunctions;
    this.uninjectedBuildBzlNativeBindings =
        createUninjectedBuildBzlNativeBindings(
            ruleFunctions, packageFunction, environmentExtensions);
    this.workspaceBzlNativeBindings = createWorkspaceBzlNativeBindings(ruleClassProvider, version);
    this.uninjectedBuildBzlEnv =
        createUninjectedBuildBzlEnv(ruleClassProvider, uninjectedBuildBzlNativeBindings);
    this.workspaceBzlEnv = createWorkspaceBzlEnv(ruleClassProvider, workspaceBzlNativeBindings);
    this.builtinsBzlEnv = createBuiltinsBzlEnv(ruleClassProvider);
    this.uninjectedBuildEnv =
        createUninjectedBuildEnv(ruleFunctions, packageFunction, environmentExtensions);
  }

  /**
   * Returns the contents of the "native" object for BUILD-loaded bzls, not accounting for builtins
   * injection.
   */
  public ImmutableMap<String, Object> getUninjectedBuildBzlNativeBindings() {
    return uninjectedBuildBzlNativeBindings;
  }

  /** Returns the contents of the "native" object for WORKSPACE-loaded bzls. */
  public ImmutableMap<String, Object> getWorkspaceBzlNativeBindings() {
    return workspaceBzlNativeBindings;
  }

  /**
   * Returns the original environment for BUILD-loaded bzl files, not accounting for builtins
   * injection.
   *
   * <p>The post-injection environment may differ from this one by what symbols a name is bound to,
   * but the set of symbols remains the same.
   */
  public ImmutableMap<String, Object> getUninjectedBuildBzlEnv() {
    return uninjectedBuildBzlEnv;
  }

  /**
   * Returns the original environment for BUILD files, not accounting for builtins injection or
   * application of the prelude.
   *
   * <p>Applying builtins injection may update name bindings, but not add or remove them. I.e. some
   * names may refer to different symbols but the static set of names remains the same. Applying the
   * prelude file may update and add name bindings but not remove them.
   */
  public ImmutableMap<String, Object> getUninjectedBuildEnv() {
    return uninjectedBuildEnv;
  }

  /** Returns the environment for WORKSPACE-loaded bzl files. */
  public ImmutableMap<String, Object> getWorkspaceBzlEnv() {
    return workspaceBzlEnv;
  }

  /** Returns the environment for bzl files in the {@code @_builtins} pseudo-repository. */
  public ImmutableMap<String, Object> getBuiltinsBzlEnv() {
    return builtinsBzlEnv;
  }

  /**
   * Produces everything that would be in the "native" object for BUILD-loaded bzl files if builtins
   * injection didn't happen.
   */
  private static ImmutableMap<String, Object> createUninjectedBuildBzlNativeBindings(
      Map<String, ?> ruleFunctions,
      Object packageFunction,
      List<PackageFactory.EnvironmentExtension> environmentExtensions) {
    ImmutableMap.Builder<String, Object> env = new ImmutableMap.Builder<>();
    env.putAll(StarlarkNativeModule.BINDINGS_FOR_BUILD_FILES);
    env.putAll(ruleFunctions);
    env.put("package", packageFunction);
    for (PackageFactory.EnvironmentExtension ext : environmentExtensions) {
      ext.updateNative(env);
    }
    return env.build();
  }

  /** Produces everything in the "native" object for WORKSPACE-loaded bzl files. */
  private static ImmutableMap<String, Object> createWorkspaceBzlNativeBindings(
      RuleClassProvider ruleClassProvider, String version) {
    return WorkspaceFactory.createNativeModuleBindings(ruleClassProvider, version);
  }

  /** Constructs a "native" module object with the given contents. */
  private static Object createNativeModule(Map<String, Object> bindings) {
    return StructProvider.STRUCT.create(bindings, "no native function or rule '%s'");
  }

  private static ImmutableMap<String, Object> createUninjectedBuildBzlEnv(
      RuleClassProvider ruleClassProvider, Map<String, Object> uninjectedBuildBzlNativeBindings) {
    Map<String, Object> env = new HashMap<>();
    env.putAll(ruleClassProvider.getEnvironment());

    // Determine the "native" module.
    // TODO(#11954): Use the same "native" object for both BUILD- and WORKSPACE-loaded .bzls, and
    // just have it be a dynamic error to call the wrong thing at the wrong time. This is a breaking
    // change.
    env.put("native", createNativeModule(uninjectedBuildBzlNativeBindings));

    return ImmutableMap.copyOf(env);
  }

  private static ImmutableMap<String, Object> createUninjectedBuildEnv(
      Map<String, ?> ruleFunctions,
      Object packageFunction,
      List<PackageFactory.EnvironmentExtension> environmentExtensions) {
    ImmutableMap.Builder<String, Object> env = ImmutableMap.builder();
    env.putAll(StarlarkLibrary.BUILD); // e.g. rule, select, depset
    env.putAll(StarlarkNativeModule.BINDINGS_FOR_BUILD_FILES);
    env.put("package", packageFunction);
    env.putAll(ruleFunctions);
    for (PackageFactory.EnvironmentExtension ext : environmentExtensions) {
      ext.update(env);
    }
    return env.build();
  }

  private static ImmutableMap<String, Object> createWorkspaceBzlEnv(
      RuleClassProvider ruleClassProvider, Map<String, Object> workspaceBzlNativeBindings) {
    Map<String, Object> env = new HashMap<>();
    env.putAll(ruleClassProvider.getEnvironment());

    // See above comments for native in BUILD bzls.
    env.put("native", createNativeModule(workspaceBzlNativeBindings));

    return ImmutableMap.copyOf(env);
  }

  private static ImmutableMap<String, Object> createBuiltinsBzlEnv(
      RuleClassProvider ruleClassProvider) {
    Map<String, Object> env = new HashMap<>();
    env.putAll(ruleClassProvider.getEnvironment());

    // Clear out rule-specific symbols like CcInfo.
    env.keySet().removeAll(ruleClassProvider.getNativeRuleSpecificBindings().keySet());

    env.put("_internal", InternalModule.INSTANCE);

    return ImmutableMap.copyOf(env);
  }

  /**
   * Constructs an environment for a BUILD-loaded bzl file based on the default environment as well
   * as the given {@code @_builtins}-injected top-level symbols and "native" bindings.
   *
   * <p>Injected symbols must override an existing symbol of that name. Furthermore, the overridden
   * symbol must be a rule or a piece of a specific ruleset's logic (e.g., {@code CcInfo} or {@code
   * cc_library}), not a generic built-in (e.g., {@code provider} or {@code glob}). Throws
   * InjectionException if these conditions are not met.
   *
   * @see StarlarkBuiltinsFunction
   */
  public ImmutableMap<String, Object> createBuildBzlEnvUsingInjection(
      Map<String, Object> injectedToplevels, Map<String, Object> injectedRules)
      throws InjectionException {
    // TODO(#11437): Builtins injection should take into account StarlarkSemantics and
    // FlagGuardedValues. If a builtin is disabled by a flag, we can either:
    //
    //   1) Treat it as if it doesn't exist for the purposes of injection. In this case it's an
    //      error to attempt to inject it, so exports.bzl is required to explicitly check the flag's
    //      value (via the _internal module) before exporting it.
    //
    //   2) Allow it to be exported and automatically suppress/omit it from the final environment,
    //      effectively rewrapping the injected builtin in the FlagGuardedValue.

    // Determine top-level symbols.
    Map<String, Object> env = new HashMap<>();
    env.putAll(uninjectedBuildBzlEnv);
    for (Map.Entry<String, Object> symbol : injectedToplevels.entrySet()) {
      String name = symbol.getKey();
      if (!env.containsKey(name) && !Starlark.UNIVERSE.containsKey(name)) {
        throw new InjectionException(
            String.format(
                "Injected top-level symbol '%s' must override an existing symbol by that name",
                name));
      } else if (!ruleClassProvider.getNativeRuleSpecificBindings().containsKey(name)) {
        throw new InjectionException(
            String.format("Cannot override top-level builtin '%s' with an injected value", name));
      } else {
        env.put(name, symbol.getValue());
      }
    }

    // Determine "native" bindings.
    // See above comments for native in BUILD bzls.
    Map<String, Object> nativeBindings = new HashMap<>();
    nativeBindings.putAll(uninjectedBuildBzlNativeBindings);
    for (Map.Entry<String, Object> symbol : injectedRules.entrySet()) {
      String name = symbol.getKey();
      Object preexisting = nativeBindings.put(name, symbol.getValue());
      if (preexisting == null) {
        throw new InjectionException(
            String.format("Injected rule '%s' must override an existing rule by that name", name));
      } else if (!ruleFunctions.containsKey(name)) {
        throw new InjectionException(
            String.format("Cannot override native module field '%s' with an injected value", name));
      }
    }

    env.put("native", createNativeModule(nativeBindings));
    return ImmutableMap.copyOf(env);
  }

  /**
   * Constructs an environment for a BUILD file based on the default environment and the given
   * {@code @_builtins}-injected rule symbols.
   *
   * <p>Injected rule symbols must override an existing native rule of that name. Only rules may be
   * overridden in this manner, not generic built-ins such as {@code package} or {@code glob}.
   * Throws InjectionException if these conditions are not met.
   */
  // TODO(adonovan): move skyframe.PackageFunction into lib.packages so we needn't expose this and
  // the other env-building functions.
  public ImmutableMap<String, Object> createBuildEnvUsingInjection(
      Map<String, Object> injectedRules) throws InjectionException {
    HashMap<String, Object> env = new HashMap<>(uninjectedBuildEnv);
    for (Map.Entry<String, Object> symbol : injectedRules.entrySet()) {
      String name = symbol.getKey();
      if (!env.containsKey(name) && !Starlark.UNIVERSE.containsKey(name)) {
        throw new InjectionException(
            String.format("Injected rule '%s' must override an existing rule by that name", name));
      } else if (!ruleFunctions.containsKey(name)) {
        throw new InjectionException(
            String.format("Cannot override top-level builtin '%s' with an injected value", name));
      } else {
        env.put(name, symbol.getValue());
      }
    }
    return ImmutableMap.copyOf(env);
  }

  /** Indicates a problem performing builtins injection. */
  public static final class InjectionException extends Exception {
    public InjectionException(String message) {
      super(message);
    }
  }
}
