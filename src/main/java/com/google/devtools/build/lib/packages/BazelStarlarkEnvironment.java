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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Sets;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import net.starlark.java.eval.FlagGuardedValue;
import net.starlark.java.eval.Starlark;

// TODO(adonovan): move skyframe.PackageFunction into lib.packages so we needn't expose this and
// the other env-building functions.
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
    this.builtinsBzlEnv =
        createBuiltinsBzlEnv(
            ruleClassProvider, uninjectedBuildBzlNativeBindings, uninjectedBuildBzlEnv);
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
      RuleClassProvider ruleClassProvider,
      ImmutableMap<String, Object> uninjectedBuildBzlNativeBindings,
      ImmutableMap<String, Object> uninjectedBuildBzlEnv) {
    Map<String, Object> env = new HashMap<>();
    env.putAll(ruleClassProvider.getEnvironment());

    // Clear out rule-specific symbols like CcInfo.
    env.keySet().removeAll(ruleClassProvider.getNativeRuleSpecificBindings().keySet());

    // For _builtins.toplevel, replace all FlagGuardedValues with the underlying value;
    // StarlarkSemantics flags do not affect @_builtins.
    //
    // We do this because otherwise we'd need to differentiate the _builtins.toplevel object (and
    // therefore the @_builtins environment) based on StarlarkSemantics. That seems unnecessary.
    // Instead we trust @_builtins to not misuse flag-guarded features, same as native code.
    //
    // If foo is flag-guarded (either experimental or incompatible), it is unconditionally visible
    // as _builtins.toplevel.foo. It is legal to list it in exported_toplevels unconditionally, but
    // the flag still controls whether the symbol is actually visible to user code.
    Map<String, Object> unwrappedBuildBzlSymbols = new HashMap<>();
    for (Map.Entry<String, Object> entry : uninjectedBuildBzlEnv.entrySet()) {
      Object symbol = entry.getValue();
      if (symbol instanceof FlagGuardedValue) {
        symbol = ((FlagGuardedValue) symbol).getObject();
      }
      unwrappedBuildBzlSymbols.put(entry.getKey(), symbol);
    }

    Object builtinsModule =
        new BuiltinsInternalModule(
            createNativeModule(uninjectedBuildBzlNativeBindings),
            // createNativeModule() is good enough for the "toplevel" and "internal" objects too.
            createNativeModule(unwrappedBuildBzlSymbols),
            createNativeModule(ruleClassProvider.getStarlarkBuiltinsInternals()));
    Object conflictingValue = env.put("_builtins", builtinsModule);
    Preconditions.checkState(
        conflictingValue == null, "'_builtins' name is reserved for builtins injection");

    return ImmutableMap.copyOf(env);
  }

  /**
   * Throws {@link InjectionException} with an appropriate error message if the given {@code symbol}
   * is not in both {@code existingSymbols} and {@code injectableSymbols}. {@code kind} is a string
   * describing the domain of {@code symbol}.
   */
  private static void validateSymbolIsInjectable(
      String symbol, Set<String> existingSymbols, Set<String> injectableSymbols, String kind)
      throws InjectionException {
    if (!existingSymbols.contains(symbol)) {
      throw new InjectionException(
          String.format(
              "Injected %s '%s' must override an existing one by that name", kind, symbol));
    } else if (!injectableSymbols.contains(symbol)) {
      throw new InjectionException(
          String.format("Cannot override '%s' with an injected %s", symbol, kind));
    }
  }

  /** Given a string prefixed with + or -, returns that prefix character, or null otherwise. */
  private static Character getKeyPrefix(String key) {
    if (key.isEmpty()) {
      return null;
    }
    char prefix = key.charAt(0);
    if (!(prefix == '+' || prefix == '-')) {
      return null;
    }
    return prefix;
  }

  /**
   * Given a string prefixed with + or -, returns the remainder of the string, or the whole string
   * otherwise.
   */
  private static String getKeySuffix(String key) {
    return getKeyPrefix(key) == null ? key : key.substring(1);
  }

  /**
   * Given a list of strings representing the +/- prefixed items in {@code
   * --experimental_builtins_injection_override}, returns a map from each item to a Boolean
   * indicating whether it last appeared with the + suffix (True) or - suffix (False).
   *
   * @throws InjectionException if an item is not prefixed with either "+" or "-"
   */
  private static Map<String, Boolean> parseInjectionOverridesList(List<String> overrides)
      throws InjectionException {
    HashMap<String, Boolean> result = new HashMap<>();
    for (String prefixedItem : overrides) {
      Character prefix = getKeyPrefix(prefixedItem);
      if (prefix == null) {
        throw new InjectionException(
            String.format("Invalid injection override item: '%s'", prefixedItem));
      }
      result.put(prefixedItem.substring(1), prefix == '+');
    }
    return result;
  }

  /**
   * Given an exports dict key, and an override map, return whether injection should be applied for
   * that key.
   */
  private static boolean injectionApplies(String key, Map<String, Boolean> overrides) {
    Character prefix = getKeyPrefix(key);
    if (prefix == null) {
      // Unprefixed; overrides don't get a say in the matter.
      return true;
    }
    Boolean override = overrides.get(key.substring(1));
    if (override == null) {
      return prefix == '+';
    } else {
      return override;
    }
  }

  /**
   * Constructs an environment for a BUILD-loaded bzl file based on the default environment, the
   * maps corresponding to the {@code exported_toplevels} and {@code exported_rules} dicts, and the
   * value of {@code --experimental_builtins_injection_override}.
   *
   * <p>Injected symbols must override an existing symbol of that name. Furthermore, the overridden
   * symbol must be a rule or a piece of a specific ruleset's logic (e.g., {@code CcInfo} or {@code
   * cc_library}), not a generic built-in (e.g., {@code provider} or {@code glob}). Throws
   * InjectionException if these conditions are not met.
   *
   * <p>Whether or not injection actually occurs for a given map key depends on its prefix (if any)
   * and the prefix of its appearance (if it appears at all) in the override list; see the
   * documentation for {@code --experimental_builtins_injection_override}. Non-injected symbols must
   * still obey the above constraints.
   *
   * @see StarlarkBuiltinsFunction
   */
  public ImmutableMap<String, Object> createBuildBzlEnvUsingInjection(
      Map<String, Object> exportedToplevels,
      Map<String, Object> exportedRules,
      List<String> overridesList)
      throws InjectionException {
    Map<String, Boolean> overridesMap = parseInjectionOverridesList(overridesList);

    // Determine top-level symbols.
    Map<String, Object> env = new HashMap<>();
    env.putAll(uninjectedBuildBzlEnv);
    for (Map.Entry<String, Object> entry : exportedToplevels.entrySet()) {
      String key = entry.getKey();
      String name = getKeySuffix(key);
      validateSymbolIsInjectable(
          name,
          Sets.union(env.keySet(), Starlark.UNIVERSE.keySet()),
          ruleClassProvider.getNativeRuleSpecificBindings().keySet(),
          "top-level symbol");
      if (injectionApplies(key, overridesMap)) {
        env.put(name, entry.getValue());
      }
    }

    // Determine "native" bindings.
    // TODO(#11954): See above comment in createUninjectedBuildBzlEnv.
    Map<String, Object> nativeBindings = new HashMap<>();
    nativeBindings.putAll(uninjectedBuildBzlNativeBindings);
    for (Map.Entry<String, Object> entry : exportedRules.entrySet()) {
      String key = entry.getKey();
      String name = getKeySuffix(key);
      validateSymbolIsInjectable(name, nativeBindings.keySet(), ruleFunctions.keySet(), "rule");
      if (injectionApplies(key, overridesMap)) {
        nativeBindings.put(name, entry.getValue());
      }
    }

    env.put("native", createNativeModule(nativeBindings));
    return ImmutableMap.copyOf(env);
  }

  /**
   * Constructs an environment for a BUILD file based on the default environment, the map
   * corresponding to the {@code exported_rules} dict, and the value of {@code
   * --experimental_builtins_injection_override}.
   *
   * <p>Injected rule symbols must override an existing native rule of that name. Only rules may be
   * overridden in this manner, not generic built-ins such as {@code package} or {@code glob}.
   * Throws InjectionException if these conditions are not met.
   *
   * <p>Whether or not injection actually occurs for a given map key depends on its prefix (if any)
   * and the prefix of its appearance (if it appears at all) in the override list; see the
   * documentation for {@code --experimental_builtins_injection_override}. Non-injected symbols must
   * still obey the above constraints.
   */
  public ImmutableMap<String, Object> createBuildEnvUsingInjection(
      Map<String, Object> exportedRules, List<String> overridesList) throws InjectionException {
    Map<String, Boolean> overridesMap = parseInjectionOverridesList(overridesList);

    HashMap<String, Object> env = new HashMap<>(uninjectedBuildEnv);
    for (Map.Entry<String, Object> entry : exportedRules.entrySet()) {
      String key = entry.getKey();
      String name = getKeySuffix(key);
      validateSymbolIsInjectable(
          name,
          Sets.union(env.keySet(), Starlark.UNIVERSE.keySet()),
          ruleFunctions.keySet(),
          "rule");
      if (injectionApplies(key, overridesMap)) {
        env.put(name, entry.getValue());
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
