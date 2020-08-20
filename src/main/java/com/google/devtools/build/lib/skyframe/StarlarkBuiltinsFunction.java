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

package com.google.devtools.build.lib.skyframe;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.InconsistentFilesystemException;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.packages.StructProvider;
import com.google.devtools.build.lib.skyframe.BzlLoadFunction.BzlLoadFailedException;
import com.google.devtools.build.lib.syntax.Dict;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Module;
import com.google.devtools.build.lib.syntax.Starlark;
import com.google.devtools.build.skyframe.RecordingSkyFunctionEnvironment;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.LinkedHashMap;
import java.util.Map;
import javax.annotation.Nullable;

// TODO(#11437): Teach ASTFileLookup about builtins keys, then have them modify their env the same
// way as BzlLoadFunction. This will   allow us to define the _internal symbol for @builtins.

// TODO(#11437): Determine places where we need to teach Skyframe about this Skyfunction. Look for
// special treatment of BzlLoadFunction or ASTFileLookupFunction in existing code.

// TODO(#11437): Add support to StarlarkModuleCycleReporter to pretty-print cycles involving
// @builtins. Blocked on us actually loading files from @builtins.

// TODO(#11437): Add tombstone feature: If a native symbol is a tombstone object, this signals to
// StarlarkBuiltinsFunction that the corresponding symbol *must* be defined by @builtins.
// Furthermore, if exports.bzl also says the symbol is a tombstone, any attempt to use it results
// in failure, as if the symbol doesn't exist at all (or with a user-friendly error message saying
// to migrate by adding a load()). Combine tombstones with reading the current incompatible flags
// within @builtins for awesomeness.

// TODO(#11437): Currently, BUILD-loaded .bzls and WORKSPACE-loaded .bzls have the same initial
// static environment. Therefore, we should apply injection of top-level symbols to both
// environments, not just BUILD .bzls. Likewise for builtins that are shared by BUILD and WORKSPACE
// files, and for when the dynamic `native` values of the two .bzl dialects are unified. This can be
// facilitated by 1) making PackageFactory better understand the similarities between BUILD and
// WORKSPACE, e.g. by refactoring things like WorkspaceFactory#createWorkspaceFunctions into
// PackageFactory; and 2) making PackageFactory responsible for performing the actual injection
// (given the override mappings from exports.bzl) and returning the modified environments. Then any
// refactoring to unify BUILD with WORKPSACE and BUILD-bzl with WORKSPACE-bzl can proceed in
// PackageFactory without regard to this file.

/**
 * A Skyframe function that evaluates the {@code @builtins} pseudo-repository and reports the values
 * exported by {@link #EXPORTS_ENTRYPOINT}.
 *
 * <p>The process of "builtins injection" refers to evaluating this Skyfunction and applying its
 * result to {@link BzlLoadFunction}'s computation. See also the <a
 * href="https://docs.google.com/document/d/1GW7UVo1s9X0cti9OMgT3ga5ozKYUWLPk9k8c4-34rC4">design
 * doc</a>:
 *
 * <p>This function has a trivial key, so there can only be one value in the build at a time. It has
 * a single dependency, on the result of evaluating the exports.bzl file to a {@link BzlLoadValue}.
 */
public class StarlarkBuiltinsFunction implements SkyFunction {

  /**
   * The label where {@code @builtins} symbols are exported from. (This is never conflated with any
   * actual repository named "{@code @builtins}" because it is only accessed through a special
   * SkyKey.
   */
  private static final Label EXPORTS_ENTRYPOINT =
      Label.parseAbsoluteUnchecked("@builtins//:exports.bzl"); // unused

  /**
   * Key for loading exports.bzl. {@code keyForBuiltins} (as opposed to {@code keyForBuild} ensures
   * that 1) we can resolve the {@code @builtins} name appropriately, and 2) loading it does not
   * trigger a cyclic call back into {@code StarlarkBuiltinsFunction}.
   */
  private static final BzlLoadValue.Key EXPORTS_ENTRYPOINT_KEY =
      BzlLoadValue.keyForBuiltins(
          // TODO(#11437): Replace by EXPORTS_ENTRYPOINT once BzlLoadFunction can resolve the
          // @builtins namespace.
          Label.parseAbsoluteUnchecked("//tools/builtins_staging:exports.bzl"));

  // Used to obtain top-level environment and "native" object.
  private final PackageFactory packageFactory;

  public StarlarkBuiltinsFunction(PackageFactory packageFactory) {
    this.packageFactory = packageFactory;
  }

  @Override
  @Nullable
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws StarlarkBuiltinsFunctionException, InterruptedException {
    // skyKey is a singleton, unused.
    try {
      return computeInternal(
          env, packageFactory, /*inliningState=*/ null, /*bzlLoadFunction=*/ null);
    } catch (BuiltinsFailedException e) {
      throw new StarlarkBuiltinsFunctionException(e);
    }
  }

  /**
   * Computes this Skyfunction under inlining of {@link BzlLoadFunction}, forwarding the given
   * inlining state.
   *
   * <p>The given Skyframe environment must be a {@link RecordingSkyFunctionEnvironment}. It is
   * unwrapped before calling {@link BzlLoadFunction}'s inlining code path.
   *
   * <p>Returns null on Skyframe restart or error.
   */
  @Nullable
  public static StarlarkBuiltinsValue computeInline(
      StarlarkBuiltinsValue.Key key, // singleton value, unused
      Environment env,
      BzlLoadFunction.InliningState inliningState,
      PackageFactory packageFactory,
      BzlLoadFunction bzlLoadFunction)
      throws BuiltinsFailedException, InterruptedException {
    Preconditions.checkState(
        env instanceof RecordingSkyFunctionEnvironment,
        "Expected to be recording dep requests when inlining StarlarkBuiltinsFunction");
    // We don't need any direct Skyframe calls because we only depend on the exports.bzl file. If we
    // did make Skyframe calls, they'd use the original recording environment (env), so that they're
    // properly registered in the CachedBzlLoadData object of the .bzl that is requesting the
    // builtins.
    //
    // We unwrap the environment before calling computeInternal() (and indirectly,
    // BzlLoadFunction#computeInternal). Any Skyframe deps needed to evaluate exports.bzl and its
    // transitive deps will be reported by their CachedBzlLoadData objects.
    //
    // TODO(#11437): Update these comments for when we can also inline builtins computations for
    // BUILD files.
    Environment strippedEnv = ((RecordingSkyFunctionEnvironment) env).getDelegate();
    return computeInternal(strippedEnv, packageFactory, inliningState, bzlLoadFunction);
  }

  // bzlLoadFunction and inliningState are non-null iff using inlining code path.
  private static StarlarkBuiltinsValue computeInternal(
      Environment env,
      PackageFactory packageFactory,
      @Nullable BzlLoadFunction.InliningState inliningState,
      @Nullable BzlLoadFunction bzlLoadFunction)
      throws BuiltinsFailedException, InterruptedException {

    // Load exports.bzl. If we were requested using inlining, make sure to inline the call back into
    // BzlLoadFunction.
    BzlLoadValue exportsValue;
    try {
      if (inliningState == null) {
        exportsValue =
            (BzlLoadValue)
                env.getValueOrThrow(EXPORTS_ENTRYPOINT_KEY, BzlLoadFailedException.class);
      } else {
        exportsValue = bzlLoadFunction.computeInline(EXPORTS_ENTRYPOINT_KEY, env, inliningState);
      }
    } catch (BzlLoadFailedException ex) {
      throw BuiltinsFailedException.errorEvaluatingBuiltinsBzls(ex);
    } catch (InconsistentFilesystemException ex) {
      throw BuiltinsFailedException.errorEvaluatingBuiltinsBzls(ex);
    }
    if (exportsValue == null) {
      return null;
    }

    // Apply declarations of exports.bzl to the native predeclared symbols.
    byte[] transitiveDigest = exportsValue.getTransitiveDigest();
    Module module = exportsValue.getModule();
    try {
      ImmutableMap<String, Object> exportedToplevels = getDict(module, "exported_toplevels");
      ImmutableMap<String, Object> exportedRules = getDict(module, "exported_rules");
      ImmutableMap<String, Object> exportedToJava = getDict(module, "exported_to_java");
      ImmutableMap<String, Object> predeclared =
          createPredeclaredForBuildBzlUsingInjection(
              packageFactory, exportedToplevels, exportedRules);
      return new StarlarkBuiltinsValue(predeclared, exportedToJava, transitiveDigest);
    } catch (EvalException ex) {
      throw BuiltinsFailedException.errorApplyingExports(ex);
    }
  }

  /**
   * Returns the set of predeclared symbols to initialize a Starlark {@link Module} with, for
   * evaluating .bzls loaded from a BUILD file.
   */
  private static ImmutableMap<String, Object> createPredeclaredForBuildBzlUsingInjection(
      PackageFactory packageFactory,
      ImmutableMap<String, Object> exportedToplevels,
      ImmutableMap<String, Object> exportedRules) {
    // TODO(#11437): Validate that all symbols supplied to exportedToplevels and exportedRules are
    // existing rule-logic symbols.

    // It's probably not necessary to preserve order, but let's do it just in case.
    Map<String, Object> predeclared = new LinkedHashMap<>();

    // Determine the top-level bindings.
    predeclared.putAll(packageFactory.getRuleClassProvider().getEnvironment());
    predeclared.putAll(exportedToplevels);
    // TODO(#11437): We *should* be able to uncomment the following line, but the native module is
    // added prematurely (without its rule-logic fields) and overridden unconditionally. Fix this
    // once ASTFileLookupFunction takes in the set of predeclared bindings (currently it directly
    // // checks getEnvironment()).
    // Preconditions.checkState(!predeclared.containsKey("native"));

    // Determine the "native" module.
    // TODO(bazel-team): Use the same "native" object for both BUILD- and WORKSPACE-loaded .bzls,
    // and just have it be a dynamic error to call the wrong thing at the wrong time. This is a
    // breaking change.
    Map<String, Object> nativeEntries = new LinkedHashMap<>();
    for (Map.Entry<String, Object> entry :
        packageFactory.getNativeModuleBindingsForBuild().entrySet()) {
      String symbolName = entry.getKey();
      Object replacementSymbol = exportedRules.get(symbolName);
      if (replacementSymbol != null) {
        nativeEntries.put(symbolName, replacementSymbol);
      } else {
        nativeEntries.put(symbolName, entry.getValue());
      }
    }
    predeclared.put("native", createNativeModule(nativeEntries));

    return ImmutableMap.copyOf(predeclared);
  }

  /** Returns a set of predeclared symbols without considering injected builtins. */
  // TODO(#11437): Delete once injection cannot be disabled.
  static ImmutableMap<String, Object> createPredeclaredForBuildBzlWithoutInjection(
      PackageFactory packageFactory) {
    return createPredeclaredForBuildBzlUsingInjection(
        packageFactory,
        /*exportedToplevels=*/ ImmutableMap.of(),
        /*exportedRules=*/ ImmutableMap.of());
  }

  /**
   * Returns the set of predeclared symbols to initialize a Starlark {@link Module} with, for
   * evaluating .bzls loaded from a WORKSPACE file.
   */
  static ImmutableMap<String, Object> createPredeclaredForWorkspaceBzl(
      PackageFactory packageFactory) {
    // Preserve order, just in case.
    Map<String, Object> predeclared = new LinkedHashMap<>();
    predeclared.putAll(packageFactory.getRuleClassProvider().getEnvironment());
    Object nativeModule = createNativeModule(packageFactory.getNativeModuleBindingsForWorkspace());
    // TODO(#11437): Assert not already present; see createPreclaredsForBuildBzl.
    predeclared.put("native", nativeModule);
    return ImmutableMap.copyOf(predeclared);
  }

  /**
   * Returns the set of predeclared symbols to initialize a Starlark {@link Module} with, for
   * evaluating .bzls loaded from the {@code @builtins} pseudo-repository.
   */
  // TODO(#11437): create the _internal name, prohibit other rule logic names. Take in a
  // PackageFactory.
  static ImmutableMap<String, Object> createPredeclaredForBuiltinsBzl(
      PackageFactory packageFactory) {
    return packageFactory.getRuleClassProvider().getEnvironment();
  }

  private static Object createNativeModule(Map<String, Object> bindings) {
    return StructProvider.STRUCT.create(bindings, "no native function or rule '%s'");
  }

  /**
   * Attempts to retrieve the string-keyed dict named {@code dictName} from the given {@code
   * module}.
   *
   * @return a copy of the dict mappings on success
   * @throws EvalException if the symbol isn't present or is not a dict whose keys are all strings
   */
  @Nullable
  private static ImmutableMap<String, Object> getDict(Module module, String dictName)
      throws EvalException {
    Object value = module.get(dictName);
    if (value == null) {
      throw Starlark.errorf("expected a '%s' dictionary to be defined", dictName);
    }
    return ImmutableMap.copyOf(Dict.cast(value, String.class, Object.class, dictName + " dict"));
  }

  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }

  /**
   * An exception that occurs while trying to determine the injected builtins.
   *
   * <p>This exception type typically wraps a {@link BzlLoadFailedException} and is wrapped by a
   * {@link BzlLoadFailedException} in turn.
   */
  static final class BuiltinsFailedException extends Exception {

    private final Transience transience;

    private BuiltinsFailedException(String errorMessage, Exception cause, Transience transience) {
      super(errorMessage, cause);
      this.transience = transience;
    }

    Transience getTransience() {
      return transience;
    }

    static BuiltinsFailedException errorEvaluatingBuiltinsBzls(BzlLoadFailedException cause) {
      return errorEvaluatingBuiltinsBzls(cause, cause.getTransience());
    }

    static BuiltinsFailedException errorEvaluatingBuiltinsBzls(
        InconsistentFilesystemException cause) {
      return errorEvaluatingBuiltinsBzls(cause, Transience.PERSISTENT);
    }

    static BuiltinsFailedException errorEvaluatingBuiltinsBzls(
        Exception cause, Transience transience) {
      return new BuiltinsFailedException(
          String.format("Failed to load builtins sources: %s", cause.getMessage()),
          cause,
          transience);
    }

    static BuiltinsFailedException errorApplyingExports(EvalException cause) {
      return new BuiltinsFailedException(
          String.format("Failed to apply declared builtins: %s", cause.getMessage()),
          cause,
          Transience.PERSISTENT);
    }
  }

  /** The exception type thrown by {@link StarlarkBuiltinsFunction}. */
  static final class StarlarkBuiltinsFunctionException extends SkyFunctionException {

    private StarlarkBuiltinsFunctionException(BuiltinsFailedException cause) {
      super(cause, cause.transience);
    }
  }
}
