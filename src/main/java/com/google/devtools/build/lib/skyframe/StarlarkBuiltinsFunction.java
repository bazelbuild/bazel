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

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.collect.ImmutableList.toImmutableList;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.BazelStarlarkEnvironment;
import com.google.devtools.build.lib.packages.BazelStarlarkEnvironment.InjectionException;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.skyframe.BzlLoadFunction.InlineCacheManager;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.skyframe.RecordingSkyFunctionEnvironment;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.LinkedHashMap;
import java.util.List;
import javax.annotation.Nullable;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Module;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.syntax.Location;

// TODO(#11437): Update the design doc to change `@builtins` -> `@_builtins`.

// TODO(#11437): Add support to BzlLoadCycleReporter to pretty-print cycles involving
// @_builtins.

// TODO(#11437): Add tombstone feature: If a native symbol is a tombstone object, this signals to
// StarlarkBuiltinsFunction that the corresponding symbol *must* be defined by @_builtins.
// Furthermore, if exports.bzl also says the symbol is a tombstone, any attempt to use it results
// in failure, as if the symbol doesn't exist at all (or with a user-friendly error message saying
// to migrate by adding a load()). Combine tombstones with reading the current incompatible flags
// within @_builtins for awesomeness.

// TODO(#11437, #11954, #11983): To the extent that BUILD-loaded .bzls and WORKSPACE-loaded .bzls
// have the same environment, builtins injection should apply to both of them, not just to
// BUILD-loaded .bzls.

/**
 * A Skyframe function that evaluates the {@code @_builtins} pseudo-repository and reports the
 * values exported by {@link #EXPORTS_ENTRYPOINT}. The {@code @_builtins} pseudo-repository shares a
 * repo mapping with the {@code @bazel_tools} repository.
 *
 * <p>The process of "builtins injection" refers to evaluating this Skyfunction and applying its
 * result to {@link BzlLoadFunction}'s computation. See also the <a
 * href="https://docs.google.com/document/d/1GW7UVo1s9X0cti9OMgT3ga5ozKYUWLPk9k8c4-34rC4">design
 * doc</a>:
 *
 * <p>This function has a trivial key, so there can only be one value in the build at a time. It has
 * a single dependency, on the result of evaluating the exports.bzl file to a {@link BzlLoadValue}.
 *
 * <p>This function supports a special "inlining" mode, similar to {@link BzlLoadFunction} (see that
 * class's javadoc and code comments). Whenever we inline {@link BzlLoadFunction} we also inline
 * {@link StarlarkBuiltinsFunction} (and {@link StarlarkBuiltinsFunction}'s calls to {@link
 * BzlLoadFunction} are then themselves inlined!). Similar to {@link BzlLoadFunction}'s inlining, we
 * cache the result of this computation, and this caching is managed by {@link
 * BzlLoadFunction.InlineCacheManager}. But since there's only a single {@link
 * StarlarkBuiltinsValue} node and we don't need to worry about that node's value changing at future
 * invocations or subsequent versions (see {@link InlineCacheManager#reset} for why), our caching
 * strategy is much simpler and we don't need to bother inlining deps of the Skyframe subgraph.
 */
public class StarlarkBuiltinsFunction implements SkyFunction {

  /**
   * The label where {@code @_builtins} symbols are exported from. (Note that this is never
   * conflated with an actual repository named "{@code @_builtins}" because 1) it is only ever
   * accessed through a special SkyKey, and 2) we disallow the user from defining a repo named
   * {@code @_builtins} to avoid confusion.)
   */
  static final Label EXPORTS_ENTRYPOINT =
      Label.parseCanonicalUnchecked("@_builtins//:exports.bzl"); // unused

  /**
   * Key for loading exports.bzl. Note that {@code keyForBuiltins} (as opposed to {@code
   * keyForBuild}) ensures we can resolve {@code @_builtins}, which is otherwise inaccessible. It
   * also prevents us from cyclically requesting StarlarkBuiltinsFunction again to evaluate
   * exports.bzl.
   */
  static final BzlLoadValue.Key EXPORTS_ENTRYPOINT_KEY =
      BzlLoadValue.keyForBuiltins(EXPORTS_ENTRYPOINT);

  // Used to obtain the injected environment.
  private final BazelStarlarkEnvironment bazelStarlarkEnvironment;

  public StarlarkBuiltinsFunction(BazelStarlarkEnvironment bazelStarlarkEnvironment) {
    this.bazelStarlarkEnvironment = bazelStarlarkEnvironment;
  }

  @Override
  @Nullable
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws StarlarkBuiltinsFunctionException, InterruptedException {
    try {
      return computeInternal(
          env,
          bazelStarlarkEnvironment,
          /* inliningState= */ null,
          /* bzlLoadFunction= */ null,
          ((StarlarkBuiltinsValue.Key) skyKey.argument()).isWithAutoloads());
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
      BzlLoadFunction.InliningState inliningState,
      BazelStarlarkEnvironment bazelStarlarkEnvironment,
      BzlLoadFunction bzlLoadFunction)
      throws BuiltinsFailedException, InterruptedException {
    checkNotNull(bzlLoadFunction.inlineCacheManager);
    StarlarkBuiltinsValue cachedBuiltins = bzlLoadFunction.inlineCacheManager.builtinsRef.get();
    if (cachedBuiltins != null) {
      // See the comment in InlineCacheManager#reset for why it's sound to not inline deps of the
      // entire subgraph here.
      return cachedBuiltins;
    }

    // See BzlLoadFunction#computeInline and BzlLoadFunction.InliningState for an explanation of the
    // inlining mechanism and its invariants. For our purposes, the Skyframe environment to use
    // comes from inliningState.
    StarlarkBuiltinsValue computedBuiltins =
        computeInternal(
            inliningState.getEnvironment(),
            bazelStarlarkEnvironment,
            inliningState,
            bzlLoadFunction,
            key.isWithAutoloads());
    if (computedBuiltins == null) {
      return null;
    }
    // There's a benign race where multiple threads may try to compute-and-cache the single builtins
    // value. Ensure the value computed by winner of that race gets used by everyone.
    bzlLoadFunction.inlineCacheManager.builtinsRef.compareAndSet(null, computedBuiltins);
    return bzlLoadFunction.inlineCacheManager.builtinsRef.get();
  }

  // bzlLoadFunction and inliningState are non-null iff using inlining code path.
  @Nullable
  private static StarlarkBuiltinsValue computeInternal(
      Environment env,
      BazelStarlarkEnvironment bazelStarlarkEnvironment,
      @Nullable BzlLoadFunction.InliningState inliningState,
      @Nullable BzlLoadFunction bzlLoadFunction,
      boolean isWithAutoloads)
      throws BuiltinsFailedException, InterruptedException {
    StarlarkSemantics starlarkSemantics = PrecomputedValue.STARLARK_SEMANTICS.get(env);
    if (starlarkSemantics == null) {
      return null;
    }
    // Return the empty value if builtins injection is disabled.
    if (starlarkSemantics.get(BuildLanguageOptions.EXPERIMENTAL_BUILTINS_BZL_PATH).isEmpty()) {
      return StarlarkBuiltinsValue.createEmpty(starlarkSemantics);
    }

    // Load exports.bzl. If we were requested using inlining, make sure to inline the call back into
    // BzlLoadFunction.
    BzlLoadValue exportsValue;
    try {
      if (inliningState == null) {
        exportsValue =
            (BzlLoadValue)
                env.getValueOrThrow(EXPORTS_ENTRYPOINT_KEY, BzlLoadFailedException.class);
      } else {
        exportsValue = bzlLoadFunction.computeInline(EXPORTS_ENTRYPOINT_KEY, inliningState);
      }
    } catch (BzlLoadFailedException ex) {
      throw BuiltinsFailedException.errorEvaluatingBuiltinsBzls(ex);
    }
    if (exportsValue == null) {
      return null;
    }

    // Apply declarations of exports.bzl to the native predeclared symbols.
    byte[] transitiveDigest = exportsValue.getTransitiveDigest();
    Module module = exportsValue.getModule();
    ImmutableMap<String, Object> exportedToplevels;
    ImmutableMap<String, Object> exportedRules;
    ImmutableMap<String, Object> exportedToJava;
    try {
      exportedToplevels = getDict(module, "exported_toplevels");
      exportedRules = getDict(module, "exported_rules");
      exportedToJava = getDict(module, "exported_to_java");
    } catch (EvalException ex) {
      throw BuiltinsFailedException.errorApplyingExports(ex);
    }

    if (isWithAutoloads) {
      ImmutableList<String> symbolsToLoad =
          starlarkSemantics.get(BuildLanguageOptions.INCOMPATIBLE_LOAD_SYMBOLS_EXTERNALLY).stream()
              .filter(s -> !s.startsWith("-"))
              .collect(toImmutableList());
      ImmutableList<String> rulesToLoad =
          starlarkSemantics.get(BuildLanguageOptions.INCOMPATIBLE_LOAD_RULES_EXTERNALLY).stream()
              .filter(s -> !s.startsWith("-"))
              .collect(toImmutableList());

      // Inject loads for rules and symbols removed from Bazel
      // This could be optimized for only symbols that were used during compilation
      ImmutableList.Builder<BzlLoadValue.Key> loadKeysBuilder = ImmutableList.builder();
      ImmutableList.Builder<Pair<String, Location>> additionalLoads = ImmutableList.builder();

      for (String symbol : symbolsToLoad) {
        // Load with prelude key, to avoid cycles
        loadKeysBuilder.add(
            BzlLoadValue.keyForBuildAutoloads(
                Label.parseCanonicalUnchecked(
                    "@bazel_tools//tools/redirects_do_not_use:" + symbol + ".bzl")));
        additionalLoads.add(Pair.of(symbol, Location.BUILTIN));
      }
      for (String rule : rulesToLoad) {
        // Load with prelude key, to avoid cycles
        loadKeysBuilder.add(
            BzlLoadValue.keyForBuildAutoloads(
                Label.parseCanonicalUnchecked(
                    "@bazel_tools//tools/redirects_do_not_use:" + rule + ".bzl")));
        additionalLoads.add(Pair.of(rule, Location.BUILTIN));
      }

      List<BzlLoadValue> loadValues;
      try {
        if (inliningState == null) {
          loadValues =
              BzlLoadFunction.computeBzlLoadsWithSkyframe(
                  env, loadKeysBuilder.build(), additionalLoads.build());
        } else {
          loadValues =
              bzlLoadFunction.computeBzlLoadsWithInlining(
                  env, loadKeysBuilder.build(), additionalLoads.build(), inliningState);
        }
        if (loadValues == null) {
          return null;
        }
      } catch (BzlLoadFailedException ex) {
        throw new BuiltinsFailedException(
            String.format(
                "There's no load specification for toplevel symbol or rule set by"
                    + " --incompatible_load_symbols_externally or"
                    + " --incompatible_load_rules_externally. Most likely a typo."),
            ex,
            Transience.PERSISTENT);
      }

      Fingerprint fp = new Fingerprint();
      fp.addBytes(transitiveDigest);

      String workspaceWarning =
          starlarkSemantics.getBool(BuildLanguageOptions.ENABLE_BZLMOD)
              ? ""
              : " Most likely you need to upgrade the version of rules repository providing it your"
                  + " WORKSPACE file.";

      // Substitute-in loaded rules and symbols (that were removed from Bazel)
      LinkedHashMap<String, Object> exportedToplevelsBuilder =
          new LinkedHashMap<>(exportedToplevels);
      int i = 0;
      for (String symbol : symbolsToLoad) {
        BzlLoadValue v = loadValues.get(i++);
        if (v.getModule().getGlobal(symbol) == null) {
          throw new BuiltinsFailedException(
              String.format(
                  "The toplevel symbol '%s' set by --incompatible_load_symbols_externally couldn't"
                      + " be loaded.%s",
                  symbol, workspaceWarning),
              null,
              Transience.PERSISTENT);
        }
        exportedToplevelsBuilder.put(symbol, v.getModule().getGlobal(symbol));
        fp.addBytes(v.getTransitiveDigest());
      }
      exportedToplevels = ImmutableMap.copyOf(exportedToplevelsBuilder);

      LinkedHashMap<String, Object> exportedRulesBuilder = new LinkedHashMap<>(exportedRules);
      for (String rule : rulesToLoad) {
        BzlLoadValue v = loadValues.get(i++);
        if (v.getModule().getGlobal(rule) == null) {
          throw new BuiltinsFailedException(
              String.format(
                  "The rule '%s' et by --incompatible_load_rules_externally couldn't be loaded.%s",
                  rule, workspaceWarning),
              null,
              Transience.PERSISTENT);
        }
        exportedRulesBuilder.put(rule, v.getModule().getGlobal(rule));
        fp.addBytes(v.getTransitiveDigest());
      }
      exportedRules = ImmutableMap.copyOf(exportedRulesBuilder);

      transitiveDigest = fp.digestAndReset();
    }

    try {
      ImmutableMap<String, Object> predeclaredForBuildBzl =
          bazelStarlarkEnvironment.createBuildBzlEnvUsingInjection(
              exportedToplevels, exportedRules, starlarkSemantics);
      ImmutableMap<String, Object> predeclaredForWorkspaceBzl =
          bazelStarlarkEnvironment.createWorkspaceBzlEnvUsingInjection(
              exportedToplevels,
              exportedRules,
              starlarkSemantics.get(BuildLanguageOptions.EXPERIMENTAL_BUILTINS_INJECTION_OVERRIDE));
      ImmutableMap<String, Object> predeclaredForBuild =
          bazelStarlarkEnvironment.createBuildEnvUsingInjection(exportedRules, starlarkSemantics);
      return StarlarkBuiltinsValue.create(
          predeclaredForBuildBzl,
          predeclaredForWorkspaceBzl,
          predeclaredForBuild,
          exportedToJava,
          transitiveDigest,
          starlarkSemantics);
    } catch (InjectionException ex) {
      throw BuiltinsFailedException.errorApplyingExports(ex);
    }
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
    Object value = module.getGlobal(dictName);
    if (value == null) {
      throw Starlark.errorf("expected a '%s' dictionary to be defined", dictName);
    }
    return ImmutableMap.copyOf(Dict.cast(value, String.class, Object.class, dictName + " dict"));
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
        Exception cause, Transience transience) {
      return new BuiltinsFailedException(
          String.format("Failed to load builtins sources: %s", cause.getMessage()),
          cause,
          transience);
    }

    static BuiltinsFailedException errorApplyingExports(Exception cause) {
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
