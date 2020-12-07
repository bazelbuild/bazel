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
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.packages.PackageFactory.InjectionException;
import com.google.devtools.build.lib.skyframe.BzlLoadFunction.BzlLoadFailedException;
import com.google.devtools.build.skyframe.RecordingSkyFunctionEnvironment;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import javax.annotation.Nullable;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Module;
import net.starlark.java.eval.Starlark;

// TODO(#11437): Update the design doc to change `@builtins` -> `@_builtins`.

// TODO(#11437): Add support to StarlarkModuleCycleReporter to pretty-print cycles involving
// @_builtins. Blocked on us actually loading files from @_builtins.

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
 * values exported by {@link #EXPORTS_ENTRYPOINT}.
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
   * The label where {@code @_builtins} symbols are exported from. (Note that this is never
   * conflated with an actual repository named "{@code @_builtins}" because 1) it is only ever
   * accessed through a special SkyKey, and 2) we disallow the user from defining a repo named
   * {@code @_builtins} to avoid confusion.)
   */
  static final Label EXPORTS_ENTRYPOINT =
      Label.parseAbsoluteUnchecked("@_builtins//:exports.bzl"); // unused

  /**
   * Key for loading exports.bzl. Note that {@code keyForBuiltins} (as opposed to {@code
   * keyForBuild}) ensures we can resolve {@code @_builtins}, which is otherwise inaccessible. It
   * also prevents us from cyclically requesting StarlarkBuiltinsFunction again to evaluate
   * exports.bzl.
   */
  static final BzlLoadValue.Key EXPORTS_ENTRYPOINT_KEY =
      BzlLoadValue.keyForBuiltins(EXPORTS_ENTRYPOINT);

  // Used to obtain the injected environment.
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
          packageFactory.createBuildBzlEnvUsingInjection(exportedToplevels, exportedRules);
      return new StarlarkBuiltinsValue(predeclared, exportedToJava, transitiveDigest);
    } catch (EvalException | InjectionException ex) {
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
