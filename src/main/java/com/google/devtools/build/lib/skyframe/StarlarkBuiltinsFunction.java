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

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.syntax.Dict;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Location;
import com.google.devtools.build.lib.syntax.Module;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import javax.annotation.Nullable;

// TODO(brandjon): Determine places where we need to teach Skyframe about this Skyfunction. Look for
// special treatment of BzlLoadFunction or ASTFileLookupFunction in existing code.

// TODO(brandjon): Add support to StarlarkModuleCycleReporter to pretty-print cycles involving
// @builtins. Blocked on us actually loading files from @builtins.

/**
 * A Skyframe function that evaluates the {@code @builtins} pseudo-repository and reports the values
 * exported by {@code @builtins//:exports.bzl}.
 *
 * <p>This function has a trivial key, so there can only be one value in the build at a time. It has
 * a single dependency, on the result of evaluating the exports.bzl file to a {@link BzlLoadValue}.
 *
 * <p>See also the design doc:
 * https://docs.google.com/document/d/1GW7UVo1s9X0cti9OMgT3ga5ozKYUWLPk9k8c4-34rC4/edit
 */
public class StarlarkBuiltinsFunction implements SkyFunction {

  /**
   * The label where {@code @builtins} symbols are exported from. (This is never conflated with any
   * actual repository named "{@code @builtins}" because it is only accessed through a special
   * SkyKey.
   */
  private static final Label EXPORTS_ENTRYPOINT =
      Label.parseAbsoluteUnchecked("@builtins//:exports.bzl");

  /** Same as above, as a {@link Location} for errors. */
  private static final Location EXPORTS_ENTRYPOINT_LOC =
      new Location(EXPORTS_ENTRYPOINT.getCanonicalForm(), /*line=*/ 0, /*column=*/ 0);

  StarlarkBuiltinsFunction() {}

  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws StarlarkBuiltinsFunctionException, InterruptedException {
    // skyKey is a singleton, unused.

    // TODO(brandjon): Replace by @builtins//:exports once BzlLoadFunction can resolve the @builtins
    // namespace.
    SkyKey exportsKey =
        BzlLoadValue.keyForBuild(
            Label.parseAbsoluteUnchecked("//tools/builtins_staging:exports.bzl"));
    BzlLoadValue exportsValue = (BzlLoadValue) env.getValue(exportsKey);
    if (exportsValue == null) {
      return null;
    }
    byte[] transitiveDigest = exportsValue.getTransitiveDigest();
    Module module = exportsValue.getModule();

    try {
      ImmutableMap<String, Object> exportedToplevels = getDict(module, "exported_toplevels");
      ImmutableMap<String, Object> exportedRules = getDict(module, "exported_rules");
      ImmutableMap<String, Object> exportedToJava = getDict(module, "exported_to_java");
      return new StarlarkBuiltinsValue(
          exportedToplevels, exportedRules, exportedToJava, transitiveDigest);
    } catch (EvalException ex) {
      ex.ensureLocation(EXPORTS_ENTRYPOINT_LOC);
      throw new StarlarkBuiltinsFunctionException(ex);
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
    Object value = module.get(dictName);
    if (value == null) {
      throw new EvalException(
          /*location=*/ null, String.format("expected a '%s' dictionary to be defined", dictName));
    }
    return ImmutableMap.copyOf(Dict.cast(value, String.class, Object.class, dictName + " dict"));
  }

  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }

  /** The exception type thrown by {@link StarlarkBuiltinsFunction}. */
  static final class StarlarkBuiltinsFunctionException extends SkyFunctionException {

    private StarlarkBuiltinsFunctionException(Exception cause) {
      super(cause, Transience.PERSISTENT);
    }
  }
}
