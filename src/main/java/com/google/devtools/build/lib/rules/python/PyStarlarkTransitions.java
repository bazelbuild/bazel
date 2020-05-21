// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.python;

import static com.google.devtools.build.lib.packages.Type.STRING;

import com.google.devtools.build.lib.analysis.config.transitions.ConfigurationTransition;
import com.google.devtools.build.lib.analysis.config.transitions.NoTransition;
import com.google.devtools.build.lib.analysis.config.transitions.TransitionFactory;
import com.google.devtools.build.lib.packages.AttributeTransitionData;
import com.google.devtools.build.lib.skylarkbuildapi.python.PyStarlarkTransitionsApi;
import com.google.devtools.build.lib.syntax.Printer;
import com.google.devtools.build.lib.syntax.StarlarkValue;

/**
 * Exposes a native transition to Starlark for changing the Python version.
 *
 * <p>This is not intended to be a stable API. It is only available when {@code
 * --experimental_google_legacy_api} is enabled. For production code, users should generally use a
 * Starlark-defined transition (SBC) based on {@code //command_line_option:python_version}.
 *
 * <p>The only reason to expose the native transition here is that it performs better;
 * Starlark-based transitions cannot yet collapse equivalent configurations together due to
 * differences in the output root. Most builds should not need this level of optimization. For the
 * ones that do, this transition can unblock PY2 -> PY3 migration.
 *
 * <p>See also internal bugs b/137574156 and b/140565289.
 */
public final class PyStarlarkTransitions implements PyStarlarkTransitionsApi {

  /** Singleton instance of {@link PyStarlarkTransitions}. */
  public static final PyStarlarkTransitions INSTANCE = new PyStarlarkTransitions();

  @Override
  public StarlarkValue getTransition() {
    return StarlarkVersionTransition.INSTANCE;
  }

  private static class StarlarkVersionTransition
      implements TransitionFactory<AttributeTransitionData>, StarlarkValue {

    private static final StarlarkVersionTransition INSTANCE = new StarlarkVersionTransition();

    @Override
    public ConfigurationTransition create(AttributeTransitionData data) {
      if (!data.attributes().has("python_version")) {
        return NoTransition.INSTANCE;
      }
      String version = data.attributes().get("python_version", STRING);
      if ("DEFAULT".equals(version)) {
        return PythonVersionTransition.toDefault();
      } else if (PythonVersion.TARGET_STRINGS.contains(version)) {
        return PythonVersionTransition.toConstant(PythonVersion.parseTargetValue(version));
      } else {
        return NoTransition.INSTANCE;
      }
    }

    @Override
    public void repr(Printer printer) {
      printer.append("<py_transitions.cfg>");
    }
  }
}
