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

package com.google.devtools.build.lib.rules.python;

import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.transitions.PatchTransition;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;

/**
 * A configuration transition that sets the Python version by setting {@link
 * PythonOptions#forcePython}.
 */
@AutoCodec
public class PythonVersionTransition implements PatchTransition {
  private final PythonVersion defaultVersion;

  /**
   * Creates a new transition that sets the given version if not already specified by
   * {@link PythonOptions#forcePython}.
   */
  PythonVersionTransition(PythonVersion defaultVersion) {
    this.defaultVersion = defaultVersion;
  }

  @Override
   public BuildOptions apply(BuildOptions options) {
    PythonOptions pyOptions = options.get(PythonOptions.class);
    // The current Python version is either explicitly set by --force_python or a
    // build-wide default.
    PythonVersion currentVersion = pyOptions.getPythonVersion();
    // The new Python version is either explicitly set by --force_python or this transition's
    // default.
    PythonVersion newVersion = pyOptions.getPythonVersion(defaultVersion);
    if (currentVersion == newVersion) {
      return options;
    }

    // forcePython must be one of PY2 or PY3 because these are the only values Blaze's output
    // directories can safely distinguish. In other words, a configuration with forcePython=PY2
    // would have the same output directory prefix as another with forcePython=PY2AND3, which is a
    // major correctness failure.
    //
    // Even though this transition doesn't enforce the above, it only gets called on
    // "default_python_version" attribute values, which happen to honor this. Proper enforcement is
    // done in PythonConfiguration#getOutputDirectoryName.
    BuildOptions newOptions = options.clone();
    newOptions.get(PythonOptions.class).forcePython = newVersion;
    return newOptions;
  }
}
