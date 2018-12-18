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

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.transitions.PatchTransition;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;

/** A configuration transition that sets the Python version. */
@AutoCodec
public class PythonVersionTransition implements PatchTransition {

  private final PythonVersion version;

  /**
   * Creates a new transition that sets the version to the given value, if transitioning is allowed.
   *
   * <p>See {@link PythonOptions#canTransitionPythonVersion} for information on when transitioning
   * is allowed.
   *
   * <p>{@code version} must be a target version (either {@code PY2} or {@code PY3}), or else null,
   * which means use the default value ({@link PythonVersion#DEFAULT_TARGET_VALUE}).
   *
   * @throws IllegalArgumentException if {@code version} is non-null and not {@code PY2} or {@code
   *     PY3}
   */
  PythonVersionTransition(PythonVersion version) {
    if (version == null) {
      version = PythonVersion.DEFAULT_TARGET_VALUE;
    }
    Preconditions.checkArgument(version.isTargetValue());
    this.version = version;
  }

  @Override
  public BuildOptions patch(BuildOptions options) {
    PythonOptions opts = options.get(PythonOptions.class);
    if (!opts.canTransitionPythonVersion(version)) {
      return options;
    }
    BuildOptions newOptions = options.clone();
    PythonOptions newOpts = newOptions.get(PythonOptions.class);
    newOpts.setPythonVersion(version);
    return newOptions;
  }
}
