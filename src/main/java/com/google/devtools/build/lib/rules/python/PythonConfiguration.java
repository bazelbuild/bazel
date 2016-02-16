// Copyright 2014 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;

/**
 * The configuration fragment containing information about the various pieces of infrastructure
 * needed to run Python compilations.
 */
@Immutable
public class PythonConfiguration extends BuildConfiguration.Fragment {
  private final boolean ignorePythonVersionAttribute;
  private final PythonVersion defaultPythonVersion;

  PythonConfiguration(PythonVersion pythonVersion, boolean ignorePythonVersionAttribute) {
    this.ignorePythonVersionAttribute = ignorePythonVersionAttribute;

    this.defaultPythonVersion = pythonVersion;
  }

  /**
   * Returns the Python version (PY2 or PY3) this configuration uses.
   */
  public PythonVersion getDefaultPythonVersion() {
    return defaultPythonVersion;
  }

  /**
   * Returns the Python version to use. Command-line flag --force_python overrides
   * the rule default, given as argument.
   */
  public PythonVersion getPythonVersion(PythonVersion attributeVersion) {
    return ignorePythonVersionAttribute || attributeVersion == null
        ? defaultPythonVersion
        : attributeVersion;
  }

  @Override
  public String getOutputDirectoryName() {
    return (defaultPythonVersion == PythonVersion.PY3) ? "py3" : null;
  }
}

