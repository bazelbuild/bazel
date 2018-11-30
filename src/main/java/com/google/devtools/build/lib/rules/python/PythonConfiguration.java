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

import com.google.common.base.Joiner;
import com.google.common.base.Verify;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.common.options.TriState;
import java.util.Arrays;
import java.util.List;

/**
 * The configuration fragment containing information about the various pieces of infrastructure
 * needed to run Python compilations.
 */
@Immutable
@SkylarkModule(
    name = "py",
    doc = "A configuration fragment for SWIG.",
    category = SkylarkModuleCategory.CONFIGURATION_FRAGMENT)
public class PythonConfiguration extends BuildConfiguration.Fragment {
  private final boolean ignorePythonVersionAttribute;
  private final PythonVersion defaultPythonVersion;
  private final TriState buildPythonZip;
  private final boolean buildTransitiveRunfilesTrees;

  PythonConfiguration(
      PythonVersion defaultPythonVersion,
      boolean ignorePythonVersionAttribute,
      TriState buildPythonZip,
      boolean buildTransitiveRunfilesTrees) {
    this.ignorePythonVersionAttribute = ignorePythonVersionAttribute;
    this.defaultPythonVersion = defaultPythonVersion;
    this.buildPythonZip = buildPythonZip;
    this.buildTransitiveRunfilesTrees = buildTransitiveRunfilesTrees;
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
    List<PythonVersion> allowedVersions = Arrays.asList(PythonVersion.getTargetValues());
    Verify.verify(
        allowedVersions.size() == 2, // If allowedVersions.size() == 1, we don't need this method.
        ">2 possible defaultPythonVersion values makes output directory clashes possible");
    // Skip this check if --force_python is set. That's because reportInvalidOptions reports
    // bad --force_python settings with a clearer user error (and Bazel's configuration
    // initialization logic calls reportInvalidOptions after this method).
    if (!ignorePythonVersionAttribute && !allowedVersions.contains(defaultPythonVersion)) {
      throw new IllegalStateException(
          String.format("defaultPythonVersion=%s not allowed: must be in %s to prevent output "
              + "directory clashes", defaultPythonVersion, Joiner.on(", ").join(allowedVersions)));
    }
    return (defaultPythonVersion == PythonVersion.PY3) ? "py3" : null;
  }

  @Override
  public void reportInvalidOptions(EventHandler reporter, BuildOptions buildOptions) {
    PythonOptions pythonOptions = buildOptions.get(PythonOptions.class);
    if (pythonOptions.forcePython != null
        && pythonOptions.forcePython != PythonVersion.PY2
        && pythonOptions.forcePython != PythonVersion.PY3) {
      reporter.handle(Event.error("'--force_python' argument must be 'PY2' or 'PY3'"));
    }
  }

  /** Returns whether to build the executable zip file for Python binaries. */
  public boolean buildPythonZip() {
    switch (buildPythonZip) {
      case YES:
        return true;
      case NO:
        return false;
      default:
        return OS.getCurrent() == OS.WINDOWS;
    }
  }

  /**
   * Return whether to build the runfiles trees of py_binary targets that appear in the transitive
   * data runfiles of another binary.
   */
  public boolean buildTransitiveRunfilesTrees() {
    return buildTransitiveRunfilesTrees;
  }
}
