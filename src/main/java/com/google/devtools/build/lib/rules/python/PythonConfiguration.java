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

import com.google.common.base.Ascii;
import com.google.common.base.Preconditions;
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

/**
 * The configuration fragment containing information about the various pieces of infrastructure
 * needed to run Python compilations.
 */
@Immutable
@SkylarkModule(
    name = "py",
    doc = "A configuration fragment for Python.",
    category = SkylarkModuleCategory.CONFIGURATION_FRAGMENT)
public class PythonConfiguration extends BuildConfiguration.Fragment {

  private final PythonVersion version;
  private final TriState buildPythonZip;
  private final boolean buildTransitiveRunfilesTrees;

  PythonConfiguration(
      PythonVersion defaultPythonVersion,
      TriState buildPythonZip,
      boolean buildTransitiveRunfilesTrees) {
    this.version = defaultPythonVersion;
    this.buildPythonZip = buildPythonZip;
    this.buildTransitiveRunfilesTrees = buildTransitiveRunfilesTrees;
  }

  /**
   * Returns the Python version to use.
   *
   * <p>Specified using either the {@code --python_version} flag and {@code python_version} rule
   * attribute (new API), or the {@code --force_python} flag and {@code default_python_version} rule
   * attribute (old API).
   */
  public PythonVersion getPythonVersion() {
    return version;
  }

  @Override
  public String getOutputDirectoryName() {
    // TODO(brandjon): Implement alternative semantics for controlling which python version(s) get
    // suffixed roots.
    Preconditions.checkState(version.isTargetValue());
    // The only possible Python target version values are PY2 and PY3. For now, PY2 gets the normal
    // output directory name, and PY3 gets a "-py3" suffix.
    Verify.verify(
        PythonVersion.TARGET_VALUES.size() == 2, // If there is only 1, we don't need this method.
        "Detected a change in PythonVersion.TARGET_VALUES so that there are no longer two Python "
            + "versions. Please check that PythonConfiguration#getOutputDirectoryName() is still "
            + "needed and is still able to avoid output directory clashes, then update this "
            + "canary message.");
    if (version.equals(PythonVersion.PY2)) {
      return null;
    } else {
      return Ascii.toLowerCase(version.toString());
    }
  }

  @Override
  public void reportInvalidOptions(EventHandler reporter, BuildOptions buildOptions) {
    PythonOptions pythonOptions = buildOptions.get(PythonOptions.class);
    if (pythonOptions.pythonVersion != null
        && !pythonOptions.experimentalBetterPythonVersionMixing) {
      reporter.handle(
          Event.error(
              "`--python_version` is only allowed with "
                  + "`--experimental_better_python_version_mixing`"));
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
