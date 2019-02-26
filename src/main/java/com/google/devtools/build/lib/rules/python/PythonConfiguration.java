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
  private final PythonVersion defaultVersion;
  private final TriState buildPythonZip;
  private final boolean buildTransitiveRunfilesTrees;

  // TODO(brandjon): Remove these once migration to the new version API is complete (#6583).
  private final boolean oldPyVersionApiAllowed;
  private final boolean useNewPyVersionSemantics;

  // TODO(brandjon): Remove this once migration to the new provider is complete (#7010).
  private final boolean disallowLegacyPyProvider;

  PythonConfiguration(
      PythonVersion version,
      PythonVersion defaultVersion,
      TriState buildPythonZip,
      boolean buildTransitiveRunfilesTrees,
      boolean oldPyVersionApiAllowed,
      boolean useNewPyVersionSemantics,
      boolean disallowLegacyPyProvider) {
    this.version = version;
    this.defaultVersion = defaultVersion;
    this.buildPythonZip = buildPythonZip;
    this.buildTransitiveRunfilesTrees = buildTransitiveRunfilesTrees;
    this.oldPyVersionApiAllowed = oldPyVersionApiAllowed;
    this.useNewPyVersionSemantics = useNewPyVersionSemantics;
    this.disallowLegacyPyProvider = disallowLegacyPyProvider;
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

  /**
   * Returns the default Python version to use on targets that omit their {@code python_version}
   * attribute.
   *
   * <p>Specified using {@code --incompatible_py3_is_default}. Long-term, the default will simply be
   * hardcoded as {@code PY3}.
   *
   * <p>This information is stored on the configuration for the benefit of callers in rule analysis.
   * However, transitions have access to the option fragment instead of the configuration fragment,
   * and should rely on {@link PythonOptions#getDefaultPythonVersion} instead.
   */
  public PythonVersion getDefaultPythonVersion() {
    return defaultVersion;
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
    PythonOptions opts = buildOptions.get(PythonOptions.class);
    if (opts.forcePython != null && opts.incompatibleRemoveOldPythonVersionApi) {
      reporter.handle(
          Event.error(
              "`--force_python` is disabled by `--incompatible_remove_old_python_version_api`"));
    }
    if (opts.incompatiblePy3IsDefault && !opts.incompatibleAllowPythonVersionTransitions) {
      reporter.handle(
          Event.error(
              "cannot enable `--incompatible_py3_is_default` without also enabling "
                  + "`--incompatible_allow_python_version_transitions`"));
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

  /**
   * Returns whether use of {@code --force_python} flag and {@code default_python_version} attribute
   * is allowed.
   */
  public boolean oldPyVersionApiAllowed() {
    return oldPyVersionApiAllowed;
  }

  /** Returns true if the new semantics should be used for transitions on the Python version. */
  public boolean useNewPyVersionSemantics() {
    return useNewPyVersionSemantics;
  }

  /**
   * Returns true if Python rules should omit the legacy "py" provider and fail-fast when given this
   * provider from their {@code deps}.
   *
   * <p>Any rules that pass this provider should be updated to pass {@code PyInfo} instead.
   */
  public boolean disallowLegacyPyProvider() {
    return disallowLegacyPyProvider;
  }
}
