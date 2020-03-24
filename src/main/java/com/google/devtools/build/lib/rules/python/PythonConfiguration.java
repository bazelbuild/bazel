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

import com.google.common.base.Preconditions;
import com.google.common.base.Verify;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.syntax.StarlarkValue;
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
public class PythonConfiguration extends BuildConfiguration.Fragment implements StarlarkValue {

  private final PythonVersion version;
  private final PythonVersion defaultVersion;
  private final TriState buildPythonZip;
  private final boolean buildTransitiveRunfilesTrees;

  // TODO(brandjon): Remove these once migration to the new version API is complete (#6583).
  private final boolean oldPyVersionApiAllowed;
  private final boolean useNewPyVersionSemantics;

  // TODO(brandjon): Remove this once migration to PY3-as-default is complete.
  private final boolean py2OutputsAreSuffixed;

  // TODO(brandjon): Remove this once migration to the new provider is complete (#7010).
  private final boolean disallowLegacyPyProvider;

  // TODO(brandjon): Remove this once migration to Python toolchains is complete.
  private final boolean useToolchains;

  // TODO(brandjon): Remove this once migration for native rule access is complete.
  private final boolean loadPythonRulesFromBzl;

  private final boolean defaultToExplicitInitPy;

  PythonConfiguration(
      PythonVersion version,
      PythonVersion defaultVersion,
      TriState buildPythonZip,
      boolean buildTransitiveRunfilesTrees,
      boolean oldPyVersionApiAllowed,
      boolean useNewPyVersionSemantics,
      boolean py2OutputsAreSuffixed,
      boolean disallowLegacyPyProvider,
      boolean useToolchains,
      boolean loadPythonRulesFromBzl,
      boolean defaultToExplicitInitPy) {
    this.version = version;
    this.defaultVersion = defaultVersion;
    this.buildPythonZip = buildPythonZip;
    this.buildTransitiveRunfilesTrees = buildTransitiveRunfilesTrees;
    this.oldPyVersionApiAllowed = oldPyVersionApiAllowed;
    this.useNewPyVersionSemantics = useNewPyVersionSemantics;
    this.py2OutputsAreSuffixed = py2OutputsAreSuffixed;
    this.disallowLegacyPyProvider = disallowLegacyPyProvider;
    this.useToolchains = useToolchains;
    this.loadPythonRulesFromBzl = loadPythonRulesFromBzl;
    this.defaultToExplicitInitPy = defaultToExplicitInitPy;
  }

  /**
   * Returns the Python version to use.
   *
   * <p>Specified using either the {@code --python_version} flag and {@code python_version} rule
   * attribute (new API), or the {@code default_python_version} rule attribute (old API).
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
    Preconditions.checkState(version.isTargetValue());
    // The only possible Python target version values are PY2 and PY3. Historically, PY3 targets got
    // a "-py3" suffix and PY2 targets got the empty suffix, so that the bazel-bin symlink pointed
    // to Python 2 targets. When --incompatible_py2_outputs_are_suffixed is enabled, this is
    // reversed: PY2 targets get "-py2" and PY3 targets get the empty suffix.
    Verify.verify(
        PythonVersion.TARGET_VALUES.size() == 2, // If there is only 1, we don't need this method.
        "Detected a change in PythonVersion.TARGET_VALUES so that there are no longer two Python "
            + "versions. Please check that PythonConfiguration#getOutputDirectoryName() is still "
            + "needed and is still able to avoid output directory clashes, then update this "
            + "canary message.");
    if (py2OutputsAreSuffixed) {
      return version == PythonVersion.PY2 ? "py2" : null;
    } else {
      return version == PythonVersion.PY3 ? "py3" : null;
    }
  }

  @Override
  public void reportInvalidOptions(EventHandler reporter, BuildOptions buildOptions) {
    PythonOptions opts = buildOptions.get(PythonOptions.class);
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

  /** Returns whether use of the {@code default_python_version} attribute is allowed. */
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

  /**
   * Returns true if executable Python rules should obtain their runtime from the Python toolchain
   * rather than legacy flags.
   */
  public boolean useToolchains() {
    return useToolchains;
  }

  /**
   * Returns true if native Python rules should fail at analysis time when the magic tag, {@code
   * __PYTHON_RULES_MIGRATION_DO_NOT_USE_WILL_BREAK__}, is not present.
   *
   * <p>This tag is set by the macros in bazelbuild/rules_python and should not be used anywhere
   * else.
   */
  public boolean loadPythonRulesFromBzl() {
    return loadPythonRulesFromBzl;
  }

  /**
   * Returns true if executable Python rules should only write out empty __init__ files to their
   * runfiles tree when explicitly requested via {@code legacy_create_init}.
   */
  public boolean defaultToExplicitInitPy() {
    return defaultToExplicitInitPy;
  }
}
