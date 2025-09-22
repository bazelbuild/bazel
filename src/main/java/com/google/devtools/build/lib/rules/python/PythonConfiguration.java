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
import com.google.devtools.build.docgen.annot.DocCategory;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.Fragment;
import com.google.devtools.build.lib.analysis.config.RequiresOptions;
import com.google.devtools.build.lib.analysis.starlark.annotations.StarlarkConfigurationField;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.common.options.TriState;
import javax.annotation.Nullable;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.StarlarkValue;

/**
 * The configuration fragment containing information about the various pieces of infrastructure
 * needed to run Python compilations.
 */
@Immutable
@StarlarkBuiltin(
    name = "py",
    doc = "A configuration fragment for Python.",
    category = DocCategory.CONFIGURATION_FRAGMENT)
@RequiresOptions(options = {PythonOptions.class})
public class PythonConfiguration extends Fragment implements StarlarkValue {

  private final PythonVersion version;
  private final PythonVersion defaultVersion;
  private final TriState buildPythonZip;

  // TODO(brandjon): Remove this once migration to PY3-as-default is complete.
  private final boolean py2OutputsAreSuffixed;

  /* Whether to include the build label in unstamped builds. */
  private final boolean includeLabelInLinkstamp;

  private final boolean defaultToExplicitInitPy;
  @Nullable private final Label nativeRulesAllowlist;
  private final boolean disallowNativeRules;
  private final boolean disablePyFragment;

  public PythonConfiguration(BuildOptions buildOptions) {
    PythonOptions pythonOptions = buildOptions.get(PythonOptions.class);
    PythonVersion pythonVersion = pythonOptions.getPythonVersion();

    this.version = pythonVersion;
    this.defaultVersion = pythonOptions.getDefaultPythonVersion();
    this.buildPythonZip = pythonOptions.buildPythonZip;
    this.py2OutputsAreSuffixed = pythonOptions.incompatiblePy2OutputsAreSuffixed;
    this.defaultToExplicitInitPy = pythonOptions.incompatibleDefaultToExplicitInitPy;
    this.nativeRulesAllowlist = pythonOptions.nativeRulesAllowlist;
    this.disallowNativeRules = pythonOptions.disallowNativeRules;
    this.includeLabelInLinkstamp = pythonOptions.includeLabelInPyBinariesLinkstamp;
    this.disablePyFragment = pythonOptions.disablePyFragment;
  }

  @Override
  public boolean shouldInclude() {
    return !disablePyFragment;
  }

  @Override
  public boolean isImmutable() {
    return true; // immutable and Starlark-hashable
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

  @StarlarkMethod(
      name = "default_python_version",
      structField = true,
      doc = "The default python version from --incompatible_py3_is_default")
  public String getDefaultPythonVersionForStarlark() {
    return defaultVersion.name();
  }

  @Override
  public void processForOutputPathMnemonic(Fragment.OutputDirectoriesContext ctx)
      throws Fragment.OutputDirectoriesContext.AddToMnemonicException {
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
    ctx.markAsExplicitInOutputPathFor("python_version");
    if (py2OutputsAreSuffixed) {
      if (version == PythonVersion.PY2) {
        ctx.addToMnemonic("py2");
      }
    } else {
      if (version == PythonVersion.PY3) {
        ctx.addToMnemonic("py3");
      }
    }
  }

  /** Returns whether to build the executable zip file for Python binaries. */
  @StarlarkMethod(
      name = "build_python_zip",
      structField = true,
      doc = "The effective value of --build_python_zip")
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
   * Returns true if executable Python rules should obtain their runtime from the Python toolchain
   * rather than legacy flags.
   */
  @StarlarkMethod(
      name = "use_toolchains",
      structField = true,
      doc = "No-op: Python toolchains are always used.")
  public boolean useToolchains() {
    return true;
  }

  @StarlarkMethod(
      name = "default_to_explicit_init_py",
      structField = true,
      doc = "The value from the --incompatible_default_to_explicit_init_py flag")
  /**
   * Returns true if executable Python rules should only write out empty __init__ files to their
   * runfiles tree when explicitly requested via {@code legacy_create_init}.
   */
  public boolean defaultToExplicitInitPy() {
    return defaultToExplicitInitPy;
  }

  @StarlarkMethod(
      name = "disable_py2",
      structField = true,
      doc = "No-op: PY2 is no longer supported.")
  public boolean getDisablePy2() {
    return true;
  }

  @StarlarkMethod(
      name = "disallow_native_rules",
      structField = true,
      doc = "The value of the --incompatible_python_disallow_native_rules flag.")
  public boolean getDisallowNativeRules() {
    return disallowNativeRules;
  }

  @StarlarkConfigurationField(
      name = "native_rules_allowlist",
      doc = "The value of --python_native_rules_allowlist; may be None if not specified")
  @Nullable
  public Label getNativeRulesAllowlist() {
    return nativeRulesAllowlist;
  }

  /** Returns whether the build label is included in unstamped builds. */
  @StarlarkMethod(
      name = "include_label_in_linkstamp",
      doc = "Whether the build label is included in unstamped builds.",
      structField = true)
  public boolean isIncludeLabelInLinkstamp() {
    return includeLabelInLinkstamp;
  }
}
