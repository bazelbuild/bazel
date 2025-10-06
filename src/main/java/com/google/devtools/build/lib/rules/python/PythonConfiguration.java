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

  private final TriState buildPythonZip;

  /* Whether to include the build label in unstamped builds. */
  private final boolean includeLabelInLinkstamp;

  private final boolean defaultToExplicitInitPy;
  @Nullable private final Label nativeRulesAllowlist;
  private final boolean disallowNativeRules;
  private final boolean disablePyFragment;

  public PythonConfiguration(BuildOptions buildOptions) {
    PythonOptions pythonOptions = buildOptions.get(PythonOptions.class);

    this.buildPythonZip = pythonOptions.buildPythonZip;
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

  @StarlarkMethod(
      name = "default_python_version",
      structField = true,
      doc = "No-op: PY3 is the default Python version.")
  public String getDefaultPythonVersionForStarlark() {
    return PythonVersion.PY3.name();
  }

  /** Returns whether to build the executable zip file for Python binaries. */
  @StarlarkMethod(
      name = "build_python_zip",
      structField = true,
      doc = "The effective value of --build_python_zip")
  public boolean buildPythonZip() {
    return switch (buildPythonZip) {
      case YES -> true;
      case NO -> false;
      default -> OS.getCurrent() == OS.WINDOWS;
    };
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
