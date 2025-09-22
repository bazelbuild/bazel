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
import com.google.devtools.build.lib.analysis.config.CoreOptionConverters.LabelConverter;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.common.options.Converter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionMetadataTag;
import com.google.devtools.common.options.OptionsParsingException;
import com.google.devtools.common.options.TriState;

/**
 * Python-related command-line options.
 *
 * <p>Due to the migration of the Python version API (see #6583) and the default Python version (see
 * (see #6647), the Python major version mode ({@code PY2} vs {@code PY3}) is a function of multiple
 * flags. See {@link #getPythonVersion} for more details.
 */
public class PythonOptions extends FragmentOptions {

  /** Converter for options that take ({@code PY2} or {@code PY3}). */
  // We don't use EnumConverter because we want to disallow non-target PythonVersion values.
  public static class TargetPythonVersionConverter extends Converter.Contextless<PythonVersion> {

    @Override
    public PythonVersion convert(String input) throws OptionsParsingException {
      try {
        // Although in rule attributes the enum values are case sensitive, the convention from
        // EnumConverter is that the options parser is case insensitive.
        input = Ascii.toUpperCase(input);
        return PythonVersion.parseTargetValue(input);
      } catch (IllegalArgumentException ex) {
        throw new OptionsParsingException(
            "Not a valid Python major version, should be PY2 or PY3", ex);
      }
    }

    @Override
    public String getTypeDescription() {
      return "PY2 or PY3";
    }
  }

  @Option(
      name = "build_python_zip",
      defaultValue = "auto",
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
      help = "Build python executable zip; on on Windows, off on other platforms")
  public TriState buildPythonZip;

  /**
   * Native rule logic should call {@link #getDefaultPythonVersion} instead of accessing this option
   * directly.
   */
  @Option(
      name = "incompatible_py3_is_default",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.GENERIC_INPUTS,
      effectTags = {
        OptionEffectTag.LOADING_AND_ANALYSIS,
        OptionEffectTag.AFFECTS_OUTPUTS // because of "-py2"/"-py3" output root
      },
      metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
      help =
          "If true, `py_binary` and `py_test` targets that do not set their `python_version` (or "
              + "`default_python_version`) attribute will default to PY3 rather than to PY2. If "
              + "you set this flag it is also recommended to set "
              + "`--incompatible_py2_outputs_are_suffixed`.")
  public boolean incompatiblePy3IsDefault;

  @Option(
      name = "incompatible_py2_outputs_are_suffixed",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.GENERIC_INPUTS,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
      metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
      help =
          "If true, targets built in the Python 2 configuration will appear under an output root "
              + "that includes the suffix '-py2', while targets built for Python 3 will appear "
              + "in a root with no Python-related suffix. This means that the `bazel-bin` "
              + "convenience symlink will point to Python 3 targets rather than Python 2. "
              + "If you enable this option it is also recommended to enable "
              + "`--incompatible_py3_is_default`.")
  public boolean incompatiblePy2OutputsAreSuffixed;

  /**
   * This field should be either null (unset), {@code PY2}, or {@code PY3}. Other {@code
   * PythonVersion} values do not represent distinct Python versions and are not allowed.
   *
   * <p>Native rule logic should call {@link #getPythonVersion} / {@link #setPythonVersion} instead
   * of accessing this option directly. BUILD/.bzl code should {@code select()} on {@code <tools
   * repo>//tools/python:python_version} rather than on this option directly.
   */
  @Option(
      name = "python_version",
      defaultValue = "null",
      converter = TargetPythonVersionConverter.class,
      documentationCategory = OptionDocumentationCategory.GENERIC_INPUTS,
      effectTags = {
        OptionEffectTag.LOADING_AND_ANALYSIS,
        OptionEffectTag.AFFECTS_OUTPUTS // because of "-py2"/"-py3" output root
      },
      help =
          "The Python major version mode, either `PY2` or `PY3`. Note that this is overridden by "
              + "`py_binary` and `py_test` targets (even if they don't explicitly specify a "
              + "version) so there is usually not much reason to supply this flag.")
  public PythonVersion pythonVersion;

  @Option(
      name = "incompatible_default_to_explicit_init_py",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.GENERIC_INPUTS,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
      metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
      help =
          "This flag changes the default behavior so that __init__.py files are no longer "
              + "automatically created in the runfiles of Python targets. Precisely, when a "
              + "py_binary or py_test target has legacy_create_init set to \"auto\" (the default), "
              + "it is treated as false if and only if this flag is set. See "
              + "https://github.com/bazelbuild/bazel/issues/10076.")
  public boolean incompatibleDefaultToExplicitInitPy;

  @Option(
      name = "python_native_rules_allowlist",
      documentationCategory = OptionDocumentationCategory.INPUT_STRICTNESS,
      effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
      defaultValue = "null",
      converter = LabelConverter.class,
      help =
          "An allowlist (package_group target) to use when enforcing "
              + "--incompatible_python_disallow_native_rules.")
  public Label nativeRulesAllowlist;

  @Option(
      name = "incompatible_python_disallow_native_rules",
      documentationCategory = OptionDocumentationCategory.STARLARK_SEMANTICS,
      effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
      defaultValue = "false",
      metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
      help =
          "When true, an error occurs when using the builtin py_* rules; instead the rule_python"
              + " rules should be used. See https://github.com/bazelbuild/bazel/issues/17773 for"
              + " more information and migration instructions.")
  public boolean disallowNativeRules;

  @Option(
      name = "experimental_py_binaries_include_label",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      metadataTags = {OptionMetadataTag.EXPERIMENTAL},
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
      help = "py_binary targets include their label even when stamping is disabled.")
  public boolean includeLabelInPyBinariesLinkstamp;

  // Helper field to store hostForcePython in exec configuration
  private PythonVersion defaultPythonVersion = null;

  @Option(
      name = "incompatible_remove_ctx_py_fragment",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS},
      metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
      help =
          "When true, Python build flags are defined with Python rules (in BUIILD files) and"
              + " ctx.fragments.py is undefined. This is a migration flag to move all Python flags "
              + " from core Bazel to Python rules.")
  public boolean disablePyFragment;

  /**
   * Returns the Python major version ({@code PY2} or {@code PY3}) that targets that do not specify
   * a version should be built for.
   */
  public PythonVersion getDefaultPythonVersion() {
    if (defaultPythonVersion != null) {
      return defaultPythonVersion;
    }
    return incompatiblePy3IsDefault ? PythonVersion.PY3 : PythonVersion.PY2;
  }

  /**
   * Returns the Python major version ({@code PY2} or {@code PY3}) that targets should be built for.
   *
   * <p>The version is taken as the value of {@code --python_version} if not null, otherwise {@link
   * #getDefaultPythonVersion}.
   */
  public PythonVersion getPythonVersion() {
    if (pythonVersion != null) {
      return pythonVersion;
    } else {
      return getDefaultPythonVersion();
    }
  }

  /**
   * Returns whether a Python version transition to {@code version} is not a no-op.
   *
   * @throws IllegalArgumentException if {@code version} is not {@code PY2} or {@code PY3}
   */
  public boolean canTransitionPythonVersion(PythonVersion version) {
    Preconditions.checkArgument(version.isTargetValue());
    return !version.equals(getPythonVersion());
  }

  /**
   * Sets the Python version to {@code version}.
   *
   * <p>Since this is a mutation, it should only be called on a newly constructed instance.
   *
   * @throws IllegalArgumentException if {@code version} is not {@code PY2} or {@code PY3}
   */
  // TODO(brandjon): Consider removing this mutator now that the various flags and semantics it
  // used to consider are gone. We'd revert to just setting the public option field directly.
  public void setPythonVersion(PythonVersion version) {
    Preconditions.checkArgument(version.isTargetValue());
    this.pythonVersion = version;
  }

  @Override
  public FragmentOptions getNormalized() {
    // We want to ensure that options with "null" physical default values are normalized, to avoid
    // #7808.
    PythonOptions newOptions = (PythonOptions) clone();
    newOptions.setPythonVersion(newOptions.getPythonVersion());
    return newOptions;
  }
}
