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

import com.google.devtools.build.lib.analysis.config.CoreOptionConverters.LabelConverter;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionMetadataTag;
import com.google.devtools.common.options.TriState;

/**
 * Python-related command-line options.
 *
 * <p>Due to the migration of the Python version API (see #6583) and the default Python version (see
 * (see #6647), the Python major version mode ({@code PY2} vs {@code PY3}) is a function of multiple
 * flags. See {@link #getPythonVersion} for more details.
 */
public class PythonOptions extends FragmentOptions {

  @Option(
      name = "build_python_zip",
      defaultValue = "auto",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.NO_OP},
      deprecationWarning =
          "The '--no' prefix is no longer supported for this flag. Please use"
              + " --build_python_zip=false instead.",
      help = "Deprecated. No-op.")
  public TriState buildPythonZip;

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

  @Option(
      name = "incompatible_remove_ctx_py_fragment",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.BUILD_FILE_SEMANTICS},
      metadataTags = {OptionMetadataTag.INCOMPATIBLE_CHANGE},
      help =
          "When true, Python build flags are defined with Python rules (in BUIILD files) and"
              + " ctx.fragments.py is undefined. This is a migration flag to move all Python flags "
              + " from core Bazel to Python rules.")
  public boolean disablePyFragment;

  /** Returns the Python major version ({@code PY3}) that targets should be built for. */
  public PythonVersion getPythonVersion() {
    return PythonVersion.PY3;
  }
}
