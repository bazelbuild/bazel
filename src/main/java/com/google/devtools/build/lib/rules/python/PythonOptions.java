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
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.common.options.Converter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDefinition;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionMetadataTag;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;
import com.google.devtools.common.options.TriState;
import java.util.Map;

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
  public static class TargetPythonVersionConverter implements Converter<PythonVersion> {

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
   * Deprecated machinery for setting the Python version; will be removed soon.
   *
   * <p>Not GraveyardOptions'd because we'll delete this alongside other soon-to-be-removed options
   * in this file.
   */
  @Option(
      name = "incompatible_remove_old_python_version_api",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
      metadataTags = {
        OptionMetadataTag.INCOMPATIBLE_CHANGE,
        OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
      },
      help = "No-op, will be removed soon.")
  public boolean incompatibleRemoveOldPythonVersionApi;

  /**
   * Deprecated machinery for setting the Python version; will be removed soon.
   *
   * <p>Not GraveyardOptions'd because we'll delete this alongside other soon-to-be-removed options
   * in this file.
   */
  @Option(
      name = "incompatible_allow_python_version_transitions",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
      metadataTags = {
        OptionMetadataTag.INCOMPATIBLE_CHANGE,
        OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
      },
      help = "No-op, will be removed soon.")
  public boolean incompatibleAllowPythonVersionTransitions;

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
      metadataTags = {
        OptionMetadataTag.INCOMPATIBLE_CHANGE,
        OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
      },
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
      metadataTags = {
        OptionMetadataTag.INCOMPATIBLE_CHANGE,
        OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
      },
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

  private static final OptionDefinition PYTHON_VERSION_DEFINITION =
      OptionsParser.getOptionDefinitionByName(PythonOptions.class, "python_version");

  /**
   * Deprecated machinery for setting the Python version; will be removed soon.
   *
   * <p>Not in GraveyardOptions because we still want to prohibit users from select()ing on it.
   */
  @Option(
      name = "force_python",
      defaultValue = "null",
      converter = TargetPythonVersionConverter.class,
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS, OptionEffectTag.AFFECTS_OUTPUTS},
      help = "No-op, will be removed soon.")
  public PythonVersion forcePython;

  private static final OptionDefinition FORCE_PYTHON_DEFINITION =
      OptionsParser.getOptionDefinitionByName(PythonOptions.class, "force_python");

  /**
   * This field should be either null (unset), {@code PY2}, or {@code PY3}. Other {@code
   * PythonVersion} values do not represent distinct Python versions and are not allowed.
   *
   * <p>Null means to use the default ({@link #getDefaultPythonVersion}).
   *
   * <p>This option is only read by {@link #getHost}. It should not be read by other native code or
   * by {@code select()}s in user code.
   */
  @Option(
      name = "host_force_python",
      defaultValue = "null",
      converter = TargetPythonVersionConverter.class,
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS, OptionEffectTag.AFFECTS_OUTPUTS},
      help = "Overrides the Python version for the host configuration. Can be \"PY2\" or \"PY3\".")
  public PythonVersion hostForcePython;

  private static final OptionDefinition HOST_FORCE_PYTHON_DEFINITION =
      OptionsParser.getOptionDefinitionByName(PythonOptions.class, "host_force_python");

  @Option(
      name = "incompatible_disallow_legacy_py_provider",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.STARLARK_SEMANTICS,
      effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
      metadataTags = {
        OptionMetadataTag.INCOMPATIBLE_CHANGE,
        OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
      },
      help =
          "If set to true, native Python rules will neither produce nor consume the legacy \"py\" "
              + "provider. Use PyInfo instead. Under this flag, passing the legacy provider to a "
              + "Python target will be an error.")
  public boolean incompatibleDisallowLegacyPyProvider;

  // TODO(b/153369373): Delete this flag.
  @Option(
      name = "incompatible_use_python_toolchains",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.GENERIC_INPUTS,
      effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
      metadataTags = {
        OptionMetadataTag.INCOMPATIBLE_CHANGE,
        OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
      },
      help =
          "If set to true, executable native Python rules will use the Python runtime specified by "
              + "the Python toolchain, rather than the runtime given by legacy flags like "
              + "--python_top.")
  public boolean incompatibleUsePythonToolchains;

  @Option(
      name = "experimental_build_transitive_python_runfiles",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS, OptionEffectTag.AFFECTS_OUTPUTS},
      help =
          "Build the runfiles trees of py_binary targets that appear in the transitive "
              + "data runfiles of another binary.")
  public boolean buildTransitiveRunfilesTrees;

  @Option(
      name = "incompatible_default_to_explicit_init_py",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.GENERIC_INPUTS,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
      metadataTags = {
        OptionMetadataTag.INCOMPATIBLE_CHANGE,
        OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
      },
      help =
          "This flag changes the default behavior so that __init__.py files are no longer "
              + "automatically created in the runfiles of Python targets. Precisely, when a "
              + "py_binary or py_test target has legacy_create_init set to \"auto\" (the default), "
              + "it is treated as false if and only if this flag is set. See "
              + "https://github.com/bazelbuild/bazel/issues/10076.")
  public boolean incompatibleDefaultToExplicitInitPy;

  @Override
  public Map<OptionDefinition, SelectRestriction> getSelectRestrictions() {
    // TODO(brandjon): Instead of referencing the python_version target, whose path depends on the
    // tools repo name, reference a standalone documentation page instead.
    ImmutableMap.Builder<OptionDefinition, SelectRestriction> restrictions = ImmutableMap.builder();
    restrictions.put(
        PYTHON_VERSION_DEFINITION,
        new SelectRestriction(
            /*visibleWithinToolsPackage=*/ true,
            "Use @bazel_tools//python/tools:python_version instead."));
    restrictions.put(
        FORCE_PYTHON_DEFINITION,
        new SelectRestriction(
            /*visibleWithinToolsPackage=*/ true,
            "Use @bazel_tools//python/tools:python_version instead."));
    restrictions.put(
        HOST_FORCE_PYTHON_DEFINITION,
        new SelectRestriction(
            /*visibleWithinToolsPackage=*/ false,
            "Use @bazel_tools//python/tools:python_version instead."));
    return restrictions.build();
  }

  /**
   * Returns the Python major version ({@code PY2} or {@code PY3}) that targets that do not specify
   * a version should be built for.
   */
  public PythonVersion getDefaultPythonVersion() {
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
  public FragmentOptions getHost() {
    PythonOptions hostPythonOptions = (PythonOptions) getDefault();
    PythonVersion hostVersion =
        (hostForcePython != null) ? hostForcePython : getDefaultPythonVersion();
    hostPythonOptions.setPythonVersion(hostVersion);
    hostPythonOptions.incompatiblePy3IsDefault = incompatiblePy3IsDefault;
    hostPythonOptions.incompatiblePy2OutputsAreSuffixed = incompatiblePy2OutputsAreSuffixed;
    hostPythonOptions.buildPythonZip = buildPythonZip;
    hostPythonOptions.incompatibleDisallowLegacyPyProvider = incompatibleDisallowLegacyPyProvider;
    hostPythonOptions.incompatibleUsePythonToolchains = incompatibleUsePythonToolchains;

    // Save host options in case of a further exec->host transition.
    hostPythonOptions.hostForcePython = hostForcePython;

    return hostPythonOptions;
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
