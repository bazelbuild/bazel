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
 * <p>Due to the migration from the old Python version API to the new (see #6583), the Python major
 * version mode ({@code PY2} vs {@code PY3}) is a function of multiple flags. See {@link
 * #getPythonVersion} for more details.
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

  // TODO(brandjon): For both experimental options below, add documentation and add a link to it
  // from the help text, then change the documentationCategory to SKYLARK_SEMANTICS.

  @Option(
      name = "experimental_remove_old_python_version_api",
      // TODO(brandjon): Do not flip until we have an answer for how to disallow the
      // "default_python_version" attribute without hacking up native.existing_rules(). See
      // #7071 and b/122596733.
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
      metadataTags = {OptionMetadataTag.EXPERIMENTAL},
      help =
          "If true, disables use of the `--force_python` flag and the `default_python_version` "
              + "attribute for `py_binary` and `py_test`.")
  public boolean experimentalRemoveOldPythonVersionApi;

  @Option(
      name = "experimental_better_python_version_mixing",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
      metadataTags = {OptionMetadataTag.EXPERIMENTAL},
      help = "If true, Python rules use the new PY2/PY3 version semantics.")
  public boolean experimentalBetterPythonVersionMixing;

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
      // TODO(brandjon): Change to OptionDocumentationCategory.GENERIC_INPUTS when this is
      // sufficiently implemented/documented.
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {
        OptionEffectTag.LOADING_AND_ANALYSIS,
        OptionEffectTag.AFFECTS_OUTPUTS // because of "-py3" output root
      },
      help =
          "The Python major version mode, either `PY2` or `PY3`. Note that this is overridden by "
              + "`py_binary` and `py_test` targets (whether or not they specify their version "
              + "explicitly), so there is usually not much reason to supply this flag.")
  public PythonVersion pythonVersion;

  private static final OptionDefinition PYTHON_VERSION_DEFINITION =
      OptionsParser.getOptionDefinitionByName(PythonOptions.class, "python_version");

  /**
   * This field should be either null (unset), {@code PY2}, or {@code PY3}. Other {@code
   * PythonVersion} values do not represent distinct Python versions and are not allowed.
   *
   * <p>This flag is not accessible to the user when {@link #experimentalRemoveOldPythonVersionApi}
   * is true.
   *
   * <p>Native rule logic should call {@link #getPythonVersion} / {@link #setPythonVersion} instead
   * of accessing this option directly. BUILD/.bzl code should {@code select()} on {@code <tools
   * repo>//tools/python:python_version} rather than on this option directly.
   */
  @Option(
      name = "force_python",
      defaultValue = "null",
      converter = TargetPythonVersionConverter.class,
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS, OptionEffectTag.AFFECTS_OUTPUTS},
      help = "Overrides default_python_version attribute. Can be \"PY2\" or \"PY3\".")
  public PythonVersion forcePython;

  private static final OptionDefinition FORCE_PYTHON_DEFINITION =
      OptionsParser.getOptionDefinitionByName(PythonOptions.class, "force_python");

  /**
   * This field should be either null (unset), {@code PY2}, or {@code PY3}. Other {@code
   * PythonVersion} values do not represent distinct Python versions and are not allowed.
   *
   * <p>Null is treated the same as the default ({@link PythonVersion#DEFAULT_TARGET_VALUE}).
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
      help =
          "Overrides default_python_version attribute for the host configuration."
              + " Can be \"PY2\" or \"PY3\".")
  public PythonVersion hostForcePython;

  private static final OptionDefinition HOST_FORCE_PYTHON_DEFINITION =
      OptionsParser.getOptionDefinitionByName(PythonOptions.class, "host_force_python");

  @Override
  public Map<OptionDefinition, SelectRestriction> getSelectRestrictions() {
    // TODO(brandjon): Add an error string that references documentation explaining to use
    // @bazel_tools//tools/python:python_version instead.
    ImmutableMap.Builder<OptionDefinition, SelectRestriction> restrictions = ImmutableMap.builder();
    restrictions.put(
        PYTHON_VERSION_DEFINITION,
        new SelectRestriction(/*visibleWithinToolsPackage=*/ true, /*errorMessage=*/ null));
    if (experimentalRemoveOldPythonVersionApi) {
      restrictions.put(
          FORCE_PYTHON_DEFINITION,
          new SelectRestriction(/*visibleWithinToolsPackage=*/ true, /*errorMessage=*/ null));
      restrictions.put(
          HOST_FORCE_PYTHON_DEFINITION,
          new SelectRestriction(/*visibleWithinToolsPackage=*/ false, /*errorMessage=*/ null));
    }
    return restrictions.build();
  }

  /**
   * Returns the Python major version ({@code PY2} or {@code PY3}) that targets should be built for.
   *
   * <p>The version is taken as the value of {@code --python_version} if not null, otherwise {@code
   * --force_python} if not null, otherwise {@link PythonVersion#DEFAULT_TARGET_VALUE}.
   */
  public PythonVersion getPythonVersion() {
    if (pythonVersion != null) {
      return pythonVersion;
    } else if (forcePython != null) {
      return forcePython;
    } else {
      return PythonVersion.DEFAULT_TARGET_VALUE;
    }
  }

  /**
   * Returns whether a Python version transition to {@code version} is allowed and not a no-op.
   *
   * <p>Under the new semantics ({@link #experimentalBetterPythonVersionMixing} is true), version
   * transitions are always allowed, so this essentially returns whether the new version is
   * different from the existing one. However, to improve compatibility for unmigrated {@code
   * select()}s that depend on {@code "force_python"}, if the old API is still enabled then
   * transitioning is still done whenever {@link #forcePython} is not in agreement with the
   * requested version, even if {@link #getPythonVersion}'s value would be unaffected.
   *
   * <p>Under the old semantics ({@link #experimentalBetterPythonVersionMixing} is false), version
   * transitions are not allowed once the version has already been set ({@link #forcePython} or
   * {@link #pythonVersion} is non-null). Due to a historical bug, it is also not allowed to
   * transition the version to the hard-coded default value. Under these constraints, there is only
   * one transition possible, from null to the non-default value, and it is never a no-op.
   *
   * @throws IllegalArgumentException if {@code version} is not {@code PY2} or {@code PY3}
   */
  public boolean canTransitionPythonVersion(PythonVersion version) {
    Preconditions.checkArgument(version.isTargetValue());
    if (experimentalBetterPythonVersionMixing) {
      boolean currentVersionNeedsUpdating = !version.equals(getPythonVersion());
      boolean forcePythonNeedsUpdating =
          !experimentalRemoveOldPythonVersionApi && !version.equals(forcePython);
      return currentVersionNeedsUpdating || forcePythonNeedsUpdating;
    } else {
      boolean currentlyUnset = forcePython == null && pythonVersion == null;
      boolean transitioningToNonDefault = !version.equals(PythonVersion.DEFAULT_TARGET_VALUE);
      return currentlyUnset && transitioningToNonDefault;
    }
  }

  /**
   * Manipulates the Python version fields so that {@link #getPythonVersion()} returns {@code
   * version}.
   *
   * <p>This method is a mutation on the current instance, so it should only be invoked on a newly
   * constructed instance. The mutation does not depend on whether or not {@link
   * #canTransitionPythonVersion} would return true.
   *
   * <p>If the old semantics are in effect ({@link #experimentalBetterPythonVersionMixing} is
   * false), after this method is called {@link #canTransitionPythonVersion} will return false.
   *
   * <p>To help avoid breaking old-API {@code select()} expressions that check the value of {@code
   * "force_python"}, both the old and new flags are updated even though {@code --python_version}
   * takes precedence over {@code --force_python}.
   *
   * @throws IllegalArgumentException if {@code version} is not {@code PY2} or {@code PY3}
   */
  public void setPythonVersion(PythonVersion version) {
    Preconditions.checkArgument(version.isTargetValue());
    this.pythonVersion = version;
    // Update forcePython, but if the flag to remove the old API is enabled, no one will be able
    // to tell anyway.
    this.forcePython = version;
  }

  @Override
  public FragmentOptions getHost() {
    PythonOptions hostPythonOptions = (PythonOptions) getDefault();
    hostPythonOptions.experimentalRemoveOldPythonVersionApi = experimentalRemoveOldPythonVersionApi;
    hostPythonOptions.experimentalBetterPythonVersionMixing = experimentalBetterPythonVersionMixing;
    PythonVersion hostVersion =
        (hostForcePython != null) ? hostForcePython : PythonVersion.DEFAULT_TARGET_VALUE;
    hostPythonOptions.setPythonVersion(hostVersion);
    hostPythonOptions.buildPythonZip = buildPythonZip;
    return hostPythonOptions;
  }

  @Option(
      name = "experimental_build_transitive_python_runfiles",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS, OptionEffectTag.AFFECTS_OUTPUTS},
      help =
          "Build the runfiles trees of py_binary targets that appear in the transitive "
              + "data runfiles of another binary.")
  public boolean buildTransitiveRunfilesTrees;
}
