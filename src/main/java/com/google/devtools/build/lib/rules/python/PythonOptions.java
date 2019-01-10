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
 * <p>Due to the migration associated with {@code --experimental_better_python_version_mixing} (see
 * #6583), the Python major version mode ({@code PY2} vs {@code PY3}) is a function of three
 * separate flags: this experimental feature-guarding flag, the old version flag {@code
 * --force_python}, and the new version flag {@code --python_version}. See {@link #getPythonVersion}
 * for more details.
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

  @Option(
      name = "experimental_better_python_version_mixing",
      // TODO(brandjon): Do not flip until we have an answer for how to guard the "python_version"
      // attribute without hacking up native.existing_rules(). See b/122596733.
      defaultValue = "false",
      // TODO(brandjon): Change to OptionDocumentationCategory.SKYLARK_SEMANTICS when this is
      // sufficiently implemented/documented. Also fill in the ref in the help text below.
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.LOADING_AND_ANALYSIS},
      metadataTags = {OptionMetadataTag.EXPERIMENTAL},
      help =
          "If true, enables use of the `--python_version` flag and the `python_version` "
              + "attribute for `py_binary` and `py_test`, and uses the new PY2/PY3 version "
              + "semantics. See <TODO: ADD LINK> for more details.")
  public boolean experimentalBetterPythonVersionMixing;

  /**
   * This field should be either null, {@code PY2}, or {@code PY3}. Other {@code PythonVersion}
   * values do not represent distinct Python versions and are not allowed.
   *
   * <p>Null represents that the value is not set by the user. This is only relevant for deciding
   * whether or not to ignore the old flag, {@code --force_python}. Rule logic can't tell whether or
   * not this field is null.
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
   * This field should be either null, {@code PY2}, or {@code PY3}. Other {@code PythonVersion}
   * values do not represent distinct Python versions and are not allowed.
   *
   * <p>Null represents that the value is not set by the user. When {@code
   * --experimental_better_python_version_mixing} is false, null means that the default value {@link
   * PythonVersion#DEFAULT_TARGET_VALUE} should be used, but that it is possible for {@link
   * PythonVersionTransition} to override this default. When the experimental flag is true, {@code
   * PythonVersionTransition} can always override the version anyway, and null has the same effect
   * as setting it to the default.
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
   * This field should be either null, {@code PY2}, or {@code PY3}. Other {@code PythonVersion}
   * values do not represent distinct Python versions and are not allowed.
   *
   * <p>Null is treated the same as the default ({@link PythonVersion#DEFAULT_TARGET_VALUE}).
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
    if (experimentalBetterPythonVersionMixing) {
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
   * <p>The version is taken from the following in order:
   *
   * <ul>
   *   <li>If {@code --experimental_better_python_version_mixing} is true, then it is the value of
   *       {@code --python_version} if not null, otherwise {@code --force_python} if not null,
   *       otherwise {@link PythonVersion#DEFAULT_TARGET_VALUE}.
   *   <li>If {@code --experimental_better_python_version_mixing} is false, then it is the same
   *       except {@code --python_version} is ignored.
   * </ul>
   */
  public PythonVersion getPythonVersion() {
    if (experimentalBetterPythonVersionMixing) {
      if (pythonVersion != null) {
        return pythonVersion;
      }
    }
    return (forcePython != null) ? forcePython : PythonVersion.DEFAULT_TARGET_VALUE;
  }

  /**
   * Returns whether a Python version transition to {@code version} is allowed and not a no-op.
   *
   * <p>Under the new semantics ({@link #experimentalBetterPythonVersionMixing} is true), version
   * transitions are always allowed, so this just returns whether the new version is different from
   * the existing one. However, as a compatibility measure for {@code select()}s that depend on
   * {@code "force_python"}, transitioning is still done when {@code forcePython} is not in
   * agreement with {@link #getPythonVersion}, even if {@code #getPythonVersion}'s value would be
   * unaffected.
   *
   * <p>Under the old semantics ({@link #experimentalBetterPythonVersionMixing} is false), version
   * transitions are not allowed once the version has already been set ({@link #forcePython} is
   * non-null). Due to a historical bug, it is also not allowed to transition {@code forcePython} to
   * the hard-coded default value. Under these constraints, any transition that is allowed is also
   * not a no-op.
   *
   * @throws IllegalArgumentException if {@code version} is not {@code PY2} or {@code PY3}
   */
  public boolean canTransitionPythonVersion(PythonVersion version) {
    Preconditions.checkArgument(version.isTargetValue());
    if (experimentalBetterPythonVersionMixing) {
      PythonVersion currentVersion = getPythonVersion();
      return !version.equals(currentVersion) || !version.equals(forcePython);
    } else {
      return forcePython == null && !version.equals(PythonVersion.DEFAULT_TARGET_VALUE);
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
   * false), after this method is called {@code transitionPythonVersion} will not be able to change
   * the version ({@code forcePython} will be non-null).
   *
   * <p>To help avoid breaking {@code select()} expressions that check the value of {@code
   * "force_python"}, under the new semantics both {@code pythonVersion} and {@code forcePython} are
   * updated. Note that it is still not guaranteed that all instances of {@code PythonOptions} that
   * use the new semantics have {@code forcePython} equal {@code pythonVersion} -- in particular,
   * this might not be the case for targets that have not gone through a {@link
   * PythonVersionTransition}.
   *
   * @throws IllegalArgumentException if {@code version} is not {@code PY2} or {@code PY3}
   */
  public void setPythonVersion(PythonVersion version) {
    Preconditions.checkArgument(version.isTargetValue());
    if (experimentalBetterPythonVersionMixing) {
      this.pythonVersion = version;
      // Meaningless to getPythonVersion(), but read by select()s that depend on "force_python".
      this.forcePython = version;
    } else {
      this.forcePython = version;
    }
  }

  @Override
  public FragmentOptions getHost() {
    PythonOptions hostPythonOptions = (PythonOptions) getDefault();
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
