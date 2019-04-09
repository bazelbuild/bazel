// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis.skylark;

import static java.nio.charset.StandardCharsets.US_ASCII;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Sets;
import com.google.common.io.BaseEncoding;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.StarlarkDefinedConfigTransition;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Mutability;
import com.google.devtools.build.lib.syntax.Runtime;
import com.google.devtools.build.lib.syntax.Runtime.NoneType;
import com.google.devtools.build.lib.syntax.SkylarkDict;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.common.options.OptionDefinition;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;
import java.lang.reflect.Field;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;

/**
 * Utility class for common work done across {@link StarlarkAttributeTransitionProvider} and {@link
 * StarlarkRuleTransitionProvider}.
 */
public class FunctionTransitionUtil {

  public static final String COMMAND_LINE_OPTION_PREFIX = "//command_line_option:";

  /**
   * Figure out what build settings the given transition changes and apply those changes to the
   * incoming {@link BuildOptions}. For native options, this involves a preprocess step of
   * converting options to their "command line form".
   *
   * <p>Also validate that transitions output the declared results.
   *
   * @param buildOptions the pre-transition build options
   * @param starlarkTransition the transition to apply
   * @param attrObject the attributes of the rule to which this transition is attached
   * @return the post-transition build options.
   */
  static List<BuildOptions> applyAndValidate(
      BuildOptions buildOptions,
      StarlarkDefinedConfigTransition starlarkTransition,
      StructImpl attrObject)
      throws EvalException, InterruptedException {
    // TODO(waltl): consider building this once and use it across different split
    // transitions.
      Map<String, OptionInfo> optionInfoMap = buildOptionInfo(buildOptions);
      SkylarkDict<String, Object> settings =
          buildSettings(buildOptions, optionInfoMap, starlarkTransition);

      ImmutableList.Builder<BuildOptions> splitBuildOptions = ImmutableList.builder();

      ImmutableList<Map<String, Object>> transitions =
          starlarkTransition.getChangedSettings(settings, attrObject);
    validateFunctionOutputsMatchesDeclaredOutputs(transitions, starlarkTransition);

      for (Map<String, Object> transition : transitions) {
        BuildOptions transitionedOptions =
            applyTransition(buildOptions, transition, optionInfoMap, starlarkTransition);
        splitBuildOptions.add(transitionedOptions);
      }
      return splitBuildOptions.build();
  }

  /**
   * Validates that function outputs exactly the set of outputs it declares. More thorough checking
   * (like type checking of output values) is done elsewhere because it requires loading. see {@link
   * StarlarkTransition#validate}
   */
  private static void validateFunctionOutputsMatchesDeclaredOutputs(
      ImmutableList<Map<String, Object>> transitions,
      StarlarkDefinedConfigTransition starlarkTransition)
      throws EvalException {
    for (Map<String, Object> transition : transitions) {
      LinkedHashSet<String> remainingOutputs =
          Sets.newLinkedHashSet(starlarkTransition.getOutputs());
      for (String outputKey : transition.keySet()) {
        if (!remainingOutputs.remove(outputKey)) {
          throw new EvalException(
              starlarkTransition.getLocationForErrorReporting(),
              String.format("transition function returned undeclared output '%s'", outputKey));
        }
      }

      if (!remainingOutputs.isEmpty()) {
        throw new EvalException(
            starlarkTransition.getLocationForErrorReporting(),
            String.format(
                "transition outputs [%s] were not defined by transition function",
                Joiner.on(", ").join(remainingOutputs)));
      }
    }
  }

  /** For all the options in the BuildOptions, build a map from option name to its information. */
  private static Map<String, OptionInfo> buildOptionInfo(BuildOptions buildOptions) {
    ImmutableMap.Builder<String, OptionInfo> builder = new ImmutableMap.Builder<>();

    ImmutableSet<Class<? extends FragmentOptions>> optionClasses =
        buildOptions.getNativeOptions().stream()
            .map(FragmentOptions::getClass)
            .collect(ImmutableSet.toImmutableSet());

    for (Class<? extends FragmentOptions> optionClass : optionClasses) {
      ImmutableList<OptionDefinition> optionDefinitions =
          OptionsParser.getOptionDefinitions(optionClass);
      for (OptionDefinition def : optionDefinitions) {
        String optionName = def.getOptionName();
        builder.put(optionName, new OptionInfo(optionClass, def));
      }
    }

    return builder.build();
  }

  /**
   * Enter the options in buildOptions into a skylark dictionary, and return the dictionary.
   *
   * @throws IllegalArgumentException If the method is unable to look up the value in buildOptions
   *     corresponding to an entry in optionInfoMap
   * @throws RuntimeException If the field corresponding to an option value in buildOptions is
   *     inaccessible due to Java language access control, or if an option name is an invalid key to
   *     the Skylark dictionary
   * @throws EvalException if any of the specified transition inputs do not correspond to a valid
   *     build setting
   */
  static SkylarkDict<String, Object> buildSettings(
      BuildOptions buildOptions,
      Map<String, OptionInfo> optionInfoMap,
      StarlarkDefinedConfigTransition starlarkTransition)
      throws EvalException {
    LinkedHashSet<String> remainingInputs = Sets.newLinkedHashSet(starlarkTransition.getInputs());

    try (Mutability mutability = Mutability.create("build_settings")) {
      SkylarkDict<String, Object> dict = SkylarkDict.withMutability(mutability);

      // Add native options
      for (Map.Entry<String, OptionInfo> entry : optionInfoMap.entrySet()) {
        String optionName = entry.getKey();
        String optionKey = COMMAND_LINE_OPTION_PREFIX + optionName;

        if (!remainingInputs.remove(optionKey)) {
          // This option was not present in inputs. Skip it.
          continue;
        }
        OptionInfo optionInfo = entry.getValue();

        try {
          Field field = optionInfo.getDefinition().getField();
          FragmentOptions options = buildOptions.get(optionInfo.getOptionClass());
          Object optionValue = field.get(options);

          dict.put(optionKey, optionValue == null ? Runtime.NONE : optionValue, null, mutability);
        } catch (IllegalAccessException e) {
          // These exceptions should not happen, but if they do, throw a RuntimeException.
          throw new RuntimeException(e);
        }
      }

      // Add Starlark options
      for (Map.Entry<Label, Object> starlarkOption : buildOptions.getStarlarkOptions().entrySet()) {
        if (!remainingInputs.remove(starlarkOption.getKey().toString())) {
          continue;
        }
        dict.put(starlarkOption.getKey().toString(), starlarkOption.getValue(), null, mutability);
      }

      if (!remainingInputs.isEmpty()) {
        throw new EvalException(
            starlarkTransition.getLocationForErrorReporting(),
            String.format(
                "transition inputs [%s] do not correspond to valid settings",
                Joiner.on(", ").join(remainingInputs)));
      }

      return dict;
    }
  }

  /**
   * Apply the transition dictionary to the build option, using optionInfoMap to look up the option
   * info.
   *
   * @param buildOptionsToTransition the pre-transition build options
   * @param newValues a map of option name: option value entries to override current option values
   *     in the buildOptions param
   * @param optionInfoMap a map of option name: option info for all native options that may be
   *     accessed in this transition
   * @param starlarkTransition transition object that is being applied. Used for error reporting and
   *     checking for analysis testing
   * @return the post-transition build options
   * @throws EvalException If a requested option field is inaccessible
   */
  private static BuildOptions applyTransition(
      BuildOptions buildOptionsToTransition,
      Map<String, Object> newValues,
      Map<String, OptionInfo> optionInfoMap,
      StarlarkDefinedConfigTransition starlarkTransition)
      throws EvalException {
    BuildOptions buildOptions = buildOptionsToTransition.clone();
    for (Map.Entry<String, Object> entry : newValues.entrySet()) {
      String optionName = entry.getKey();
      Object optionValue = entry.getValue();

      if (!optionName.startsWith(COMMAND_LINE_OPTION_PREFIX)) {
        buildOptions =
            BuildOptions.builder()
                .merge(buildOptions)
                .addStarlarkOption(Label.parseAbsoluteUnchecked(optionName), optionValue)
                .build();
      } else {
        optionName = optionName.substring(COMMAND_LINE_OPTION_PREFIX.length());

        // Convert NoneType to null.
        if (optionValue instanceof NoneType) {
          optionValue = null;
        }
        try {
          if (!optionInfoMap.containsKey(optionName)) {
            throw new EvalException(
                starlarkTransition.getLocationForErrorReporting(),
                String.format(
                    "transition output '%s' does not correspond to a valid setting",
                    entry.getKey()));
          }

          OptionInfo optionInfo = optionInfoMap.get(optionName);
          OptionDefinition def = optionInfo.getDefinition();
          Field field = def.getField();
          FragmentOptions options = buildOptions.get(optionInfo.getOptionClass());

          if (!def.allowsMultiple()) {
            if (optionValue == null || def.getType().isInstance(optionValue)) {
              field.set(options, optionValue);
            } else if (optionValue instanceof String) {
              field.set(options, def.getConverter().convert((String) optionValue));
            } else {
              throw new EvalException(
                  starlarkTransition.getLocationForErrorReporting(),
                  "Invalid value type for option '" + optionName + "'");
            }
          } else {
            SkylarkList rawValues =
                optionValue instanceof SkylarkList
                    ? (SkylarkList) optionValue
                    : SkylarkList.createImmutable(Collections.singletonList(optionValue));
            List<Object> allValues = new ArrayList<>(rawValues.size());
            for (Object singleValue : rawValues) {
              if (singleValue instanceof String) {
                allValues.add(def.getConverter().convert((String) singleValue));
              } else {
                allValues.add(singleValue);
              }
            }
            field.set(options, ImmutableList.copyOf(allValues));
          }
        } catch (IllegalAccessException e) {
          throw new EvalException(
              starlarkTransition.getLocationForErrorReporting(),
              "IllegalAccess for option " + optionName + ": " + e.getMessage());
        } catch (OptionsParsingException e) {
          throw new EvalException(
              starlarkTransition.getLocationForErrorReporting(),
              "OptionsParsingError for option '" + optionName + "': " + e.getMessage());
        }
      }
    }

    BuildConfiguration.Options buildConfigOptions;
    buildConfigOptions = buildOptions.get(BuildConfiguration.Options.class);

    if (starlarkTransition.isForAnalysisTesting()) {
      buildConfigOptions.evaluatingForAnalysisTest = true;
    }
    updateOutputDirectoryNameFragment(buildConfigOptions, newValues);

    return buildOptions;
  }

  /**
   * Compute the output directory name fragment corresponding to the transition, and append it to
   * the existing name fragment in buildConfigOptions.
   *
   * @throws IllegalStateException If MD5 support is not available
   */
  private static void updateOutputDirectoryNameFragment(
      BuildConfiguration.Options buildConfigOptions, Map<String, Object> transition) {
    String transitionString = "";
    for (Map.Entry<String, Object> entry : transition.entrySet()) {
      transitionString += entry.getKey() + ":";
      if (entry.getValue() != null) {
        transitionString += entry.getValue() + "@";
      }
    }

    // TODO(waltl): for transitions that don't read settings, it is possible to precompute and
    // reuse the MD5 digest and even the transition itself.
    try {
      byte[] bytes = transitionString.getBytes(US_ASCII);
      MessageDigest md = MessageDigest.getInstance("MD5");
      byte[] digest = md.digest(bytes);
      String hexDigest = BaseEncoding.base16().lowerCase().encode(digest);

      if (buildConfigOptions.transitionDirectoryNameFragment == null) {
        buildConfigOptions.transitionDirectoryNameFragment = hexDigest;
      } else {
        buildConfigOptions.transitionDirectoryNameFragment += "-" + hexDigest;
      }
    } catch (NoSuchAlgorithmException e) {
      throw new IllegalStateException("MD5 not available", e);
    }
  }

  /** Stores option info useful to a FunctionSplitTransition. */
  static class OptionInfo {
    private final Class<? extends FragmentOptions> optionClass;
    private final OptionDefinition definition;

    public OptionInfo(Class<? extends FragmentOptions> optionClass, OptionDefinition definition) {
      this.optionClass = optionClass;
      this.definition = definition;
    }

    Class<? extends FragmentOptions> getOptionClass() {
      return optionClass;
    }

    OptionDefinition getDefinition() {
      return definition;
    }
  }
}
