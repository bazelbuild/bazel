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

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.StarlarkDefinedConfigTransition;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.syntax.Dict;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Mutability;
import com.google.devtools.build.lib.syntax.NoneType;
import com.google.devtools.build.lib.syntax.Starlark;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.common.options.OptionDefinition;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;
import java.lang.reflect.Field;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;

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
    Dict<String, Object> settings = buildSettings(buildOptions, optionInfoMap, starlarkTransition);

    ImmutableList.Builder<BuildOptions> splitBuildOptions = ImmutableList.builder();

    ImmutableList<Map<String, Object>> transitions =
        starlarkTransition.evaluate(settings, attrObject);
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
  static Dict<String, Object> buildSettings(
      BuildOptions buildOptions,
      Map<String, OptionInfo> optionInfoMap,
      StarlarkDefinedConfigTransition starlarkTransition)
      throws EvalException {
    LinkedHashSet<String> remainingInputs = Sets.newLinkedHashSet(starlarkTransition.getInputs());

    try (Mutability mutability = Mutability.create("build_settings")) {
      Dict<String, Object> dict = Dict.of(mutability);

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

          dict.put(optionKey, optionValue == null ? Starlark.NONE : optionValue, /*loc=*/ null);
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
        dict.put(starlarkOption.getKey().toString(), starlarkOption.getValue(), /*loc=*/ null);
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
   * @param optionInfoMap a map of all native options (name -> OptionInfo) present in {@code
   *     toOptions}.
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
    HashMap<String, Object> convertedNewValues = new HashMap<>();
    for (Map.Entry<String, Object> entry : newValues.entrySet()) {
      String optionName = entry.getKey();
      Object optionValue = entry.getValue();

      if (!optionName.startsWith(COMMAND_LINE_OPTION_PREFIX)) {
        buildOptions =
            BuildOptions.builder()
                .merge(buildOptions)
                .addStarlarkOption(Label.parseAbsoluteUnchecked(optionName), optionValue)
                .build();
        convertedNewValues.put(optionName, optionValue);
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
          if (optionValue == null || def.getType().isInstance(optionValue)) {
            field.set(options, optionValue);
            convertedNewValues.put(entry.getKey(), optionValue);
          } else if (optionValue instanceof String) {
            Object convertedValue = def.getConverter().convert((String) optionValue);
            field.set(options, convertedValue);
            convertedNewValues.put(entry.getKey(), convertedValue);
          } else {
            throw new EvalException(
                starlarkTransition.getLocationForErrorReporting(),
                "Invalid value type for option '" + optionName + "'");
          }
        } catch (IllegalArgumentException e) {
          throw new EvalException(
              starlarkTransition.getLocationForErrorReporting(),
              "IllegalArgumentError for option '" + optionName + "': " + e.getMessage());
        } catch (IllegalAccessException e) {
          throw new RuntimeException(
              "IllegalAccess for option " + optionName + ": " + e.getMessage());
        } catch (OptionsParsingException e) {
          throw new EvalException(
              starlarkTransition.getLocationForErrorReporting(),
              "OptionsParsingError for option '" + optionName + "': " + e.getMessage());
        }
      }
    }

    CoreOptions buildConfigOptions;
    buildConfigOptions = buildOptions.get(CoreOptions.class);

    if (starlarkTransition.isForAnalysisTesting()) {
      buildConfigOptions.evaluatingForAnalysisTest = true;
    }
    updateOutputDirectoryNameFragment(convertedNewValues.keySet(), optionInfoMap, buildOptions);

    return buildOptions;
  }

  /**
   * Compute the output directory name fragment corresponding to the new BuildOptions based on (1)
   * the names and values of all native options previously transitioned anywhere in the build by
   * starlark options, (2) names and values of all entries in the starlark options map.
   *
   * @param changedOptions the names of all options changed by this transition in label form e.g.
   *     "//command_line_option:cpu" for native options and "//myapp:foo" for starlark options.
   * @param optionInfoMap a map of all native options (name -> OptionInfo) present in {@code
   *     toOptions}.
   * @param toOptions the newly transitioned {@link BuildOptions} for which we need to updated
   *     {@code transitionDirectoryNameFragment} and {@code affectedByStarlarkTransition}.
   */
  // TODO(bazel-team): This hashes different forms of equivalent values differently though they
  // should be the same configuration. Starlark transitions are flexible about the values they
  // take (e.g. bool-typed options can take 0/1, True/False, "0"/"1", or "True"/"False") which
  // makes it so that two configurations that are the same in value may hash differently.
  private static void updateOutputDirectoryNameFragment(
      Set<String> changedOptions, Map<String, OptionInfo> optionInfoMap, BuildOptions toOptions) {
    CoreOptions buildConfigOptions = toOptions.get(CoreOptions.class);
    Set<String> updatedAffectedByStarlarkTransition =
        new TreeSet<>(buildConfigOptions.affectedByStarlarkTransition);
    // Add newly changed native options to overall list of changed native options
    for (String option : changedOptions) {
      if (option.startsWith(COMMAND_LINE_OPTION_PREFIX)) {
        updatedAffectedByStarlarkTransition.add(
            option.substring(COMMAND_LINE_OPTION_PREFIX.length()));
      }
    }
    buildConfigOptions.affectedByStarlarkTransition =
        ImmutableList.sortedCopyOf(updatedAffectedByStarlarkTransition);

    // hash all relevant native option values;
    TreeMap<String, Object> toHash = new TreeMap<>();
    for (String nativeOption : updatedAffectedByStarlarkTransition) {
      Object value;
      try {
        value =
            optionInfoMap
                .get(nativeOption)
                .getDefinition()
                .getField()
                .get(toOptions.get(optionInfoMap.get(nativeOption).getOptionClass()));
      } catch (IllegalAccessException e) {
        throw new RuntimeException(
            "IllegalAccess for option " + nativeOption + ": " + e.getMessage());
      }
      toHash.put(nativeOption, value);
    }

    // hash all starlark options in map.
    toOptions.getStarlarkOptions().forEach((opt, value) -> toHash.put(opt.toString(), value));

    Fingerprint fp = new Fingerprint();
    for (Map.Entry<String, Object> singleOptionAndValue : toHash.entrySet()) {
      String toAdd = singleOptionAndValue.getKey() + "=" + singleOptionAndValue.getValue();
      fp.addString(toAdd);
    }
    // Make this hash somewhat recognizable
    buildConfigOptions.transitionDirectoryNameFragment = "ST-" + fp.hexDigestAndReset();
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
