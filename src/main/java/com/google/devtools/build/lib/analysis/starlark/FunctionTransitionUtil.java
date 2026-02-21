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

package com.google.devtools.build.lib.analysis.starlark;

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.devtools.build.lib.analysis.config.transitions.ConfigurationTransition.PATCH_TRANSITION_KEY;
import static com.google.devtools.build.lib.cmdline.LabelConstants.COMMAND_LINE_OPTION_PACKAGE_IDENTIFIER;
import static com.google.devtools.build.lib.cmdline.LabelConstants.COMMAND_LINE_OPTION_PREFIX;
import static java.util.stream.Collectors.joining;

import com.google.common.base.Joiner;
import com.google.common.base.Predicate;
import com.google.common.base.VerifyException;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Sets;
import com.google.common.collect.Sets.SetView;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.CoreOptionConverters.CustomFlagConverter;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.OptionInfo;
import com.google.devtools.build.lib.analysis.config.Scope;
import com.google.devtools.build.lib.analysis.config.StarlarkDefinedConfigTransition;
import com.google.devtools.build.lib.analysis.config.StarlarkDefinedConfigTransition.ValidationException;
import com.google.devtools.build.lib.analysis.test.TestConfiguration.TestOptions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.common.options.OptionDefinition;
import com.google.devtools.common.options.OptionMetadataTag;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.Collection;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;
import javax.annotation.Nullable;
import net.starlark.java.eval.NoneType;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkInt;

/**
 * Utility class for common work done across {@link StarlarkAttributeTransitionProvider} and {@link
 * StarlarkRuleTransitionProvider}.
 */
public final class FunctionTransitionUtil {

  private static final Predicate<String> IS_NATIVE_OPTION =
      setting -> setting.startsWith(LabelConstants.COMMAND_LINE_OPTION_PREFIX);

  /**
   * Figure out what build settings the given transition changes and apply those changes to the
   * incoming {@link BuildOptions}. For native options, this involves a preprocess step of
   * converting options to their "command line form".
   *
   * <p>Also perform validation on the inputs and outputs:
   *
   * <ol>
   *   <li>Ensure that all native input options exist
   *   <li>Ensure that all native output options exist
   *   <li>Ensure that there are no attempts to update the {@code --define} option.
   *   <li>Ensure that no {@link OptionMetadataTag#NON_CONFIGURABLE non-configurable} native options
   *       are updated.
   *   <li>Ensure that transitions output all of the declared options.
   * </ol>
   *
   * @param fromOptions the pre-transition build options
   * @param starlarkTransition the transition to apply
   * @param attrObject the attributes of the rule to which this transition is attached
   * @return the post-transition build options, or null if errors were reported to handler.
   */
  @Nullable
  static ImmutableMap<String, BuildOptions> applyAndValidate(
      BuildOptions fromOptions,
      StarlarkDefinedConfigTransition starlarkTransition,
      boolean allowNonConfigurableFlagChanges,
      boolean isExecTransition,
      StructImpl attrObject,
      EventHandler handler,
      @Nullable StarlarkBuildSettingsDetailsValue scopeDetails)
      throws InterruptedException {
    try {
      // TODO(waltl): Consider building this once and using it across different split transitions,
      // or reusing BuildOptionDetails.
      ImmutableMap<String, OptionInfo> optionInfoMap = OptionInfo.buildMapFrom(fromOptions);
      ImmutableMap<String, Label> flagsAliases;
      if (isExecTransition) {
        // Ignore flag aliases for exec transitions. Starlark flags will provide their exec
        // transition semantics in the flag definition.
        flagsAliases = ImmutableMap.of();
      } else {
        flagsAliases = fromOptions.get(CoreOptions.class).getCommandLineFlagAliases();
      }

      validateInputOptions(
          starlarkTransition.getInputs(),
          allowNonConfigurableFlagChanges,
          optionInfoMap,
          flagsAliases);
      validateOutputOptions(
          starlarkTransition.getOutputs(),
          allowNonConfigurableFlagChanges,
          optionInfoMap,
          flagsAliases);

      ImmutableMap<String, Object> settings =
          buildSettings(fromOptions, optionInfoMap, flagsAliases, starlarkTransition);

      ImmutableMap.Builder<String, BuildOptions> splitBuildOptions = ImmutableMap.builder();

      // For anything except the exec transition this is just fromOptions. See maybeGetExecDefaults
      // for why the exec transition is different.
      BuildOptions baselineToOptions =
          maybeGetExecDefaults(fromOptions, starlarkTransition, scopeDetails);

      ImmutableMap<String, Map<Label, Object>> transitions =
          starlarkTransition.evaluate(settings, attrObject, optionInfoMap, handler);
      if (transitions == null) {
        return null; // errors reported to handler
      } else if (transitions.isEmpty()) {
        // The transition produced a no-op.
        return ImmutableMap.of(PATCH_TRANSITION_KEY, baselineToOptions);
      }

      for (Map.Entry<String, Map<Label, Object>> entry : transitions.entrySet()) {
        Map<Label, Object> newValues =
            handleImplicitPlatformChange(
                baselineToOptions, applyStarlarkFlagsAliases(flagsAliases, entry.getValue()));

        BuildOptions transitionedOptions =
            applyTransition(baselineToOptions, newValues, optionInfoMap, starlarkTransition);
        splitBuildOptions.put(entry.getKey(), transitionedOptions);
      }
      return splitBuildOptions.buildOrThrow();

    } catch (ValidationException ex) {
      handler.handle(Event.error(starlarkTransition.getLocation(), ex.getMessage()));
      return null;
    }
  }

  /**
   * For all transitions except the exec transition, returns {@code fromOptions}.
   *
   * <p>The exec transition is special: any options not explicitly set by the transition take their
   * defaults, not {@code fromOptions}'s values. This method adjusts the baseline options
   * accordingly.
   *
   * <p>The exec transition's full sequence is:
   *
   * <ol>
   *   <li>The transition's Starlark function runs over {@code fromOptions}: {@code
   *       {"//command_line_option:foo": settings["//command_line_option:foo"}} sets {@code foo} to
   *       {@code fromOptions}'s value (i.e. propagates from the source config)
   *   <li>This method constructs a {@link BuildOptions} default value (which doesn't inherit from
   *       the source config)
   *   <li>{@link #applyTransition} creates final options: use whatever options the Starlark logic
   *       set (which may propagate from the source config). For all other options, use default
   *       values
   *       <p>See {@link com.google.devtools.build.lib.analysis.config.ExecutionTransitionFactory}.
   */
  private static BuildOptions maybeGetExecDefaults(
      BuildOptions fromOptions,
      StarlarkDefinedConfigTransition starlarkTransition,
      @Nullable StarlarkBuildSettingsDetailsValue scopeDetails) {
    if (starlarkTransition == null || !starlarkTransition.isExecTransition()) {
      // Not an exec transition: the baseline options are just the input options.
      return fromOptions;
    }
    BuildOptions.Builder defaultBuilder = BuildOptions.builder();
    // Get the defaults:
    fromOptions.getNativeOptions().forEach(o -> defaultBuilder.addFragmentOptions(o.getDefault()));
    // Propagate Starlark options from the source config if allowed.
    defaultBuilder.addStarlarkOptions(
        getExecPropagatingStarlarkFlags(
            fromOptions.getStarlarkOptions(), fromOptions, scopeDetails));
    // Hard-code TestConfiguration for now, which clones the source options.
    // TODO(b/295936652): handle this directly in Starlark. This has two complications:
    //  1: --trim_test_configuration means the flags may not exist. Starlark logic needs to handle
    //     that possibility.
    //  2: --runs_per_test has a non-Starlark readable type.
    var testOptions = fromOptions.get(TestOptions.class);
    if (testOptions != null) {
      defaultBuilder.removeFragmentOptions(TestOptions.class);
      defaultBuilder.addFragmentOptions(testOptions);
    }
    BuildOptions ans = defaultBuilder.build();
    if (fromOptions.get(CoreOptions.class).excludeDefinesFromExecConfig) {
      ans.get(CoreOptions.class).commandLineBuildVariables =
          fromOptions.get(CoreOptions.class).commandLineBuildVariables.stream()
              .filter(
                  (define) ->
                      fromOptions
                          .get(CoreOptions.class)
                          .customFlagsToPropagate
                          .contains(define.getKey()))
              .collect(toImmutableList());
    } else {
      ans.get(CoreOptions.class).commandLineBuildVariables =
          fromOptions.get(CoreOptions.class).commandLineBuildVariables;
    }
    return ans;
  }

  /**
   * Filters a map of Starlark flag <Label, value> pairs to those that should propagate from the
   * target configuration to exec configuration.
   *
   * @param scopeDetails scope info loaded from Skyframe, or null if there are no starlark flags
   */
  private static ImmutableMap<Label, Object> getExecPropagatingStarlarkFlags(
      Map<Label, Object> starlarkOptions,
      BuildOptions options,
      @Nullable StarlarkBuildSettingsDetailsValue scopeDetails) {
    if (starlarkOptions.isEmpty()) {
      return ImmutableMap.of();
    }
    // Look up scope type for a flag via scopeDetails (from Skyframe) rather than BuildOptions.
    // The scopeDetails map is keyed by actual (non-alias) label. Flags in the config may be
    // aliases, so we resolve through the alias table.
    ImmutableMap<Label, Scope.ScopeType> scopeTypeMap =
        scopeDetails != null ? scopeDetails.buildSettingToScopeType() : ImmutableMap.of();
    ImmutableMap<Label, Label> aliasToActual =
        scopeDetails != null ? scopeDetails.aliasToActual() : ImmutableMap.of();
    ImmutableMap<Label, Object> onLeaveScopeValues =
        scopeDetails != null ? scopeDetails.buildSettingToOnLeaveScopeValue() : ImmutableMap.of();

    // Collect flags that are referenced by custom exec scopes (exec:--<flag>). These flags
    // should be treated as having TARGET scope so they don't propagate to exec config.
    Set<Label> customExecReferencedFlags = new HashSet<>();
    for (Scope.ScopeType scope : scopeTypeMap.values()) {
      if (scope.scopeType().startsWith(Scope.CUSTOM_EXEC_SCOPE_PREFIX)) {
        customExecReferencedFlags.add(
            Label.parseCanonicalUnchecked(scope.scopeType().substring(7)));
      }
    }

    if (!options.get(CoreOptions.class).excludeStarlarkFlagsFromExecConfig) {
      // Starlark flags propagate to exec by default. This can only be changed by a flag explicitly
      // setting "scope = 'target'" or being referenced by a custom exec scope.
      return starlarkOptions.entrySet().stream()
          .filter(
              entry -> {
                Label actual = aliasToActual.getOrDefault(entry.getKey(), entry.getKey());
                if (customExecReferencedFlags.contains(actual)) {
                  return false; // Treated as TARGET scope - don't propagate.
                }
                Scope.ScopeType scope = scopeTypeMap.get(actual);
                String scopeType = scope != null ? scope.scopeType() : Scope.ScopeType.DEFAULT;
                return scopeType.equals(Scope.ScopeType.UNIVERSAL)
                    || scopeType.equals(Scope.ScopeType.DEFAULT);
              })
          .collect(ImmutableMap.toImmutableMap(Map.Entry::getKey, Map.Entry::getValue));
    }

    // --incompatible_exclude_starlark_flags_from_exec_config=True: Starlark flags don't propagate
    // to the exec config by default. This can be overridden by a flag setting "scope = 'universal'"
    // or --experimental_propagate_custom_flag. If both are set, the flag setting takes precedence.
    Map<Boolean, List<String>> partitioned =
        options.get(CoreOptions.class).customFlagsToPropagate.stream()
            .collect(
                Collectors.partitioningBy(f -> f.endsWith(CustomFlagConverter.SUBPACKAGES_SUFFIX)));
    // Holds --experimental_propagate_custom_flag patterns=//pkg/... patterns. These are rare.
    List<String> customPropagatingFlagPatterns = partitioned.get(true);
    // Flags that should propagate according to --experimental_propagate_custom_flag.
    Set<String> customPropagatingFlags = new HashSet<>(partitioned.get(false));

    ImmutableMap.Builder<Label, Object> ans = ImmutableMap.builder();
    for (Map.Entry<Label, Object> entry : starlarkOptions.entrySet()) {
      Label actual = aliasToActual.getOrDefault(entry.getKey(), entry.getKey());
      Scope.ScopeType scope = scopeTypeMap.get(actual);
      // Flags referenced by custom exec scopes are treated as TARGET.
      String scopeType =
          customExecReferencedFlags.contains(actual)
              ? Scope.ScopeType.TARGET
              : (scope != null ? scope.scopeType() : Scope.ScopeType.DEFAULT);
      if (scopeType.equals(Scope.ScopeType.UNIVERSAL)) {
        ans.put(entry);
      } else if (scopeType.equals(Scope.ScopeType.TARGET)) {
        Object onLeaveScopeValue = onLeaveScopeValues.get(actual);
        if (onLeaveScopeValue != null) {
          // if on_leave_scope is set, propagate to exec config with this value.
          ans.put(entry.getKey(), onLeaveScopeValue);
        }
        // else Don't propagate this flag.
      } else if (customPropagatingFlags.contains(entry.getKey().getUnambiguousCanonicalForm())) {
        ans.put(entry);
      } else if (customPropagatingFlagPatterns.stream()
          .anyMatch(
              pattern ->
                  entry
                      .getKey()
                      .getUnambiguousCanonicalForm()
                      .startsWith(
                          pattern.substring(
                              0, pattern.lastIndexOf(CustomFlagConverter.SUBPACKAGES_SUFFIX))))) {
        ans.put(entry);
      } else if (scopeType.startsWith(Scope.CUSTOM_EXEC_SCOPE_PREFIX)) {
        Label anotherFlag = Label.parseCanonicalUnchecked(scopeType.substring(7));
        if (starlarkOptions.containsKey(anotherFlag)) {
          ans.put(entry.getKey(), starlarkOptions.get(anotherFlag));
        } else {
          boolean found = false;
          for (FragmentOptions fragment : options.getNativeOptions()) {
            Map<String, Object> nativeOptions = fragment.asMap();
            if (nativeOptions.containsKey(anotherFlag.getUnambiguousCanonicalForm())) {
              ans.put(entry.getKey(), nativeOptions.get(anotherFlag.getUnambiguousCanonicalForm()));
              found = true;
              break;
            }
          }
          // if the flag is not found in both starlark and the native options, it's an error.
          if (!found) {
            throw new IllegalStateException(
                "Flag "
                    + anotherFlag
                    + " is not found in the starlark options or native options. It should be one of"
                    + " them.");
          }
        }
      }
    }

    return ans.buildOrThrow();
  }

  private static final Label CPU_OPTION =
      Label.createUnvalidated(LabelConstants.COMMAND_LINE_OPTION_PACKAGE_IDENTIFIER, "cpu");
  private static final Label PLATFORMS_OPTION =
      Label.createUnvalidated(LabelConstants.COMMAND_LINE_OPTION_PACKAGE_IDENTIFIER, "platforms");

  /**
   * If the transition changes --cpu but not --platforms, clear out --platforms.
   *
   * <p>Purpose:
   *
   * <ol>
   *   <li>A platform mapping sets --cpu=foo when --platforms=foo.
   *   <li>A transition sets --cpu=bar.
   *   <li>Because --platforms=foo, the platform mapping kicks in to set --cpu back to foo.
   *   <li>Result: the mapping accidentally overrides the transition
   * </ol>
   *
   * <p>Transitions can also explicitly set --platforms to be clear what platform they set.
   *
   * <p>Platform mappings: https://bazel.build/concepts/platforms-intro#platform-mappings.
   */
  private static Map<Label, Object> handleImplicitPlatformChange(
      BuildOptions options, Map<Label, Object> rawTransitionOutput) {
    Object newCpu = rawTransitionOutput.get(CPU_OPTION);
    if (newCpu == null || newCpu.equals(options.get(CoreOptions.class).cpu)) {
      // No effective change to --cpu, so no need to prevent the platform mapping from resetting it.
      return rawTransitionOutput;
    }
    if (rawTransitionOutput.containsKey(PLATFORMS_OPTION)) {
      // Explicitly setting --platforms overrides the implicit clearing.
      return rawTransitionOutput;
    }
    return ImmutableMap.<Label, Object>builder()
        .putAll(rawTransitionOutput)
        .put(PLATFORMS_OPTION, ImmutableList.<Label>of())
        .buildOrThrow();
  }

  /** Set the Starlark flag value to the value of its alias. */
  private static Map<Label, Object> applyStarlarkFlagsAliases(
      ImmutableMap<String, Label> flagsAliases, Map<Label, Object> rawTransitionOutput)
      throws ValidationException {
    if (flagsAliases.isEmpty()) {
      return rawTransitionOutput;
    }

    LinkedHashMap<Label, Object> result = new LinkedHashMap<>(rawTransitionOutput);

    for (Map.Entry<String, Label> flagAlias : flagsAliases.entrySet()) {
      Label nativeFlag =
          Label.createUnvalidated(
              LabelConstants.COMMAND_LINE_OPTION_PACKAGE_IDENTIFIER, flagAlias.getKey());
      Label starlarkFlag = flagAlias.getValue();

      if (rawTransitionOutput.containsKey(starlarkFlag)
          && rawTransitionOutput.containsKey(nativeFlag)) {
        if (!rawTransitionOutput.get(starlarkFlag).equals(rawTransitionOutput.get(nativeFlag))) {
          throw new ValidationException(
              String.format(
                  "Starlark flag '%s' and its alias '%s' have different values: '%s' and '%s'",
                  starlarkFlag,
                  nativeFlag,
                  rawTransitionOutput.get(starlarkFlag),
                  rawTransitionOutput.get(nativeFlag)));
        }
      }

      if (rawTransitionOutput.containsKey(nativeFlag)) {
        // Add the starlark flag to the result, using the value of the alias.
        result.put(starlarkFlag, rawTransitionOutput.get(nativeFlag));
        // Remove the entry of the alias.
        result.remove(nativeFlag);
      }
    }
    return result;
  }

  private static boolean isNativeOptionValid(
      ImmutableMap<String, OptionInfo> optionInfoMap,
      ImmutableMap<String, Label> flagsAliases,
      String optionName) {
    // Make sure the option exists, or it is an alias.
    return optionInfoMap.containsKey(optionName) || flagsAliases.containsKey(optionName);
  }

  /**
   * Check if a native option is non-configurable.
   *
   * @return whether or not the option is non-configurable
   * @throws VerifyException if the option does not exist
   */
  private static boolean isNativeOptionNonConfigurable(
      ImmutableMap<String, OptionInfo> optionInfoMap,
      ImmutableMap<String, Label> flagsAliases,
      String optionName) {
    OptionInfo optionInfo = optionInfoMap.get(optionName);
    if (optionInfo == null) {
      if (flagsAliases.containsKey(optionName)) {
        // All aliases are configurable (for now).
        return false;
      }
      throw new VerifyException(
          "Cannot check if option %s is non-configurable: it does not exist".formatted(optionName));
    }
    return optionInfo.hasOptionMetadataTag(OptionMetadataTag.NON_CONFIGURABLE);
  }

  private static void validateInputOptions(
      ImmutableList<String> options,
      boolean allowNonConfigurableFlagChanges,
      ImmutableMap<String, OptionInfo> optionInfoMap,
      ImmutableMap<String, Label> flagsAliases)
      throws ValidationException {
    checkForInvalidNativeOptions(
        /* transitionParameterType= */ "inputs", options, optionInfoMap, flagsAliases);

    checkForNonConfigurableOptions(
        /* transitionParameterType= */ "inputs",
        options,
        allowNonConfigurableFlagChanges,
        optionInfoMap,
        flagsAliases);
  }

  private static void validateOutputOptions(
      Collection<String> options,
      boolean allowNonConfigurableFlagChanges,
      ImmutableMap<String, OptionInfo> optionInfoMap,
      ImmutableMap<String, Label> flagsAliases)
      throws ValidationException {
    if (options.contains("//command_line_option:define")) {
      throw new ValidationException(
          "Starlark transition on --define not supported - try using build settings"
              + " (https://bazel.build/rules/config#user-defined-build-settings).");
    }

    // TODO: blaze-configurability - Move the checks for incompatible and experimental flags to here
    // (currently in ConfigGlobalLibrary.validateBuildSettingKeys).

    checkForInvalidNativeOptions(
        /* transitionParameterType= */ "outputs", options, optionInfoMap, flagsAliases);

    checkForNonConfigurableOptions(
        /* transitionParameterType= */ "outputs",
        options,
        allowNonConfigurableFlagChanges,
        optionInfoMap,
        flagsAliases);
  }

  private static void checkForInvalidNativeOptions(
      String transitionParameterType,
      Collection<String> options,
      ImmutableMap<String, OptionInfo> optionInfoMap,
      ImmutableMap<String, Label> flagsAliases)
      throws ValidationException {
    ImmutableList<String> invalidNativeOptions =
        options.stream()
            .filter(IS_NATIVE_OPTION)
            .filter(
                option ->
                    !isNativeOptionValid(
                        optionInfoMap,
                        flagsAliases,
                        option.substring(COMMAND_LINE_OPTION_PREFIX.length())))
            .collect(toImmutableList());
    if (!invalidNativeOptions.isEmpty()) {
      throw ValidationException.format(
          "transition %s [%s] do not correspond to valid settings",
          transitionParameterType, Joiner.on(", ").join(invalidNativeOptions));
    }
  }

  private static void checkForNonConfigurableOptions(
      String transitionParameterType,
      Collection<String> options,
      boolean allowNonConfigurableFlagChanges,
      ImmutableMap<String, OptionInfo> optionInfoMap,
      ImmutableMap<String, Label> flagsAliases)
      throws ValidationException {
    if (!allowNonConfigurableFlagChanges) {
      ImmutableList<String> nonConfigurableNativeOptions =
          options.stream()
              .filter(IS_NATIVE_OPTION)
              .filter(
                  option ->
                      isNativeOptionNonConfigurable(
                          optionInfoMap,
                          flagsAliases,
                          option.substring(COMMAND_LINE_OPTION_PREFIX.length())))
              .collect(toImmutableList());
      if (!nonConfigurableNativeOptions.isEmpty()) {
        throw ValidationException.format(
            "transition %s [%s] cannot be changed: they are non-configurable",
            transitionParameterType, Joiner.on(", ").join(nonConfigurableNativeOptions));
      }
    }
  }

  /**
   * Return an ImmutableMap containing only BuildOptions explicitly registered as transition inputs.
   *
   * <p>nulls are converted to Starlark.NONE but no other conversions are done.
   *
   * @throws IllegalArgumentException If the method is unable to look up the value in buildOptions
   *     corresponding to an entry in optionInfoMap
   * @throws RuntimeException If the field corresponding to an option value in buildOptions is
   *     inaccessible due to Java language access control, or if an option name is an invalid key to
   *     the Starlark dictionary
   * @throws ValidationException if any of the specified transition inputs do not correspond to a
   *     valid build setting
   */
  private static ImmutableMap<String, Object> buildSettings(
      BuildOptions buildOptions,
      Map<String, OptionInfo> optionInfoMap,
      ImmutableMap<String, Label> flagsAliases,
      StarlarkDefinedConfigTransition starlarkTransition)
      throws ValidationException {
    ImmutableMap<Label, String> inputsCanonicalizedToGiven =
        starlarkTransition.getInputsCanonicalizedToGiven();

    ImmutableMap.Builder<String, Object> optionsBuilder = ImmutableMap.builder();

    // Convert the canonical form to the user requested form that they expect to see.
    inputsCanonicalizedToGiven.forEach(
        (canonical, given) -> {
          if (canonical.getPackageIdentifier().equals(COMMAND_LINE_OPTION_PACKAGE_IDENTIFIER)) {
            findNativeOptionValue(buildOptions, optionInfoMap, flagsAliases, canonical)
                .ifPresent(optionValue -> optionsBuilder.put(given, optionValue));
          } else {
            Object optionValue = findStarlarkOptionValue(buildOptions, canonical);
            optionsBuilder.put(given, optionValue);
          }
        });

    ImmutableMap<String, Object> result = optionsBuilder.buildOrThrow();
    SetView<String> remainingInputs =
        Sets.difference(ImmutableSet.copyOf(inputsCanonicalizedToGiven.values()), result.keySet());
    if (!remainingInputs.isEmpty()) {
      throw ValidationException.format(
          "transition inputs [%s] do not correspond to valid settings",
          Joiner.on(", ").join(remainingInputs));
    }

    return result;
  }

  private static Optional<Object> findNativeOptionValue(
      BuildOptions buildOptions,
      Map<String, OptionInfo> optionInfoMap,
      ImmutableMap<String, Label> flagsAliases,
      Label setting) {
    String optionName = setting.getName();
    if (flagsAliases.containsKey(optionName)) {
      // If the setting is an alias to a starlark option, use the starlark option value.
      return Optional.of(findStarlarkOptionValue(buildOptions, flagsAliases.get(optionName)));
    }

    if (!optionInfoMap.containsKey(optionName)) {
      return Optional.empty();
    }
    OptionInfo optionInfo = optionInfoMap.get(optionName);
    FragmentOptions options = buildOptions.get(optionInfo.getOptionClass());
    // Get the raw value to avoid the default handling for null values.
    Object optionValue = optionInfo.getDefinition().getRawValue(options);
    // convert nulls here b/c ImmutableMap bans null values
    return Optional.of(optionValue == null ? Starlark.NONE : optionValue);
  }

  private static Object findStarlarkOptionValue(BuildOptions buildOptions, Label setting) {
    return buildOptions.getStarlarkOptions().get(setting);
  }

  /**
   * Apply the transition dictionary to the build option, using optionInfoMap to look up the option
   * info.
   *
   * @param fromOptions the pre-transition build options
   * @param newValues a map of option Label: option value entries to override current option values
   *     in the buildOptions param
   * @param optionInfoMap a map of all native options (name -> OptionInfo) present in {@code
   *     toOptions}.
   * @param starlarkTransition transition object that is being applied. Used for error reporting and
   *     checking for analysis testing
   * @return the post-transition build options
   * @throws ValidationException If a requested option field is inaccessible
   */
  private static BuildOptions applyTransition(
      BuildOptions fromOptions,
      Map<Label, Object> newValues,
      Map<String, OptionInfo> optionInfoMap,
      StarlarkDefinedConfigTransition starlarkTransition)
      throws ValidationException {
    // toOptions being null means the transition hasn't changed anything. We avoid preemptively
    // cloning it from fromOptions since options cloning is an expensive operation.
    BuildOptions toOptions = null;
    // Starlark options that are different after this transition. We collect all of them, then clone
    // the build options once with all cumulative changes. Native option changes, in contrast, are
    // set directly in the BuildOptions instance. The former approach is preferred since it makes
    // BuildOptions objects more immutable. Native options use the latter approach for legacy
    // reasons. While not preferred, direct mutation doesn't require expensive cloning.
    Map<Label, Object> changedStarlarkOptions = new LinkedHashMap<>();
    for (Map.Entry<Label, Object> entry : newValues.entrySet()) {
      Label optionKey = entry.getKey();
      Object optionValue = entry.getValue();

      if (!optionKey
          .getPackageIdentifier()
          .equals(LabelConstants.COMMAND_LINE_OPTION_PACKAGE_IDENTIFIER)) {
        // The transition changes a Starlark option.
        Object oldValue = fromOptions.getStarlarkOptions().get(optionKey);
        if (oldValue instanceof Label) {
          // If this is a label-typed build setting, we need to convert the provided new value into
          // a Label object.
          if (optionValue instanceof String) {
            try {
              optionValue =
                  Label.parseWithPackageContext(
                      (String) optionValue, starlarkTransition.getPackageContext());
            } catch (LabelSyntaxException e) {
              throw ValidationException.format(
                  "Error parsing value for option '%s': %s", optionKey, e.getMessage());
            }
          } else if (!(optionValue instanceof Label)) {
            throw ValidationException.format(
                "Invalid value type for option '%s': want label, got %s",
                optionKey, Starlark.type(optionValue));
          }
        } else if (oldValue instanceof Set) {
          // If this is a set-typed build setting, for backwards compatibility, if the provided
          // value is a List, we need to convert it to a Set.
          if (optionValue instanceof List<?>) {
            optionValue = ImmutableSet.copyOf((List<?>) optionValue);
          } else if (!(optionValue instanceof Set)) {
            throw ValidationException.format(
                "Invalid value type for option '%s': want set, got %s",
                optionKey, Starlark.type(optionValue));
          }
        }
        if (!Objects.equals(oldValue, optionValue)) {
          changedStarlarkOptions.put(optionKey, optionValue);
        }
      } else {
        // The transition changes a native option.
        String optionName = optionKey.getName();
        OptionInfo optionInfo = optionInfoMap.get(optionName);

        // Convert NoneType to null.
        if (optionValue instanceof NoneType) {
          optionValue = null;
        } else if (optionValue instanceof StarlarkInt starlarkInt) {
          optionValue = starlarkInt.toIntUnchecked();
        } else if (optionValue instanceof List<?>) {
          // Converting back to the Java-native type makes it easier to check if a Starlark
          // transition set the same value a native transition would. This is important for
          // ExecutionTransitionFactory#ComparingTransition.
          // TODO(b/288258583): remove this case when ComparingTransition is no longer needed for
          // debugging. Production code just iterates over the lists, which both Starlark and
          // native List types implement.
          optionValue = ImmutableList.copyOf((List<?>) optionValue);
        } else if (optionValue instanceof Map<?, ?>) {
          // TODO(b/288258583): remove this case when ComparingTransition is no longer needed for
          // debugging. See above TODO.
          optionValue = ImmutableMap.copyOf(((Map<?, ?>) optionValue));
        }
        try {
          OptionDefinition def = optionInfo.getDefinition();
          // TODO(b/153867317): check for crashing options types in this logic.
          Object convertedValue;
          if (def.getType() == List.class && optionValue instanceof List<?> optionValueAsList) {
            // This is possible with Starlark code like "{ //command_line_option:foo: ["a", "b"] }".
            // In that case def.getType() == List.class while optionValue.type == StarlarkList.
            // Unfortunately we can't check the *element* types because OptionDefinition won't tell
            // us that about def (def.getConverter() returns LabelListConverter but nowhere does it
            // mention Label.class). Worse, def.getConverter().convert takes a String input. This
            // forces us to serialize optionValue back to a scalar string to convert. There's no
            // generically safe way to do this. We convert its elements with .toString() with a ","
            // separator, which happens to work for most implementations. But that's not universally
            // guaranteed.
            if (optionValueAsList.isEmpty()) {
              convertedValue = ImmutableList.of();
            } else if (!def.allowsMultiple()) {
              convertedValue =
                  def.getConverter()
                      .convert(
                          optionValueAsList.stream()
                              .map(
                                  element ->
                                      element instanceof Label label
                                          ? label.getUnambiguousCanonicalForm()
                                          : element.toString())
                              .collect(joining(",")),
                          starlarkTransition.getPackageContext());
            } else {
              var valueBuilder = ImmutableList.builder();
              // We can't use streams because def.getConverter().convert may throw an
              // OptionsParsingException.
              for (Object e : optionValueAsList) {
                Object converted =
                    def.getConverter()
                        .convert(e.toString(), starlarkTransition.getPackageContext());
                if (converted instanceof List<?> list) {
                  valueBuilder.addAll(list);
                } else {
                  valueBuilder.add(converted);
                }
              }
              convertedValue = valueBuilder.build();
            }
          } else if (def.getType() == List.class && optionValue == null) {
            throw ValidationException.format(
                "'None' value not allowed for List-type option '%s'. Please use '[]' instead if"
                    + " trying to set option to empty value.",
                optionName);
          } else if (optionValue == null || def.getType().isInstance(optionValue)) {
            convertedValue = optionValue;
          } else if (def.getType().equals(int.class) && optionValue instanceof Integer) {
            convertedValue = optionValue;
          } else if (def.getType().equals(boolean.class) && optionValue instanceof Boolean) {
            convertedValue = optionValue;
          } else if (optionValue instanceof String) {
            convertedValue =
                def.getConverter()
                    .convert((String) optionValue, starlarkTransition.getPackageContext());
          } else {
            throw ValidationException.format("Invalid value type for option '%s'", optionName);
          }

          Object oldValue = def.getRawValue(fromOptions.get(optionInfo.getOptionClass()));
          if (!Objects.equals(oldValue, convertedValue)) {
            if (toOptions == null) {
              toOptions = fromOptions.clone();
            }
            def.setValue(toOptions.get(optionInfo.getOptionClass()), convertedValue);
          }

        } catch (IllegalArgumentException e) {
          throw ValidationException.format(
              "IllegalArgumentError for option '%s': %s", optionName, e.getMessage());
        } catch (OptionsParsingException e) {
          throw ValidationException.format(
              "OptionsParsingError for option '%s': %s", optionName, e.getMessage());
        }
      }
    }

    if (toOptions == null && changedStarlarkOptions.isEmpty()) {
      return fromOptions;
    }
    // Note that rebuilding also calls FragmentOptions.getNormalized() to guarantee --define,
    // --features, and similar flags are consistently ordered.
    toOptions =
        BuildOptions.builder()
            .merge(toOptions == null ? fromOptions.clone() : toOptions)
            .addStarlarkOptions(changedStarlarkOptions)
            .build();
    if (starlarkTransition.isForAnalysisTesting()) {
      toOptions.get(CoreOptions.class).evaluatingForAnalysisTest = true;
    }
    return toOptions;
  }

  private FunctionTransitionUtil() {}
}
