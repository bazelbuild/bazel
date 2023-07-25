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

package com.google.devtools.build.lib.runtime;

import static com.google.devtools.build.lib.analysis.config.CoreOptionConverters.BUILD_SETTING_CONVERTERS;
import static com.google.devtools.build.lib.packages.RuleClass.Builder.STARLARK_BUILD_SETTING_DEFAULT_ATTR_NAME;
import static com.google.devtools.build.lib.packages.Type.BOOLEAN;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.LinkedListMultimap;
import com.google.common.collect.Multimap;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.packages.BuildSetting;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.common.options.Converter;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.TreeMap;
import javax.annotation.Nullable;

/**
 * An options parser for starlark defined options. Takes a mutable {@link OptionsParser} that has
 * already parsed all native options (including those needed for loading). This class is in charge
 * of parsing and setting the starlark options for this {@link OptionsParser}.
 */
public class StarlarkOptionsParser {

  private final OptionsParser nativeOptionsParser;

  /**
   * Interface for caller-specific logic to convert flag names to {@link Target}s.
   *
   * <p>The most important distinction is whether the caller is in a {@link
   * com.google.devtools.build.skyframe.SkyFunction} evaluation environment.
   */
  @FunctionalInterface
  public interface BuildSettingLoader {
    /**
     * Converts a flag name into a {@link Target}, or throws an exception if this can't be done.
     *
     * @param name the flag to lookup, expected to be a valid {@link Label}
     * @return the {@link Target} corresponding to the flag, or null if the caller has to do more
     *     work to retrieve the target (after which it'll call this parser again)
     */
    @Nullable
    Target loadBuildSetting(String name) throws InterruptedException, TargetParsingException;
  }

  private final BuildSettingLoader buildSettingLoader;

  // Result of #parse, store the parsed options and their values.
  private final Map<String, Object> starlarkOptions = new TreeMap<>();
  // Map of parsed starlark options to their loaded BuildSetting objects (used for canonicalization)
  private final Map<String, BuildSetting> parsedBuildSettings = new HashMap<>();

  // Local cache of build settings so we don't repeatedly load them.
  private final Map<String, Target> buildSettings = new HashMap<>();

  private StarlarkOptionsParser(
      BuildSettingLoader buildSettingLoader, OptionsParser nativeOptionsParser) {
    this.buildSettingLoader = buildSettingLoader;
    this.nativeOptionsParser = nativeOptionsParser;
  }

  public static StarlarkOptionsParser newStarlarkOptionsParser(
      BuildSettingLoader buildSettingLoader, OptionsParser optionsParser) {
    return new StarlarkOptionsParser(buildSettingLoader, optionsParser);
  }

  /**
   * Parses all pre "--" residue for Starlark options.
   *
   * @return true if the flags are parsed, false if the {@link BuildSettingLoader} needs to do more
   *     work to retrieve build setting targets (after which it'll call this method again)
   */
  // TODO(blaze-configurability): This method somewhat reinvents the wheel of
  // OptionsParserImpl.identifyOptionAndPossibleArgument. Consider combining. This would probably
  // require multiple rounds of parsing to fit starlark-defined options into native option format.
  @VisibleForTesting
  public boolean parse() throws OptionsParsingException {
    return parseGivenArgs(nativeOptionsParser.getSkippedArgs());
  }

  /**
   * Parses a specific set of flags.
   *
   * @return true if the flags are parsed, false if the {@link BuildSettingLoader} needs to do more
   *     work to retrieve build setting targets (after which it'll call this method again)
   */
  @VisibleForTesting
  public boolean parseGivenArgs(List<String> args) throws OptionsParsingException {
    // Map of <option name (label), <unparsed option value, loaded option>>.
    Multimap<String, Pair<String, Target>> unparsedOptions = LinkedListMultimap.create();

    boolean allTargetsAvailable = true;
    for (String arg : args) {
      if (!parseArg(arg, unparsedOptions)) {
        allTargetsAvailable = false;
      }
    }

    if (!allTargetsAvailable) {
      return false;
    } else if (unparsedOptions.isEmpty()) {
      return true;
    }

    // Map of flag label as a string to its loaded target and set value after parsing.
    HashMap<String, Pair<Target, Object>> buildSettingWithTargetAndValue = new HashMap<>();
    for (Map.Entry<String, Pair<String, Target>> option : unparsedOptions.entries()) {
      String loadedFlag = option.getKey();
      String unparsedValue = option.getValue().first;
      Target buildSettingTarget = option.getValue().second;
      BuildSetting buildSetting =
          buildSettingTarget.getAssociatedRule().getRuleClassObject().getBuildSetting();
      // Do not recognize internal options, which are treated as if they did not exist.
      if (!buildSetting.isFlag()) {
        throw new OptionsParsingException(
            String.format("Unrecognized option: %s=%s", loadedFlag, unparsedValue));
      }
      Type<?> type = buildSetting.getType();
      if (buildSetting.isRepeatableFlag()) {
        type = Preconditions.checkNotNull(type.getListElementType());
      }
      Converter<?> converter = BUILD_SETTING_CONVERTERS.get(type);
      Object value;
      try {
        value = converter.convert(unparsedValue, nativeOptionsParser.getConversionContext());
      } catch (OptionsParsingException e) {
        throw new OptionsParsingException(
            String.format(
                "While parsing option %s=%s: '%s' is not a %s",
                loadedFlag, unparsedValue, unparsedValue, type),
            e);
      }
      if (buildSetting.allowsMultiple() || buildSetting.isRepeatableFlag()) {
        List<Object> newValue;
        if (buildSettingWithTargetAndValue.containsKey(loadedFlag)) {
          newValue =
              new ArrayList<>(
                  (Collection<?>) buildSettingWithTargetAndValue.get(loadedFlag).getSecond());
        } else {
          newValue = new ArrayList<>();
        }
        newValue.add(value);
        value = newValue;
      }
      buildSettingWithTargetAndValue.put(loadedFlag, Pair.of(buildSettingTarget, value));
    }

    Map<String, Object> parsedOptions = new HashMap<>();
    for (String buildSetting : buildSettingWithTargetAndValue.keySet()) {
      Pair<Target, Object> buildSettingAndFinalValue =
          buildSettingWithTargetAndValue.get(buildSetting);
      Target buildSettingTarget = buildSettingAndFinalValue.getFirst();
      BuildSetting buildSettingObject =
          buildSettingTarget.getAssociatedRule().getRuleClassObject().getBuildSetting();
      boolean allowsMultiple = buildSettingObject.allowsMultiple();
      parsedBuildSettings.put(buildSetting, buildSettingObject);
      Object value = buildSettingAndFinalValue.getSecond();
      if (allowsMultiple) {
        List<?> defaultValue =
            ImmutableList.of(
                Objects.requireNonNull(
                    buildSettingTarget
                        .getAssociatedRule()
                        .getAttr(STARLARK_BUILD_SETTING_DEFAULT_ATTR_NAME)));
        List<?> newValue = (List<?>) value;
        if (!newValue.equals(defaultValue)) {
          parsedOptions.put(buildSetting, value);
        }
      } else {
        if (!value.equals(
            buildSettingTarget
                .getAssociatedRule()
                .getAttr(STARLARK_BUILD_SETTING_DEFAULT_ATTR_NAME))) {
          parsedOptions.put(buildSetting, buildSettingAndFinalValue.getSecond());
        }
      }
    }
    nativeOptionsParser.setStarlarkOptions(ImmutableMap.copyOf(parsedOptions));
    this.starlarkOptions.putAll(parsedOptions);
    return true;
  }

  /**
   * Parses the given {@code flag=value} setting.
   *
   * @return true if parsing finishes, false if the {@link BuildSettingLoader} needs to do more work
   *     to retrieve the build setting target
   */
  private boolean parseArg(String arg, Multimap<String, Pair<String, Target>> unparsedOptions)
      throws OptionsParsingException {
    if (!arg.startsWith("--")) {
      throw new OptionsParsingException("Invalid options syntax: " + arg, arg);
    }
    int equalsAt = arg.indexOf('=');
    String name = equalsAt == -1 ? arg.substring(2) : arg.substring(2, equalsAt);
    if (name.trim().isEmpty()) {
      throw new OptionsParsingException("Invalid options syntax: " + arg, arg);
    }
    String value = equalsAt == -1 ? null : arg.substring(equalsAt + 1);

    if (value != null) {
      // --flag=value or -flag=value form
      Target buildSettingTarget = loadBuildSetting(name);
      if (buildSettingTarget == null) {
        return false;
      }
      // Use the canonical form to ensure we don't have
      // duplicate options getting into the starlark options map.
      unparsedOptions.put(
          buildSettingTarget.getLabel().getCanonicalForm(), new Pair<>(value, buildSettingTarget));
    } else {
      boolean booleanValue = true;
      // check --noflag form
      if (name.startsWith("no")) {
        booleanValue = false;
        name = name.substring(2);
      }
      Target buildSettingTarget = loadBuildSetting(name);
      if (buildSettingTarget == null) {
        return false;
      }
      BuildSetting current =
          buildSettingTarget.getAssociatedRule().getRuleClassObject().getBuildSetting();
      if (current.getType().equals(BOOLEAN)) {
        // --boolean_flag or --noboolean_flag
        // Ditto w/r/t canonical form.
        unparsedOptions.put(
            buildSettingTarget.getLabel().getCanonicalForm(),
            new Pair<>(String.valueOf(booleanValue), buildSettingTarget));
      } else {
        if (!booleanValue) {
          // --no(non_boolean_flag)
          throw new OptionsParsingException(
              "Illegal use of 'no' prefix on non-boolean option: " + name, name);
        }
        throw new OptionsParsingException("Expected value after " + arg);
      }
    }
    return true;
  }

  /**
   * Returns the given build setting's {@link Target}.
   *
   * @return the target, or null if the {@link BuildSettingLoader} needs to do more work to retrieve
   *     the target
   */
  @Nullable
  private Target loadBuildSetting(String targetToBuild) throws OptionsParsingException {
    if (buildSettings.containsKey(targetToBuild)) {
      return buildSettings.get(targetToBuild);
    }

    Target buildSetting;
    try {
      buildSetting = buildSettingLoader.loadBuildSetting(targetToBuild);
      if (buildSetting == null) {
        return null;
      }
    } catch (InterruptedException | TargetParsingException e) {
      Thread.currentThread().interrupt();
      throw new OptionsParsingException(
          "Error loading option " + targetToBuild + ": " + e.getMessage(), targetToBuild, e);
    }
    Rule associatedRule = buildSetting.getAssociatedRule();
    if (associatedRule == null || associatedRule.getRuleClassObject().getBuildSetting() == null) {
      throw new OptionsParsingException("Unrecognized option: " + targetToBuild, targetToBuild);
    }
    buildSettings.put(targetToBuild, buildSetting);
    return buildSetting;
  }

  @VisibleForTesting
  public OptionsParser getNativeOptionsParserFortesting() {
    return nativeOptionsParser;
  }

  public boolean checkIfParsedOptionAllowsMultiple(String option) {
    BuildSetting setting = parsedBuildSettings.get(option);
    return setting.allowsMultiple() || setting.isRepeatableFlag();
  }

  public Type<?> getParsedOptionType(String option) {
    return parsedBuildSettings.get(option).getType();
  }

  /** Return a canoncalized list of the starlark options and values that this parser has parsed. */
  @SuppressWarnings("unchecked")
  public List<String> canonicalize() {
    ImmutableList.Builder<String> result = new ImmutableList.Builder<>();
    for (Map.Entry<String, Object> starlarkOption : starlarkOptions.entrySet()) {
      String starlarkOptionName = starlarkOption.getKey();
      Object starlarkOptionValue = starlarkOption.getValue();
      String starlarkOptionString = "--" + starlarkOptionName + "=";
      if (checkIfParsedOptionAllowsMultiple(starlarkOptionName)) {
        Preconditions.checkState(
            starlarkOption.getValue() instanceof List,
            "Found a starlark option value that isn't a list for an allow multiple option.");
        for (Object singleValue : (List) starlarkOptionValue) {
          result.add(starlarkOptionString + singleValue);
        }
      } else if (getParsedOptionType(starlarkOptionName).equals(Type.STRING_LIST)) {
        result.add(
            starlarkOptionString + String.join(",", ((Iterable<String>) starlarkOptionValue)));
      } else {
        result.add(starlarkOptionString + starlarkOptionValue);
      }
    }
    return result.build();
  }
}
