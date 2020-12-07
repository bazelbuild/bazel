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
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.cmdline.LabelValidator;
import com.google.devtools.build.lib.cmdline.LabelValidator.BadLabelException;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.packages.BuildSetting;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import com.google.devtools.build.lib.skyframe.TargetPatternPhaseValue;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.common.options.Converter;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * An options parser for starlark defined options. Takes a mutable {@link OptionsParser} that has
 * already parsed all native options (including those needed for loading). This class is in charge
 * of parsing and setting the starlark options for this {@link OptionsParser}.
 */
// TODO(juliexxia): confront the spectre of aliased build settings
public class StarlarkOptionsParser {

  private final SkyframeExecutor skyframeExecutor;
  private final PathFragment relativeWorkingDirectory;
  private final Reporter reporter;
  private final OptionsParser nativeOptionsParser;

  private StarlarkOptionsParser(
      SkyframeExecutor skyframeExecutor,
      PathFragment relativeWorkingDirectory,
      Reporter reporter,
      OptionsParser nativeOptionsParser) {
    this.skyframeExecutor = skyframeExecutor;
    this.relativeWorkingDirectory = relativeWorkingDirectory;
    this.reporter = reporter;
    this.nativeOptionsParser = nativeOptionsParser;
  }

  public static StarlarkOptionsParser newStarlarkOptionsParser(
      CommandEnvironment env, OptionsParser optionsParser) {
    return new StarlarkOptionsParser(
        env.getSkyframeExecutor(),
        env.getRelativeWorkingDirectory(),
        env.getReporter(),
        optionsParser);
  }

  /** Parses all pre "--" residue for Starlark options. */
  // TODO(juliexxia): This method somewhat reinvents the wheel of
  // OptionsParserImpl.identifyOptionAndPossibleArgument. Consider combining. This would probably
  // require multiple rounds of parsing to fit starlark-defined options into native option format.
  @VisibleForTesting
  public void parse(ExtendedEventHandler eventHandler) throws OptionsParsingException {
    ImmutableList.Builder<String> residue = new ImmutableList.Builder<>();
    // Map of <option name (label), <unparsed option value, loaded option>>.
    Map<String, Pair<String, Target>> unparsedOptions =
        Maps.newHashMapWithExpectedSize(nativeOptionsParser.getResidue().size());

    // sort the old residue into starlark flags and legitimate residue
    for (String arg : nativeOptionsParser.getPreDoubleDashResidue()) {
      // TODO(bazel-team): support single dash options?
      if (!arg.startsWith("--")) {
        residue.add(arg);
        continue;
      }

      parseArg(arg, unparsedOptions, eventHandler);
    }

    List<String> postDoubleDashResidue = nativeOptionsParser.getPostDoubleDashResidue();
    residue.addAll(postDoubleDashResidue);
    nativeOptionsParser.setResidue(residue.build(), postDoubleDashResidue);

    if (unparsedOptions.isEmpty()) {
      return;
    }

    ImmutableMap.Builder<String, Object> parsedOptions = new ImmutableMap.Builder<>();
    for (Map.Entry<String, Pair<String, Target>> option : unparsedOptions.entrySet()) {
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
      Converter<?> converter = BUILD_SETTING_CONVERTERS.get(type);
      Object value;
      try {
        value = converter.convert(unparsedValue);
      } catch (OptionsParsingException e) {
        throw new OptionsParsingException(
            String.format(
                "While parsing option %s=%s: '%s' is not a %s",
                loadedFlag, unparsedValue, unparsedValue, type),
            e);
      }
      if (!value.equals(
          buildSettingTarget
              .getAssociatedRule()
              .getAttr(STARLARK_BUILD_SETTING_DEFAULT_ATTR_NAME))) {
        parsedOptions.put(loadedFlag, value);
      }
    }
    nativeOptionsParser.setStarlarkOptions(parsedOptions.build());
  }

  private void parseArg(
      String arg,
      Map<String, Pair<String, Target>> unparsedOptions,
      ExtendedEventHandler eventHandler)
      throws OptionsParsingException {
    int equalsAt = arg.indexOf('=');
    String name = equalsAt == -1 ? arg.substring(2) : arg.substring(2, equalsAt);
    if (name.trim().isEmpty()) {
      throw new OptionsParsingException("Invalid options syntax: " + arg, arg);
    }
    String value = equalsAt == -1 ? null : arg.substring(equalsAt + 1);

    if (value != null) {
      // --flag=value or -flag=value form
      Target buildSettingTarget = loadBuildSetting(name, eventHandler);
      unparsedOptions.put(name, new Pair<>(value, buildSettingTarget));
    } else {
      boolean booleanValue = true;
      // check --noflag form
      if (name.startsWith("no")) {
        booleanValue = false;
        name = name.substring(2);
      }
      Target buildSettingTarget = loadBuildSetting(name, eventHandler);
      BuildSetting current =
          buildSettingTarget.getAssociatedRule().getRuleClassObject().getBuildSetting();
      if (current.getType().equals(BOOLEAN)) {
        // --boolean_flag or --noboolean_flag
        unparsedOptions.put(name, new Pair<>(String.valueOf(booleanValue), buildSettingTarget));
      } else {
        if (!booleanValue) {
          // --no(non_boolean_flag)
          throw new OptionsParsingException(
              "Illegal use of 'no' prefix on non-boolean option: " + name, name);
        }
        throw new OptionsParsingException("Expected value after " + arg);
      }
    }
  }

  private Target loadBuildSetting(String targetToBuild, ExtendedEventHandler eventHandler)
      throws OptionsParsingException {
    Target buildSetting;
    try {
      TargetPatternPhaseValue result =
          skyframeExecutor.loadTargetPatternsWithoutFilters(
              reporter,
              Collections.singletonList(targetToBuild),
              relativeWorkingDirectory,
              SkyframeExecutor.DEFAULT_THREAD_COUNT,
              /*keepGoing=*/ false);
      buildSetting =
          Iterables.getOnlyElement(
              result.getTargets(eventHandler, skyframeExecutor.getPackageManager()));
    } catch (InterruptedException | TargetParsingException e) {
      Thread.currentThread().interrupt();
      throw new OptionsParsingException(
          "Error loading option " + targetToBuild + ": " + e.getMessage(), targetToBuild, e);
    }
    Rule associatedRule = buildSetting.getAssociatedRule();
    if (associatedRule == null || associatedRule.getRuleClassObject().getBuildSetting() == null) {
      throw new OptionsParsingException("Unrecognized option: " + targetToBuild, targetToBuild);
    }
    return buildSetting;
  }

  /**
   * Separates out any Starlark options from the given list
   *
   * <p>This method doesn't go through the trouble to actually load build setting targets and verify
   * they are build settings, it just assumes all strings that look like they could be build
   * settings, aka are formatted like a flag and can parse out to a proper label, are build
   * settings. Use actual parsing functions above to do full build setting verification.
   *
   * @param list List of strings from which to parse out starlark options
   * @return Returns a pair of string lists. The first item contains the list of starlark options
   *     that were removed; the second contains the remaining string from the original list.
   */
  public static Pair<ImmutableList<String>, ImmutableList<String>> removeStarlarkOptions(
      List<String> list) {
    ImmutableList.Builder<String> keep = ImmutableList.builder();
    ImmutableList.Builder<String> remove = ImmutableList.builder();
    for (String name : list) {
      // Check if the string is a flag and trim off "--" if so.
      if (!name.startsWith("--")) {
        keep.add(name);
        continue;
      }
      String potentialStarlarkFlag = name.substring(2);
      // Check if the string uses the "no" prefix for setting boolean flags to false, trim
      // off "no" if so.
      if (name.startsWith("no")) {
        potentialStarlarkFlag = potentialStarlarkFlag.substring(2);
      }
      // Check if we can properly parse the (potentially trimmed) string as a label. If so, count
      // as starlark flag, else count as regular residue.
      try {
        LabelValidator.validateAbsoluteLabel(potentialStarlarkFlag);
        remove.add(name);
      } catch (BadLabelException e) {
        keep.add(name);
      }
    }
    return Pair.of(remove.build(), keep.build());
  }

  @VisibleForTesting
  public static StarlarkOptionsParser newStarlarkOptionsParserForTesting(
      SkyframeExecutor skyframeExecutor,
      Reporter reporter,
      PathFragment relativeWorkingDirectory,
      OptionsParser nativeOptionsParser) {
    return new StarlarkOptionsParser(
        skyframeExecutor, relativeWorkingDirectory, reporter, nativeOptionsParser);
  }

  @VisibleForTesting
  public void setResidueForTesting(List<String> residue) {
    nativeOptionsParser.setResidue(residue, ImmutableList.of());
  }

  @VisibleForTesting
  public OptionsParser getNativeOptionsParserFortesting() {
    return nativeOptionsParser;
  }
}
