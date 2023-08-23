// Copyright 2023 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis.config.transitions;

import static java.util.stream.Collectors.joining;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;
import com.google.common.collect.Sets.SetView;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.BuildOptionsView;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.util.Pair;
import java.util.HashMap;
import java.util.Map;
import java.util.StringJoiner;
import java.util.TreeMap;
import java.util.function.Predicate;

/**
 * A transition that runs two other transitions independently on the same input and compares their
 * results. For Bazel developer debugging.
 */
public class ComparingTransition implements PatchTransition {

  private final ConfigurationTransition activeTransition;
  private final String activeTransitionDesc;

  private final ConfigurationTransition altTransition;
  private final String altTransitionDesc;

  private final Predicate<BuildOptions> runBoth;

  /**
   * Creates a transition that applies {@code activeTransition} and possibly compares with {@code
   * alternativeTransition}.
   *
   * @param activeTransition the transition this one delegates to. If {@code runBoth} is false, the
   *     comparing transition is a pure alias of this.
   * @param activeTransitionDesc user-friendly description of the active transition
   * @param altTransition An alternative transition to compare against {@code activeTransition}..
   *     This only runs when {@code runBoth} is true. In that case, both this and {@code
   *     activeTransition} run independently and their results are compared.
   * @param altTransitionDesc user-friendly description of the alternative transition
   * @param runBoth User-supplied predicate that determines if this should run in comparison mode.
   *     This can be used to toggle debug output with a build flag.
   */
  public ComparingTransition(
      ConfigurationTransition activeTransition,
      String activeTransitionDesc,
      ConfigurationTransition altTransition,
      String altTransitionDesc,
      Predicate<BuildOptions> runBoth) {
    this.activeTransition = activeTransition;
    this.activeTransitionDesc = activeTransitionDesc;
    this.altTransition = altTransition;
    this.altTransitionDesc = altTransitionDesc;
    this.runBoth = runBoth;
  }

  @Override
  public BuildOptions patch(BuildOptionsView buildOptions, EventHandler eventHandler)
      throws InterruptedException {
    Map.Entry<String, BuildOptions> activeOptions =
        Iterables.getOnlyElement(activeTransition.apply(buildOptions, eventHandler).entrySet());
    if (activeOptions.getKey().equals("error")) {
      eventHandler.handle(Event.error(activeTransitionDesc + " transition failed"));
    } else if (runBoth.test(buildOptions.underlying())) {
      compare(
          buildOptions.underlying(),
          activeOptions.getValue(),
          Iterables.getOnlyElement(altTransition.apply(buildOptions, eventHandler).values()),
          eventHandler);
    }
    return activeOptions.getValue();
  }

  private static String prettyClassName(Class<?> clazz) {
    String full = clazz.getName();
    int dot = full.lastIndexOf(".");
    return dot == -1 ? full : full.substring(dot + 1);
  }

  /** Shows differences between two {@link BuildOptions} as debugging terminal output. */
  private void compare(
      BuildOptions fromOptions,
      BuildOptions activeOptions,
      BuildOptions altOptions,
      EventHandler eventHandler) {
    // Log fragments that only exist in one output.
    SetView<Class<? extends FragmentOptions>> onlyInActive =
        Sets.difference(activeOptions.getFragmentClasses(), altOptions.getFragmentClasses());
    SetView<Class<? extends FragmentOptions>> onlyInAlt =
        Sets.difference(altOptions.getFragmentClasses(), activeOptions.getFragmentClasses());
    StringJoiner s = new StringJoiner("\n");
    s.add("------------------------------------------");
    s.add(String.format("ComparingTransition(%s, %s):", activeTransitionDesc, altTransitionDesc));
    s.add(
        String.format(
            "- from: %s, %s to: %s, %s to: %s",
            fromOptions.shortId(),
            activeTransitionDesc,
            activeOptions.shortId(),
            altTransitionDesc,
            altOptions.shortId()));
    s.add(
        String.format(
            "- unique fragments in %s mode: %s",
            activeTransitionDesc,
            onlyInActive.isEmpty()
                ? "none"
                : onlyInActive.stream().map(c -> prettyClassName(c)).collect(joining())));
    s.add(
        String.format(
            "- unique fragments in %s mode: %s",
            altTransitionDesc,
            onlyInAlt.isEmpty()
                ? "none"
                : onlyInAlt.stream().map(c -> prettyClassName(c)).collect(joining())));

    ImmutableMap<String, String> activeMap = serialize(activeOptions);
    ImmutableMap<String, String> altMap = serialize(altOptions);

    // For every option, compute { optionName: <activeValue, altValue> }.
    var combinedMap = new TreeMap<String, Pair<String, String>>();
    for (Map.Entry<String, String> o : activeMap.entrySet()) {
      combinedMap.put(o.getKey(), Pair.of(o.getValue(), "DOES NOT EXIST"));
    }
    for (Map.Entry<String, String> o : altMap.entrySet()) {
      if (!combinedMap.containsKey(o.getKey())) {
        combinedMap.put(o.getKey(), Pair.of("DOES NOT EXIST", o.getValue()));
      } else {
        String newMapValue = combinedMap.get(o.getKey()).getFirst();
        combinedMap.put(o.getKey(), Pair.of(newMapValue, o.getValue()));
      }
    }

    // Print options with different values.
    StringJoiner s2 = new StringJoiner("\n");
    int diffs = 0;
    for (Map.Entry<String, Pair<String, String>> combined : combinedMap.entrySet()) {
      String option = combined.getKey();
      String activeVal = combined.getValue().getFirst();
      String altVal = combined.getValue().getSecond();
      if (activeVal.equals("DOES NOT EXIST")) {
        s2.add(String.format("   only in %s mode: --%s=%s", activeTransitionDesc, option, altVal));
        diffs++;
      } else if (altVal.equals("DOES NOT EXIST")) {
        s2.add(String.format("   only in %s mode: --%s=%s", altTransitionDesc, option, activeVal));
        diffs++;
      } else if (!activeVal.equals(altVal)) {
        s2.add(
            String.format(
                "   --%s: %s mode=%s, %s mode=%s",
                option, activeTransitionDesc, activeVal, altTransitionDesc, altVal));
        diffs++;
      }
    }

    // Summarize diff count both before and after the full diff for easy reading.
    s.add(String.format("- total option differences: %d", diffs));
    s.add(s2.toString());
    if (diffs > 1) {
      s.add(String.format("- total option differences: %d", diffs));
    }
    eventHandler.handle(Event.info(s.toString()));
  }

  /**
   * Maps a {@link BuildOptions} to a user-friendly key=value string map.
   *
   * <p>Splits each {@code --define}, {@code --features} and {@code --host_features} into its own
   * key=value pair.
   */
  private static ImmutableMap<String, String> serialize(BuildOptions o) {
    var ans = ImmutableMap.<String, String>builder();
    for (FragmentOptions f : o.getNativeOptions()) {
      for (Map.Entry<String, Object> op : f.asMap().entrySet()) {
        if (op.getKey().equals("define")) {
          for (Map.Entry<String, String> define :
              o.get(CoreOptions.class).getNormalizedCommandLineBuildVariables().entrySet()) {
            ans.put("user-defined define=" + define.getKey(), define.getValue());
          }
        } else if (op.getKey().equals("features")) {
          var seen = new HashMap<String, Integer>();
          for (String feature : o.get(CoreOptions.class).defaultFeatures) {
            String suffix = "";
            if (seen.containsKey(feature)) {
              int dupeNum = seen.get(feature) + 1;
              suffix = String.format(" (%d)", dupeNum);
              seen.put(feature, dupeNum);
            }
            ans.put(String.format("user-defined feature=%s%s", feature, suffix), "");
          }
        } else if (op.getKey().equals("host_features")) {
          var seen = new HashMap<String, Integer>();
          for (String feature : o.get(CoreOptions.class).hostFeatures) {
            String suffix = "";
            if (seen.containsKey(feature)) {
              int dupeNum = seen.get(feature) + 1;
              suffix = String.format(" (%d)", dupeNum);
              seen.put(feature, dupeNum);
            }
            ans.put(String.format("user-defined host_feature=%s%s", feature, suffix), "");
          }
        } else {
          ans.put(prettyClassName(f.getClass()) + " " + op.getKey(), String.valueOf(op.getValue()));
        }
      }
    }
    for (Map.Entry<Label, Object> starlarkOption : o.getStarlarkOptions().entrySet()) {
      ans.put("user-defined " + starlarkOption.getKey(), starlarkOption.getValue().toString());
    }
    return ans.buildOrThrow();
  }

  @Override
  public String reasonForOverride() {
    return "Adds ability to compare difference between native vs. Starlark transitions";
  }
}
