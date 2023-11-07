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

import static com.google.common.collect.ImmutableList.toImmutableList;
import static java.util.stream.Collectors.joining;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;
import com.google.common.collect.Sets.SetView;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.BuildOptionsView;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.util.Pair;
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
        s2.add(String.format("   only in %s mode: --%s=%s", altTransitionDesc, option, altVal));
        diffs++;
      } else if (altVal.equals("DOES NOT EXIST")) {
        s2.add(
            String.format("   only in %s mode: --%s=%s", activeTransitionDesc, option, activeVal));
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
          ans.putAll(
              serializeUserDefinedOption(
                  o.get(CoreOptions.class).commandLineBuildVariables.stream()
                      .map(d -> Map.entry(d.getKey(), d.getValue()))
                      .collect(toImmutableList()),
                  "define"));
        } else if (op.getKey().equals("features")) {
          ans.putAll(
              serializeUserDefinedOption(
                  o.get(CoreOptions.class).defaultFeatures.stream()
                      .map(d -> Map.entry(d, ""))
                      .collect(toImmutableList()),
                  "feature"));
        } else if (op.getKey().equals("host_features")) {
          ans.putAll(
              serializeUserDefinedOption(
                  o.get(CoreOptions.class).hostFeatures.stream()
                      .map(d -> Map.entry(d, ""))
                      .collect(toImmutableList()),
                  "host feature"));
        } else {
          ans.put(prettyClassName(f.getClass()) + " " + op.getKey(), String.valueOf(op.getValue()));
        }
      }
    }
    ans.putAll(
        serializeUserDefinedOption(
            o.getStarlarkOptions().entrySet().stream()
                .map(d -> Map.entry(d.getKey().toString(), d.getValue().toString()))
                .collect(toImmutableList()),
            ""));
    return ans.buildOrThrow();
  }

  /**
   * Expands a {@link BuildOptions} native flag that represents a set of user-defined options.
   *
   * <p>For example: <code>--define</code> or <code>--features</code>.
   */
  private static ImmutableMap<String, String> serializeUserDefinedOption(
      Iterable<Map.Entry<String, String>> userDefinedOption, String desc) {
    ImmutableMap.Builder<String, String> ans = ImmutableMap.builder();
    int index = 0;
    for (Map.Entry<String, String> entry : userDefinedOption) {
      ans.put(
          String.format("user-defined %s %s (index %d)", desc, entry.getKey(), index),
          entry.getValue());
      index++;
    }
    return ans.buildOrThrow();
  }

  @Override
  public String reasonForOverride() {
    return "Adds ability to compare difference between native vs. Starlark transitions";
  }

  /**
   * Implement {@link ConfigurationTransition#visit}} so {@link
   * com.google.devtools.build.lib.analysis.config.StarlarkTransitionCache} kicks in if this calls a
   * Starlark transition.
   *
   * <p>Reason: {@link com.google.devtools.build.lib.analysis.config.StarlarkTransitionCache} is
   * responsible for Starlark transition applications. Let it decisively own that responsibility vs.
   * writing a new ad hoc cache playing the same role for special corner cases. This keeps the
   * overall logic clearer.
   *
   * <p>If both the delegate transitions are native, we need some other way to avoid redundant
   * applications at a possible performance cost. In the long term if we eliminate native
   * transitions, we can eliminate this concern.
   */
  @Override
  public <E extends Exception> void visit(Visitor<E> visitor) throws E {
    this.activeTransition.visit(visitor);
    this.altTransition.visit(visitor);
  }
}
