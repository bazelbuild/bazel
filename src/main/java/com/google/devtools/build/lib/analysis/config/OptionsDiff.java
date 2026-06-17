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
package com.google.devtools.build.lib.analysis.config;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ListMultimap;
import com.google.common.collect.Multimap;
import com.google.common.collect.MultimapBuilder;
import com.google.common.collect.SetMultimap;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.util.OrderedSetMultimap;
import com.google.devtools.common.options.OptionDefinition;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.SequencedMap;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * A diff class for BuildOptions. Fields are meant to be populated and returned by {@link
 * OptionsDiff#diff}.
 */
public final class OptionsDiff {
  /** Returns the difference between two BuildOptions in a new {@link OptionsDiff}. */
  public static OptionsDiff diff(BuildOptions first, BuildOptions second) {
    return diff(new OptionsDiff(), first, second);
  }

  /**
   * Returns the difference between two BuildOptions in a pre-existing {@link OptionsDiff}.
   *
   * <p>In a single pass through this method, the method can only compare a single "first" {@link
   * BuildOptions} and single "second" BuildOptions; but an OptionsDiff instance can store the diff
   * between a single "first" BuildOptions and multiple "second" BuildOptions. Being able to
   * maintain a single OptionsDiff over multiple calls to diff is useful for, for example,
   * aggregating the difference between a single BuildOptions and the results of applying a {@link
   * com.google.devtools.build.lib.analysis.config.transitions.SplitTransition}) to it.
   */
  @SuppressWarnings("ReferenceEquality") // See comment above == comparison.
  public static OptionsDiff diff(OptionsDiff diff, BuildOptions first, BuildOptions second) {
    checkArgument(
        !diff.hasStarlarkOptions,
        "OptionsDiff cannot handle multiple 'second' BuildOptions with Starlark options and is"
            + " trying to diff against %s",
        diff);
    checkNotNull(first);
    checkNotNull(second);
    if (first.equals(second)) {
      return diff;
    }

    // Check and report if either class has been trimmed of an options class that exists in the
    // other.
    ImmutableSet<Class<? extends FragmentOptions>> firstOptionClasses =
        first.getNativeOptions().stream()
            .map(FragmentOptions::getOptionsClass)
            .collect(ImmutableSet.toImmutableSet());
    ImmutableSet<Class<? extends FragmentOptions>> secondOptionClasses =
        second.getNativeOptions().stream()
            .map(FragmentOptions::getOptionsClass)
            .collect(ImmutableSet.toImmutableSet());
    Sets.difference(firstOptionClasses, secondOptionClasses).forEach(diff::addExtraFirstFragment);
    Sets.difference(secondOptionClasses, firstOptionClasses).stream()
        .map(second::get)
        .forEach(diff::addExtraSecondFragment);

    // For fragments in common, report differences.
    for (Class<? extends FragmentOptions> clazz :
        Sets.intersection(firstOptionClasses, secondOptionClasses)) {
      FragmentOptions firstOptions = first.get(clazz);
      FragmentOptions secondOptions = second.get(clazz);
      // We avoid calling #equals because we are going to do a field-by-field comparison anyway.
      if (firstOptions == secondOptions) {
        continue;
      }
      for (OptionDefinition definition : OptionDefinition.getOptionDefinitions(clazz)) {
        Object firstValue = definition.getValue(firstOptions);
        Object secondValue = definition.getValue(secondOptions);
        if (!Objects.equals(firstValue, secondValue)) {
          diff.addDiff(clazz, definition, firstValue, secondValue);
        }
      }
    }

    // Compare Starlark options for the two classes.
    ImmutableMap<Label, Object> starlarkFirst = first.getStarlarkOptions();
    ImmutableMap<Label, Object> starlarkSecond = second.getStarlarkOptions();
    ImmutableMap<Label, Scope.ScopeType> scopesFirst = first.getScopeTypeMap();
    ImmutableMap<Label, Scope.ScopeType> scopesSecond = second.getScopeTypeMap();
    ImmutableMap<Label, Object> onLeaveFirst = first.getOnLeaveScopeValues();
    ImmutableMap<Label, Object> onLeaveSecond = second.getOnLeaveScopeValues();

    Set<Label> allStarlarkKeys = new LinkedHashSet<>();
    allStarlarkKeys.addAll(starlarkFirst.keySet());
    allStarlarkKeys.addAll(starlarkSecond.keySet());

    for (Label buildSetting : allStarlarkKeys) {
      if (!starlarkFirst.containsKey(buildSetting)) {
        diff.addExtraSecondStarlarkOption(
            buildSetting,
            starlarkSecond.get(buildSetting),
            scopesSecond.get(buildSetting),
            onLeaveSecond.get(buildSetting));
      } else if (!starlarkSecond.containsKey(buildSetting)) {
        diff.addExtraFirstStarlarkOption(buildSetting);
      } else {
        boolean valuesDifferent =
            !starlarkFirst.get(buildSetting).equals(starlarkSecond.get(buildSetting));
        boolean scopesDifferent =
            !Objects.equals(scopesFirst.get(buildSetting), scopesSecond.get(buildSetting));
        boolean onLeavesDifferent =
            !Objects.equals(onLeaveFirst.get(buildSetting), onLeaveSecond.get(buildSetting));

        // If the values are different, we report the diff for the values, scopes, and on-leave
        // values. If the values are the same, we only report the diff for the
        // scopes and on-leave values. The reason for this is that @{link
        // OutputPathMnemonicComputer} calls @{link getChangedStarlarkOptions}
        // to compute the configuration segment of output paths. We want the
        // metadata to be in the diff for serialization purposes but metadata
        // should not affect the configuration segment (ST-<hash>) of output
        // paths.
        if (valuesDifferent) {
          diff.putStarlarkDiff(
              buildSetting,
              starlarkFirst.get(buildSetting),
              starlarkSecond.get(buildSetting),
              scopesFirst.get(buildSetting),
              scopesSecond.get(buildSetting),
              onLeaveFirst.get(buildSetting),
              onLeaveSecond.get(buildSetting));
        } else if (scopesDifferent || onLeavesDifferent) {
          if (scopesFirst.containsKey(buildSetting)) {
            diff.starlarkFirstScopes.put(buildSetting, scopesFirst.get(buildSetting));
          }
          if (scopesSecond.containsKey(buildSetting)) {
            diff.starlarkSecondScopes.put(buildSetting, scopesSecond.get(buildSetting));
          }
          if (onLeaveFirst.containsKey(buildSetting)) {
            diff.starlarkFirstOnLeaveValues.put(buildSetting, onLeaveFirst.get(buildSetting));
          }
          if (onLeaveSecond.containsKey(buildSetting)) {
            diff.starlarkSecondOnLeaveValues.put(buildSetting, onLeaveSecond.get(buildSetting));
          }
          diff.hasStarlarkOptions = true;
        }
      }
    }
    return diff;
  }

  private final ListMultimap<Class<? extends FragmentOptions>, OptionDefinition> differingOptions =
      MultimapBuilder.linkedHashKeys().arrayListValues().build();
  // The keyset for the {@link first} and {@link second} maps are identical and indicate which
  // specific options differ between the first and second built options.
  private final SequencedMap<OptionDefinition, Object> first = new LinkedHashMap<>();
  // Since this class can be used to track the result of transitions, {@link second} is a multimap
  // to be able to handle {@link SplitTransition}s.
  private final SetMultimap<OptionDefinition, Object> second = OrderedSetMultimap.create();
  // List of "extra" fragments for each BuildOption aka fragments that were trimmed off one
  // BuildOption but not the other.
  private final Set<Class<? extends FragmentOptions>> extraFirstFragments = new LinkedHashSet<>();
  private final Set<FragmentOptions> extraSecondFragments = new LinkedHashSet<>();

  private final SequencedMap<Label, Object> starlarkFirst = new LinkedHashMap<>();
  // TODO(b/112041323): This should also be multimap but we don't diff multiple times with
  // Starlark options anywhere yet so add that feature when necessary.
  private final SequencedMap<Label, Object> starlarkSecond = new LinkedHashMap<>();

  private final List<Label> extraStarlarkOptionsFirst = new ArrayList<>();
  private final SequencedMap<Label, Object> extraStarlarkOptionsSecond = new LinkedHashMap<>();

  // Starlark option metadata difference maps
  private final SequencedMap<Label, Scope.ScopeType> extraStarlarkOptionsSecondScopes =
      new LinkedHashMap<>();
  private final SequencedMap<Label, Object> extraStarlarkOptionsSecondOnLeaveValues =
      new LinkedHashMap<>();

  private final SequencedMap<Label, Scope.ScopeType> starlarkFirstScopes = new LinkedHashMap<>();
  private final SequencedMap<Label, Scope.ScopeType> starlarkSecondScopes = new LinkedHashMap<>();
  private final SequencedMap<Label, Object> starlarkFirstOnLeaveValues = new LinkedHashMap<>();
  private final SequencedMap<Label, Object> starlarkSecondOnLeaveValues = new LinkedHashMap<>();

  private boolean hasStarlarkOptions = false;

  @VisibleForTesting
  Set<Class<? extends FragmentOptions>> getExtraFirstFragmentClassesForTesting() {
    return extraFirstFragments;
  }

  @VisibleForTesting
  Set<FragmentOptions> getExtraSecondFragmentsForTesting() {
    return extraSecondFragments;
  }

  public Map<OptionDefinition, Object> getFirst() {
    return first;
  }

  public Multimap<OptionDefinition, Object> getSecond() {
    return second;
  }

  private void addDiff(
      Class<? extends FragmentOptions> fragmentOptionsClass,
      OptionDefinition option,
      Object firstValue,
      Object secondValue) {
    differingOptions.put(fragmentOptionsClass, option);
    first.put(option, firstValue);
    second.put(option, secondValue);
  }

  private void addExtraFirstFragment(Class<? extends FragmentOptions> options) {
    extraFirstFragments.add(options);
  }

  private void addExtraSecondFragment(FragmentOptions options) {
    extraSecondFragments.add(options);
  }

  private void putStarlarkDiff(
      Label buildSetting,
      Object firstValue,
      Object secondValue,
      @Nullable Scope.ScopeType firstScope,
      @Nullable Scope.ScopeType secondScope,
      @Nullable Object firstOnLeave,
      @Nullable Object secondOnLeave) {
    starlarkFirst.put(buildSetting, firstValue);
    starlarkSecond.put(buildSetting, secondValue);
    if (firstScope != null) {
      starlarkFirstScopes.put(buildSetting, firstScope);
    }
    if (secondScope != null) {
      starlarkSecondScopes.put(buildSetting, secondScope);
    }
    if (firstOnLeave != null) {
      starlarkFirstOnLeaveValues.put(buildSetting, firstOnLeave);
    }
    if (secondOnLeave != null) {
      starlarkSecondOnLeaveValues.put(buildSetting, secondOnLeave);
    }
    hasStarlarkOptions = true;
  }

  private void addExtraFirstStarlarkOption(Label buildSetting) {
    extraStarlarkOptionsFirst.add(buildSetting);
    hasStarlarkOptions = true;
  }

  private void addExtraSecondStarlarkOption(
      Label buildSetting, Object value, @Nullable Scope.ScopeType scope, @Nullable Object onLeave) {
    extraStarlarkOptionsSecond.put(buildSetting, value);
    if (scope != null) {
      extraStarlarkOptionsSecondScopes.put(buildSetting, scope);
    }
    if (onLeave != null) {
      extraStarlarkOptionsSecondOnLeaveValues.put(buildSetting, onLeave);
    }
    hasStarlarkOptions = true;
  }

  public Map<Label, Scope.ScopeType> getExtraStarlarkOptionsSecondScopes() {
    return extraStarlarkOptionsSecondScopes;
  }

  public Map<Label, Object> getExtraStarlarkOptionsSecondOnLeaveValues() {
    return extraStarlarkOptionsSecondOnLeaveValues;
  }

  public Map<Label, Scope.ScopeType> getStarlarkFirstScopes() {
    return starlarkFirstScopes;
  }

  public Map<Label, Scope.ScopeType> getStarlarkSecondScopes() {
    return starlarkSecondScopes;
  }

  public Map<Label, Object> getStarlarkFirstOnLeaveValues() {
    return starlarkFirstOnLeaveValues;
  }

  public Map<Label, Object> getStarlarkSecondOnLeaveValues() {
    return starlarkSecondOnLeaveValues;
  }

  /**
   * Returns the labels of all starlark options that caused a difference between the first and
   * second options set.
   */
  public ImmutableSet<Label> getChangedStarlarkOptions() {
    return ImmutableSet.<Label>builder()
        .addAll(starlarkFirst.keySet())
        .addAll(starlarkSecond.keySet())
        .addAll(extraStarlarkOptionsFirst)
        .addAll(extraStarlarkOptionsSecond.keySet())
        .build();
  }

  @VisibleForTesting
  Map<Label, Object> getStarlarkFirstForTesting() {
    return starlarkFirst;
  }

  @VisibleForTesting
  Map<Label, Object> getStarlarkSecondForTesting() {
    return starlarkSecond;
  }

  @VisibleForTesting
  List<Label> getExtraStarlarkOptionsFirstForTesting() {
    return extraStarlarkOptionsFirst;
  }

  @VisibleForTesting
  Map<Label, Object> getExtraStarlarkOptionsSecondForTesting() {
    return extraStarlarkOptionsSecond;
  }

  /**
   * Note: it's not enough for first and second to be empty, with trimming, they must also contain
   * the same options classes.
   */
  boolean areSame() {
    return first.isEmpty()
        && second.isEmpty()
        && extraSecondFragments.isEmpty()
        && extraFirstFragments.isEmpty()
        && differingOptions.isEmpty()
        && starlarkFirst.isEmpty()
        && starlarkSecond.isEmpty()
        && extraStarlarkOptionsFirst.isEmpty()
        && extraStarlarkOptionsSecond.isEmpty()
        && extraStarlarkOptionsSecondScopes.isEmpty()
        && extraStarlarkOptionsSecondOnLeaveValues.isEmpty()
        && starlarkFirstScopes.isEmpty()
        && starlarkSecondScopes.isEmpty()
        && starlarkFirstOnLeaveValues.isEmpty()
        && starlarkSecondOnLeaveValues.isEmpty();
  }

  public String prettyPrint() {
    StringBuilder toReturn = new StringBuilder();
    for (String diff : getPrettyPrintList()) {
      toReturn.append(diff).append("\n");
    }
    return toReturn.toString();
  }

  public List<String> getPrettyPrintList() {
    List<String> toReturn = new ArrayList<>();
    extraFirstFragments.forEach(
        clazz -> toReturn.add("Removed fragment: " + clazz.getSimpleName()));
    extraSecondFragments.forEach(
        options -> toReturn.add("Added fragment: " + options.getClass().getSimpleName()));
    first.forEach(
        (option, value) ->
            toReturn.add(option.getOptionName() + ":" + value + " -> " + second.get(option)));
    starlarkFirst.forEach(
        (option, value) -> {
          Object secondVal = starlarkSecond.get(option);
          boolean valuesDiff = !Objects.equals(value, secondVal);

          Scope.ScopeType scopeFirst = starlarkFirstScopes.get(option);
          Scope.ScopeType scopeSecond = starlarkSecondScopes.get(option);
          boolean scopesDiff = !Objects.equals(scopeFirst, scopeSecond);

          Object onLeaveFirst = starlarkFirstOnLeaveValues.get(option);
          Object onLeaveSecond = starlarkSecondOnLeaveValues.get(option);
          boolean onLeavesDiff = !Objects.equals(onLeaveFirst, onLeaveSecond);

          if (valuesDiff) {
            toReturn.add(option + ":" + value + " -> " + secondVal);
          } else {
            StringBuilder explanation = new StringBuilder();
            explanation
                .append(option)
                .append(": value is ")
                .append(value)
                .append(" (differs due to: ");
            List<String> causes = new ArrayList<>();
            if (scopesDiff) {
              causes.add("scope: " + scopeFirst + " -> " + scopeSecond);
            }
            if (onLeavesDiff) {
              causes.add("onLeave: " + onLeaveFirst + " -> " + onLeaveSecond);
            }
            explanation.append(String.join(", ", causes)).append(")");
            toReturn.add(explanation.toString());
          }
        });
    extraStarlarkOptionsFirst.forEach(option -> toReturn.add("Removed Starlark option: " + option));
    extraStarlarkOptionsSecond.forEach(
        (option, value) -> toReturn.add("Added Starlark option: " + option + " -> " + value));
    return toReturn;
  }

  /**
   * Reconstructs and returns the target BuildOptions (B) by applying this diff (A - B) to the given
   * base BuildOptions (A).
   */
  public static BuildOptions applyDiff(BuildOptions base, OptionsDiff diff) {
    BuildOptions.Builder builder = base.toBuilder();

    // 1. Remove native fragments present in A but trimmed in B
    for (Class<? extends FragmentOptions> fragmentClass : diff.extraFirstFragments) {
      builder.removeFragmentOptions(fragmentClass);
    }

    // 2. Add native fragments introduced in B but absent in A
    for (FragmentOptions fragment : diff.extraSecondFragments) {
      builder.addFragmentOptions(fragment);
    }

    // 3. Update differing native options inside their active fragments
    for (Map.Entry<OptionDefinition, Collection<Object>> entry : diff.second.asMap().entrySet()) {
      OptionDefinition option = entry.getKey();
      Collection<Object> values = entry.getValue();
      if (values.isEmpty()) {
        continue;
      }
      Object newValue = values.iterator().next();

      Class<? extends FragmentOptions> fragmentClass =
          option.getDeclaringClass(FragmentOptions.class);
      FragmentOptions fragment = builder.getFragmentOptions(fragmentClass);
      if (fragment != null) {
        if (newValue instanceof ListDiff listDiff) {
          ImmutableList.Builder<Object> listBuilder = ImmutableList.builder();
          if (!listDiff.isReset()) {
            listBuilder.addAll((Collection<?>) option.getValue(fragment));
          }
          listBuilder.addAll(listDiff.getElements());
          option.setValue(fragment, listBuilder.build());
        } else {
          option.setValue(fragment, newValue);
        }
      }
    }

    // 4. Trim Starlark options present in A but missing in B
    for (Label label : diff.extraStarlarkOptionsFirst) {
      builder.removeStarlarkOption(label);
    }

    // 5. Inject extra Starlark options and their metadata introduced in B
    for (Map.Entry<Label, Object> entry : diff.extraStarlarkOptionsSecond.entrySet()) {
      builder.addStarlarkOption(entry.getKey(), entry.getValue());
    }

    for (Map.Entry<Label, Scope.ScopeType> entry :
        diff.extraStarlarkOptionsSecondScopes.entrySet()) {
      builder.addScopeType(entry.getKey(), entry.getValue());
    }

    for (Map.Entry<Label, Object> entry : diff.extraStarlarkOptionsSecondOnLeaveValues.entrySet()) {
      builder.addOnLeaveScopeValue(entry.getKey(), entry.getValue());
    }

    // 6. Overwrite differing Starlark options and their metadata
    for (Map.Entry<Label, Object> entry : diff.starlarkSecond.entrySet()) {
      builder.addStarlarkOption(entry.getKey(), entry.getValue());
    }

    for (Map.Entry<Label, Scope.ScopeType> entry : diff.starlarkSecondScopes.entrySet()) {
      builder.addScopeType(entry.getKey(), entry.getValue());
    }

    for (Map.Entry<Label, Object> entry : diff.starlarkSecondOnLeaveValues.entrySet()) {
      builder.addOnLeaveScopeValue(entry.getKey(), entry.getValue());
    }

    return builder.build();
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (!(o instanceof OptionsDiff other)) {
      return false;
    }
    return differingOptions.equals(other.differingOptions)
        && first.equals(other.first)
        && second.equals(other.second)
        && extraFirstFragments.equals(other.extraFirstFragments)
        && extraSecondFragments.equals(other.extraSecondFragments)
        && starlarkFirst.equals(other.starlarkFirst)
        && starlarkSecond.equals(other.starlarkSecond)
        && extraStarlarkOptionsFirst.equals(other.extraStarlarkOptionsFirst)
        && extraStarlarkOptionsSecond.equals(other.extraStarlarkOptionsSecond)
        && extraStarlarkOptionsSecondScopes.equals(other.extraStarlarkOptionsSecondScopes)
        && extraStarlarkOptionsSecondOnLeaveValues.equals(
            other.extraStarlarkOptionsSecondOnLeaveValues)
        && starlarkFirstScopes.equals(other.starlarkFirstScopes)
        && starlarkSecondScopes.equals(other.starlarkSecondScopes)
        && starlarkFirstOnLeaveValues.equals(other.starlarkFirstOnLeaveValues)
        && starlarkSecondOnLeaveValues.equals(other.starlarkSecondOnLeaveValues);
  }

  @Override
  public int hashCode() {
    return Objects.hash(
        differingOptions,
        first,
        second,
        extraFirstFragments,
        extraSecondFragments,
        starlarkFirst,
        starlarkSecond,
        extraStarlarkOptionsFirst,
        extraStarlarkOptionsSecond,
        extraStarlarkOptionsSecondScopes,
        extraStarlarkOptionsSecondOnLeaveValues,
        starlarkFirstScopes,
        starlarkSecondScopes,
        starlarkFirstOnLeaveValues,
        starlarkSecondOnLeaveValues);
  }

  /**
   * Represents the difference between two lists, showing added elements and whether the list was
   * reset.
   */
  @AutoCodec
  public static final class ListDiff implements Serializable {
    private final boolean reset;
    private final ImmutableList<Object> elements;

    public ListDiff(boolean reset, ImmutableList<Object> elements) {
      this.reset = reset;
      this.elements = elements;
    }

    public boolean isReset() {
      return reset;
    }

    public ImmutableList<Object> getElements() {
      return elements;
    }

    @Override
    public boolean equals(Object o) {
      if (!(o instanceof ListDiff other)) {
        return false;
      }
      return reset == other.reset && elements.equals(other.elements);
    }

    @Override
    public int hashCode() {
      return Objects.hash(reset, elements);
    }
  }
}
