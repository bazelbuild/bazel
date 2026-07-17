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
import com.google.devtools.build.lib.skyframe.serialization.AsyncDeserializationContext;
import com.google.devtools.build.lib.skyframe.serialization.DeferredObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.SerializationContext;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.util.OrderedSetMultimap;
import com.google.devtools.common.options.OptionDefinition;
import com.google.devtools.common.options.OptionsBase;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
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

/**
 * A diff class for BuildOptions. Fields are meant to be populated and returned by {@link
 * OptionsDiff#diff}.
 */
public final class OptionsDiff {
  public OptionsDiff() {}

  OptionsDiff(
      ListMultimap<Class<? extends FragmentOptions>, OptionDefinition> differingOptions,
      Map<OptionDefinition, Object> first,
      SetMultimap<OptionDefinition, Object> second,
      Set<Class<? extends FragmentOptions>> extraFirstFragments,
      Set<FragmentOptions> extraSecondFragments,
      Map<Label, Object> starlarkFirst,
      Map<Label, Object> starlarkSecond,
      List<Label> extraStarlarkOptionsFirst,
      SequencedMap<Label, Object> extraStarlarkOptionsSecond) {
    this.differingOptions.putAll(differingOptions);
    this.first.putAll(first);
    this.second.putAll(second);
    this.extraFirstFragments.addAll(extraFirstFragments);
    this.extraSecondFragments.addAll(extraSecondFragments);
    this.starlarkFirst.putAll(starlarkFirst);
    this.starlarkSecond.putAll(starlarkSecond);
    this.extraStarlarkOptionsFirst.addAll(extraStarlarkOptionsFirst);
    this.extraStarlarkOptionsSecond.putAll(extraStarlarkOptionsSecond);
    this.hasStarlarkOptions =
        !this.starlarkFirst.isEmpty()
            || !this.starlarkSecond.isEmpty()
            || !this.extraStarlarkOptionsFirst.isEmpty()
            || !this.extraStarlarkOptionsSecond.isEmpty();
  }

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

    Set<Label> allStarlarkKeys = new LinkedHashSet<>();
    allStarlarkKeys.addAll(starlarkFirst.keySet());
    allStarlarkKeys.addAll(starlarkSecond.keySet());

    for (Label buildSetting : allStarlarkKeys) {
      if (!starlarkFirst.containsKey(buildSetting)) {
        diff.addExtraSecondStarlarkOption(buildSetting, starlarkSecond.get(buildSetting));
      } else if (!starlarkSecond.containsKey(buildSetting)) {
        diff.addExtraFirstStarlarkOption(buildSetting);
      } else if (!starlarkFirst.get(buildSetting).equals(starlarkSecond.get(buildSetting))) {
        diff.putStarlarkDiff(
            buildSetting, starlarkFirst.get(buildSetting), starlarkSecond.get(buildSetting));
      }
    }
    return diff;
  }

  private final ListMultimap<Class<? extends FragmentOptions>, OptionDefinition> differingOptions =
      MultimapBuilder.linkedHashKeys().arrayListValues().build();
  // The keyset for the {@link first} and {@link second} maps are identical and indicate which
  // specific options differ between the first and second built options.
  // Note (Latent Risk): After Skyframe serialization/deserialization, the values in this map
  // are replaced with "RESET" strings (for scalars) or ListDiff objects (for lists), violating
  // the expected OptionDefinition value type contract.
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

  private boolean hasStarlarkOptions = false;

  @VisibleForTesting
  Set<Class<? extends FragmentOptions>> getExtraFirstFragmentClassesForTesting() {
    return extraFirstFragments;
  }

  @VisibleForTesting
  Set<FragmentOptions> getExtraSecondFragmentsForTesting() {
    return extraSecondFragments;
  }

  /**
   * Returns the baseline values of the differing options from the first BuildOptions.
   *
   * <p><b>WARNING - Latent Risk / Type Contract Mismatch:</b> If this {@link OptionsDiff} instance
   * was deserialized via Skyframe (e.g. using {@link OptionsDiffCodec}), the values in this map do
   * NOT match the original option types. The codec replaces all non-list base values with the
   * String {@code "RESET"}, and list values with a {@link ListDiff}. Querying values from a
   * deserialized instance and casting them to their expected type (e.g., casting to {@code
   * RunUnder} in {@code OptionsDiffPredicate}) will cause a {@link ClassCastException} in
   * production.
   */
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

  private void putStarlarkDiff(Label buildSetting, Object firstValue, Object secondValue) {
    starlarkFirst.put(buildSetting, firstValue);
    starlarkSecond.put(buildSetting, secondValue);
    hasStarlarkOptions = true;
  }

  private void addExtraFirstStarlarkOption(Label buildSetting) {
    extraStarlarkOptionsFirst.add(buildSetting);
    hasStarlarkOptions = true;
  }

  private void addExtraSecondStarlarkOption(Label buildSetting, Object value) {
    extraStarlarkOptionsSecond.put(buildSetting, value);
    hasStarlarkOptions = true;
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
        && extraStarlarkOptionsSecond.isEmpty();
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
        (option, value) -> toReturn.add(option + ":" + value + " -> " + starlarkSecond.get(option)));
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

    // 5. Inject extra Starlark options introduced in B
    for (Map.Entry<Label, Object> entry : diff.extraStarlarkOptionsSecond.entrySet()) {
      builder.addStarlarkOption(entry.getKey(), entry.getValue());
    }

    // 6. Overwrite differing Starlark options
    for (Map.Entry<Label, Object> entry : diff.starlarkSecond.entrySet()) {
      builder.addStarlarkOption(entry.getKey(), entry.getValue());
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
        && extraStarlarkOptionsSecond.equals(other.extraStarlarkOptionsSecond);
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
        extraStarlarkOptionsSecond);
  }

  /** Serializable reference to an OptionDefinition. */
  @AutoCodec
  public static final class OptionRef {
    public final Class<? extends OptionsBase> declaringClass;
    public final String optionName;

    public OptionRef(Class<? extends OptionsBase> declaringClass, String optionName) {
      this.declaringClass = declaringClass;
      this.optionName = optionName;
    }

    public OptionDefinition getDefinition() {
      for (OptionDefinition definition : OptionDefinition.getOptionDefinitions(declaringClass)) {
        if (definition.getOptionName().equals(optionName)) {
          return definition;
        }
      }
      throw new IllegalStateException(
          "Could not find OptionDefinition for " + optionName + " in " + declaringClass);
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (!(o instanceof OptionRef other)) {
        return false;
      }
      return declaringClass.equals(other.declaringClass) && optionName.equals(other.optionName);
    }

    @Override
    public int hashCode() {
      return Objects.hash(declaringClass, optionName);
    }
  }

  /**
   * Codec to serialize and deserialize {@link OptionsDiff}.
   *
   * <p>Care should be taken when using this codec due to the potential {@link ClassCastExceptions}
   * when using deserialized values. Currently, this codec is only being used to serialize
   * BuildConfigurationKeys based on diffs with a baseline {@link BuildOptions} object.
   * Deserialization happens in a context where the same baseline BuildOptions object is available.
   */
  public static final class OptionsDiffCodec extends DeferredObjectCodec<OptionsDiff> {
    @Override
    public boolean autoRegister() {
      return false;
    }

    @Override
    public Class<OptionsDiff> getEncodedClass() {
      return OptionsDiff.class;
    }

    @Override
    public void serialize(SerializationContext context, OptionsDiff obj, CodedOutputStream codedOut)
        throws SerializationException, IOException {
      // 1. Serialize differingOptions using OptionRef
      Map<Class<? extends FragmentOptions>, List<OptionRef>> differingOptionsRef =
          new LinkedHashMap<>();
      for (Map.Entry<Class<? extends FragmentOptions>, Collection<OptionDefinition>> entry :
          obj.differingOptions.asMap().entrySet()) {
        List<OptionRef> refs = new ArrayList<>();
        for (OptionDefinition def : entry.getValue()) {
          refs.add(new OptionRef(def.getDeclaringClass(OptionsBase.class), def.getOptionName()));
        }
        differingOptionsRef.put(entry.getKey(), refs);
      }
      context.serialize(differingOptionsRef, codedOut);

      // 2. Serialize first using OptionRef
      Map<OptionRef, Object> firstRef = new LinkedHashMap<>();
      for (Map.Entry<OptionDefinition, Object> entry : obj.first.entrySet()) {
        OptionDefinition def = entry.getKey();
        Object firstValue = entry.getValue();
        // Latent Risk / Type Contract Mismatch: Replacing non-list base values with the string
        // "RESET" violates the Map<OptionDefinition, Object> type contract upon deserialization.
        // If a consumer queries the base state fields of a deserialized OptionsDiff object, this
        // can cause a ClassCastException in production (e.g., TestConfiguration casting oldValue
        // directly to RunUnder).
        Object serializedFirstValue = "RESET";
        if (firstValue instanceof List<?>) {
          Set<Object> secondColl = obj.second.get(def);
          if (!secondColl.isEmpty() && secondColl.iterator().next() instanceof List<?>) {
            List<?> list1 = (List<?>) firstValue;
            List<?> list2 = (List<?>) secondColl.iterator().next();
            boolean isAppend =
                list2.size() >= list1.size() && list2.subList(0, list1.size()).equals(list1);
            boolean reset = !isAppend;
            serializedFirstValue = new ListDiff(reset, ImmutableList.of());
          }
        }
        firstRef.put(
            new OptionRef(def.getDeclaringClass(OptionsBase.class), def.getOptionName()),
            serializedFirstValue);
      }
      context.serialize(firstRef, codedOut);

      // 3. Serialize second using OptionRef
      Map<OptionRef, List<Object>> secondRef = new LinkedHashMap<>();
      for (Map.Entry<OptionDefinition, Collection<Object>> entry : obj.second.asMap().entrySet()) {
        OptionDefinition def = entry.getKey();
        Collection<Object> secondColl = entry.getValue();
        List<Object> secondList = new ArrayList<>();
        for (Object secondValue : secondColl) {
          Object serializedSecondValue = secondValue;
          if (secondValue instanceof List<?>) {
            Object firstValue = obj.first.get(def);
            if (firstValue instanceof List<?>) {
              List<?> list1 = (List<?>) firstValue;
              List<?> list2 = (List<?>) secondValue;
              boolean isAppend =
                  list2.size() >= list1.size() && list2.subList(0, list1.size()).equals(list1);
              boolean reset = !isAppend;
              ImmutableList.Builder<Object> added = ImmutableList.builder();
              if (isAppend) {
                added.addAll(list2.subList(list1.size(), list2.size()));
              } else {
                added.addAll(list2);
              }
              serializedSecondValue = new ListDiff(reset, added.build());
            }
          }
          secondList.add(serializedSecondValue);
        }
        secondRef.put(
            new OptionRef(def.getDeclaringClass(OptionsBase.class), def.getOptionName()),
            secondList);
      }
      context.serialize(secondRef, codedOut);

      // 4. Serialize other fields directly
      context.serialize(ImmutableSet.copyOf(obj.extraFirstFragments), codedOut);
      context.serialize(ImmutableSet.copyOf(obj.extraSecondFragments), codedOut);

      Map<Label, Object> starlarkFirstCensored = new LinkedHashMap<>();
      for (Label label : obj.starlarkFirst.keySet()) {
        starlarkFirstCensored.put(label, "RESET");
      }
      context.serialize(starlarkFirstCensored, codedOut);

      context.serialize(obj.starlarkSecond, codedOut);
      context.serialize(ImmutableList.copyOf(obj.extraStarlarkOptionsFirst), codedOut);
      context.serialize(obj.extraStarlarkOptionsSecond, codedOut);
    }

    @Override
    public DeferredObjectCodec.DeferredValue<OptionsDiff> deserializeDeferred(
        AsyncDeserializationContext context, CodedInputStream codedIn)
        throws SerializationException, IOException {
      var builder = new Builder();

      context.deserialize(codedIn, builder, Builder::setDifferingOptionsRef);
      context.deserialize(codedIn, builder, Builder::setFirstRef);
      context.deserialize(codedIn, builder, Builder::setSecondRef);
      context.deserialize(codedIn, builder, Builder::setExtraFirstFragments);
      context.deserialize(codedIn, builder, Builder::setExtraSecondFragments);
      context.deserialize(codedIn, builder, Builder::setStarlarkFirst);
      context.deserialize(codedIn, builder, Builder::setStarlarkSecond);
      context.deserialize(codedIn, builder, Builder::setExtraStarlarkOptionsFirst);
      context.deserialize(codedIn, builder, Builder::setExtraStarlarkOptionsSecond);

      return builder;
    }

    // Safe because the static setters are invoked by Skyframe deserialization, which decodes nested
    // generic objects to their original type parameters.
    @SuppressWarnings("unchecked")
    private static class Builder implements DeferredObjectCodec.DeferredValue<OptionsDiff> {
      private Map<Class<? extends FragmentOptions>, List<OptionRef>> differingOptionsRef;
      private Map<OptionRef, Object> firstRef;
      private Map<OptionRef, List<Object>> secondRef;
      private Set<Class<? extends FragmentOptions>> extraFirstFragments;
      private Set<FragmentOptions> extraSecondFragments;
      private Map<Label, Object> starlarkFirst;
      private Map<Label, Object> starlarkSecond;
      private List<Label> extraStarlarkOptionsFirst;
      private Map<Label, Object> extraStarlarkOptionsSecond;

      private static void setDifferingOptionsRef(Builder b, Object v) {
        b.differingOptionsRef = (Map<Class<? extends FragmentOptions>, List<OptionRef>>) v;
      }

      private static void setFirstRef(Builder b, Object v) {
        b.firstRef = (Map<OptionRef, Object>) v;
      }

      private static void setSecondRef(Builder b, Object v) {
        b.secondRef = (Map<OptionRef, List<Object>>) v;
      }

      private static void setExtraFirstFragments(Builder b, Object v) {
        b.extraFirstFragments = (Set<Class<? extends FragmentOptions>>) v;
      }

      private static void setExtraSecondFragments(Builder b, Object v) {
        b.extraSecondFragments = (Set<FragmentOptions>) v;
      }

      private static void setStarlarkFirst(Builder b, Object v) {
        b.starlarkFirst = (Map<Label, Object>) v;
      }

      private static void setStarlarkSecond(Builder b, Object v) {
        b.starlarkSecond = (Map<Label, Object>) v;
      }

      private static void setExtraStarlarkOptionsFirst(Builder b, Object v) {
        b.extraStarlarkOptionsFirst = (List<Label>) v;
      }

      private static void setExtraStarlarkOptionsSecond(Builder b, Object v) {
        b.extraStarlarkOptionsSecond = (Map<Label, Object>) v;
      }

      @Override
      public OptionsDiff call() {
        // 1. Reconstruct differingOptions
        ListMultimap<Class<? extends FragmentOptions>, OptionDefinition> differingOptions =
            MultimapBuilder.linkedHashKeys().arrayListValues().build();
        for (Map.Entry<Class<? extends FragmentOptions>, List<OptionRef>> entry :
            differingOptionsRef.entrySet()) {
          for (OptionRef ref : entry.getValue()) {
            differingOptions.put(entry.getKey(), ref.getDefinition());
          }
        }

        // 2. Reconstruct first
        Map<OptionDefinition, Object> first = new LinkedHashMap<>();
        for (Map.Entry<OptionRef, Object> entry : firstRef.entrySet()) {
          first.put(entry.getKey().getDefinition(), entry.getValue());
        }

        // 3. Reconstruct second
        SetMultimap<OptionDefinition, Object> second = OrderedSetMultimap.create();
        for (Map.Entry<OptionRef, List<Object>> entry : secondRef.entrySet()) {
          second.putAll(entry.getKey().getDefinition(), entry.getValue());
        }

        SequencedMap<Label, Object> extraStarlarkOptionsSecondMap =
            new LinkedHashMap<>(extraStarlarkOptionsSecond);

        return new OptionsDiff(
            differingOptions,
            first,
            second,
            extraFirstFragments,
            extraSecondFragments,
            starlarkFirst,
            starlarkSecond,
            extraStarlarkOptionsFirst,
            extraStarlarkOptionsSecondMap);
      }
    }
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
