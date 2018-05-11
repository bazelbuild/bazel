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

package com.google.devtools.build.lib.analysis.config;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.MoreObjects;
import com.google.common.base.Preconditions;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Multimap;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.runtime.proto.InvocationPolicyOuterClass.InvocationPolicy;
import com.google.devtools.build.lib.skyframe.serialization.DeserializationContext;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.SerializationContext;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.util.OrderedSetMultimap;
import com.google.devtools.common.options.InvocationPolicyEnforcer;
import com.google.devtools.common.options.OptionDefinition;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsClassProvider;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.stream.Collectors;
import javax.annotation.Nullable;

/** Stores the command-line options from a set of configuration fragments. */
// TODO(janakr): If overhead of FragmentOptions class names is too high, add constructor that just
// takes fragments and gets names from them.
@AutoCodec
public final class BuildOptions implements Cloneable, Serializable {
  private static final Comparator<Class<? extends FragmentOptions>>
      lexicalFragmentOptionsComparator = Comparator.comparing(Class::getName);

  /**
   * Creates a BuildOptions object with all options set to their default values, processed by the
   * given {@code invocationPolicy}.
   */
  static BuildOptions createDefaults(
      Iterable<Class<? extends FragmentOptions>> options, InvocationPolicy invocationPolicy) {
    return of(options, createDefaultParser(options, invocationPolicy));
  }

  private static OptionsParser createDefaultParser(
      Iterable<Class<? extends FragmentOptions>> options, InvocationPolicy invocationPolicy) {
    OptionsParser optionsParser = OptionsParser.newOptionsParser(options);
    try {
      new InvocationPolicyEnforcer(invocationPolicy).enforce(optionsParser);
    } catch (OptionsParsingException e) {
      throw new IllegalStateException(e);
    }
    return optionsParser;
  }

  /**
   * Creates a new BuildOptions instance for host.
   */
  public BuildOptions createHostOptions() {
    Builder builder = builder();
    for (FragmentOptions options : fragmentOptionsMap.values()) {
      builder.add(options.getHost());
    }
    return builder.build();
  }

  /**
   * Returns an equivalent instance to this one with only options from the given
   * {@link FragmentOptions} classes.
   */
  public BuildOptions trim(Set<Class<? extends FragmentOptions>> optionsClasses) {
    Builder builder = builder();
    for (FragmentOptions options : fragmentOptionsMap.values()) {
      if (optionsClasses.contains(options.getClass())
          // TODO(bazel-team): make this non-hacky while not requiring BuildConfiguration access
          // to BuildOptions.
          || options.toString().contains("BuildConfiguration$Options")) {
        builder.add(options);
      }
    }
    return builder.build();
  }

  /**
   * Creates a BuildOptions class by taking the option values from an options provider
   * (eg. an OptionsParser).
   */
  public static BuildOptions of(
      Iterable<Class<? extends FragmentOptions>> optionsList, OptionsClassProvider provider) {
    Builder builder = builder();
    for (Class<? extends FragmentOptions> optionsClass : optionsList) {
      builder.add(provider.getOptions(optionsClass));
    }
    return builder.build();
  }

  /**
   * Creates a BuildOptions class by taking the option values from command-line arguments.
   */
  @VisibleForTesting
  public static BuildOptions of(List<Class<?extends FragmentOptions>> optionsList, String... args)
      throws OptionsParsingException {
    Builder builder = builder();
    OptionsParser parser = OptionsParser.newOptionsParser(
        ImmutableList.<Class<? extends OptionsBase>>copyOf(optionsList));
    parser.parse(args);
    for (Class<? extends FragmentOptions> optionsClass : optionsList) {
      builder.add(parser.getOptions(optionsClass));
    }
    return builder.build();
  }

  /**
   * Returns the actual instance of a FragmentOptions class.
   */
  public <T extends FragmentOptions> T get(Class<T> optionsClass) {
    FragmentOptions options = fragmentOptionsMap.get(optionsClass);
    Preconditions.checkNotNull(options, "fragment options unavailable: " + optionsClass.getName());
    return optionsClass.cast(options);
  }

  /**
   * Returns true if these options contain the given {@link FragmentOptions}.
   */
  public boolean contains(Class<? extends FragmentOptions> optionsClass) {
    return fragmentOptionsMap.containsKey(optionsClass);
  }

  // It would be very convenient to use a Multimap here, but we cannot do that because we need to
  // support defaults labels that have zero elements.
  ImmutableMap<String, ImmutableSet<Label>> getDefaultsLabels() {
    Map<String, Set<Label>> collector  = new TreeMap<>();
    for (FragmentOptions fragment : fragmentOptionsMap.values()) {
      for (Map.Entry<String, Set<Label>> entry : fragment.getDefaultsLabels().entrySet()) {
        if (!collector.containsKey(entry.getKey())) {
          collector.put(entry.getKey(), new TreeSet<Label>());
        }
        collector.get(entry.getKey()).addAll(entry.getValue());
      }
    }

    ImmutableMap.Builder<String, ImmutableSet<Label>> result = new ImmutableMap.Builder<>();
    for (Map.Entry<String, Set<Label>> entry : collector.entrySet()) {
      result.put(entry.getKey(), ImmutableSet.copyOf(entry.getValue()));
    }

    return result.build();
  }

  /**
   * Returns true if actions should be enabled for this configuration.
   */
  public boolean enableActions() {
    // It's not necessarily safe to cache this value. This is because BuildOptions is not immutable.
    // So caching the value correctly would require keeping it updated after relevant changes.
    for (FragmentOptions fragment : fragmentOptionsMap.values()) {
      if (!fragment.enableActions()) {
        return false;
      }
    }
    return true;
   }

  /**
   * The cache key for the options collection. Recomputes cache key every time it's called.
   */
  public String computeCacheKey() {
    StringBuilder keyBuilder = new StringBuilder();
    for (FragmentOptions options : fragmentOptionsMap.values()) {
      keyBuilder.append(options.cacheKey());
    }
    return keyBuilder.toString();
  }

  public String computeChecksum() {
    return Fingerprint.md5Digest(computeCacheKey());
  }

  /**
   * String representation of build options.
   */
  @Override
  public String toString() {
    StringBuilder stringBuilder = new StringBuilder();
    for (FragmentOptions options : fragmentOptionsMap.values()) {
      stringBuilder.append(options);
    }
    return stringBuilder.toString();
  }

  /**
   * Returns the options contained in this collection.
   */
  public Collection<FragmentOptions> getOptions() {
    return fragmentOptionsMap.values();
  }

  /**
   * Creates a copy of the BuildOptions object that contains copies of the FragmentOptions.
   */
  @Override
  public BuildOptions clone() {
    ImmutableMap.Builder<Class<? extends FragmentOptions>, FragmentOptions> builder =
        ImmutableMap.builder();
    for (Map.Entry<Class<? extends FragmentOptions>, FragmentOptions> entry :
        fragmentOptionsMap.entrySet()) {
      builder.put(entry.getKey(), entry.getValue().clone());
    }
    return new BuildOptions(builder.build());
  }

  /**
   * Lazily initialize {@link #fingerprint} and {@link #hashCode}. Keeps computation off critical
   * path of build, while still avoiding expensive computation for equality and hash code each time.
   *
   * <p>We check for nullity of {@link #fingerprint} to see if this method has already been called.
   * Using {@link #hashCode} after this method is called is safe because it is set here before
   * {@link #fingerprint} is set, so if {@link #fingerprint} is non-null then {@link #hashCode} is
   * definitely set.
   */
  private void maybeInitializeFingerprintAndHashCode() {
    if (fingerprint != null) {
      return;
    }
    synchronized (this) {
      if (fingerprint != null) {
        return;
      }
      Fingerprint fingerprint = new Fingerprint();
      for (Map.Entry<Class<? extends FragmentOptions>, FragmentOptions> entry :
          fragmentOptionsMap.entrySet()) {
        fingerprint.addString(entry.getKey().getName());
        fingerprint.addString(entry.getValue().cacheKey());
      }
      byte[] computedFingerprint = fingerprint.digestAndReset();
      hashCode = Arrays.hashCode(computedFingerprint);
      this.fingerprint = computedFingerprint;
    }
  }

  @Override
  public boolean equals(Object other) {
    if (this == other) {
      return true;
    } else if (!(other instanceof BuildOptions)) {
      return false;
    } else {
      maybeInitializeFingerprintAndHashCode();
      BuildOptions otherOptions = (BuildOptions) other;
      otherOptions.maybeInitializeFingerprintAndHashCode();
      return Arrays.equals(this.fingerprint, otherOptions.fingerprint);
    }
  }

  @Override
  public int hashCode() {
    maybeInitializeFingerprintAndHashCode();
    return hashCode;
  }

  // Lazily initialized.
  @Nullable private volatile byte[] fingerprint;
  private volatile int hashCode;

  /**
   * Maps options class definitions to FragmentOptions objects.
   */
  private final ImmutableMap<Class<? extends FragmentOptions>, FragmentOptions> fragmentOptionsMap;

  @AutoCodec.VisibleForSerialization
  BuildOptions(ImmutableMap<Class<? extends FragmentOptions>, FragmentOptions> fragmentOptionsMap) {
    this.fragmentOptionsMap = fragmentOptionsMap;
  }

  public BuildOptions applyDiff(OptionsDiffForReconstruction optionsDiff) {
    if (optionsDiff.isEmpty()) {
      return this;
    }
    maybeInitializeFingerprintAndHashCode();
    if (!Arrays.equals(fingerprint, optionsDiff.baseFingerprint)) {
      throw new IllegalArgumentException("Can not reconstruct BuildOptions with a different base.");
    }
    Builder builder = builder();
    for (FragmentOptions options : fragmentOptionsMap.values()) {
      FragmentOptions newOptions = optionsDiff.transformOptions(options);
      if (newOptions != null) {
        builder.add(newOptions);
      }
    }
    for (FragmentOptions extraSecondFragment : optionsDiff.extraSecondFragments) {
      builder.add(extraSecondFragment);
    }
    return builder.build();
  }

  /**
   * Creates a builder object for BuildOptions
   */
  public static Builder builder() {
    return new Builder();
  }

  /**
   * Builder class for BuildOptions.
   */
  public static class Builder {
    /**
     * Adds a new FragmentOptions instance to the builder. Overrides previous instances of the
     * exact same subclass of FragmentOptions.
     */
    public <T extends FragmentOptions> Builder add(T options) {
      builderMap.put(options.getClass(), options);
      return this;
    }

    public BuildOptions build() {
      return new BuildOptions(
          ImmutableSortedMap.copyOf(builderMap, lexicalFragmentOptionsComparator));
    }

    private Map<Class<? extends FragmentOptions>, FragmentOptions> builderMap;

    private Builder() {
      builderMap = new HashMap<>();
    }
  }

  /** Returns the difference between two BuildOptions in a new {@link BuildOptions.OptionsDiff}. */
  public static OptionsDiff diff(@Nullable BuildOptions first, @Nullable BuildOptions second) {
    return diff(new OptionsDiff(), first, second);
  }

  /**
   * Returns the difference between two BuildOptions in a pre-existing {@link
   * BuildOptions.OptionsDiff}.
   *
   * <p>In a single pass through this method, the method can only compare a single "first" {@link
   * BuildOptions} and single "second" BuildOptions; but an OptionsDiff instance can store the diff
   * between a single "first" BuildOptions and multiple "second" BuildOptions. Being able to
   * maintain a single OptionsDiff over multiple calls to diff is useful for, for example,
   * aggregating the difference between a single BuildOptions and the results of applying a {@link
   * com.google.devtools.build.lib.analysis.config.transitions.SplitTransition}) to it.
   */
  public static OptionsDiff diff(
      OptionsDiff diff, @Nullable BuildOptions first, @Nullable BuildOptions second) {
    if (first == null || second == null) {
      throw new IllegalArgumentException("Cannot diff null BuildOptions");
    }
    if (first.equals(second)) {
      return diff;
    }
    // Check and report if either class has been trimmed of an options class that exists in the
    // other.
    ImmutableSet<Class<? extends FragmentOptions>> firstOptionClasses =
        first.getOptions()
            .stream()
            .map(FragmentOptions::getClass)
            .collect(ImmutableSet.toImmutableSet());
    ImmutableSet<Class<? extends FragmentOptions>> secondOptionClasses =
        second.getOptions()
            .stream()
            .map(FragmentOptions::getClass)
            .collect(ImmutableSet.toImmutableSet());
    Sets.difference(firstOptionClasses, secondOptionClasses).forEach(diff::addExtraFirstFragment);
    Sets.difference(secondOptionClasses, firstOptionClasses)
        .stream()
        .map(second::get)
        .forEach(diff::addExtraSecondFragment);
    // For fragments in common, report differences.
    for (Class<? extends FragmentOptions> clazz :
        Sets.intersection(firstOptionClasses, secondOptionClasses)) {
      if (!first.get(clazz).equals(second.get(clazz))) {
        ImmutableList<OptionDefinition> definitions = OptionsParser.getOptionDefinitions(clazz);
        Map<String, Object> firstClazzOptions = first.get(clazz).asMap();
        Map<String, Object> secondClazzOptions = second.get(clazz).asMap();
        for (OptionDefinition definition : definitions) {
          String name = definition.getOptionName();
          Object firstValue = firstClazzOptions.get(name);
          Object secondValue = secondClazzOptions.get(name);
          if (!Objects.equals(firstValue, secondValue)) {
            diff.addDiff(clazz, definition, firstValue, secondValue);
          }
        }
      }
    }
    return diff;
  }

  /**
   * Returns a {@link OptionsDiffForReconstruction} object that can be applied to {@code first} via
   * {@link #applyDiff} to get a {@link BuildOptions} object equal to {@code second}.
   */
  public static OptionsDiffForReconstruction diffForReconstruction(
      BuildOptions first, BuildOptions second) {
    OptionsDiff diff = diff(first, second);
    if (diff.areSame()) {
      return OptionsDiffForReconstruction.getEmpty(first.fingerprint, second.computeChecksum());
    }
    LinkedHashMap<Class<? extends FragmentOptions>, Map<String, Object>> differingOptions =
        new LinkedHashMap<>(diff.differingOptions.keySet().size());
    for (Class<? extends FragmentOptions> clazz : diff.differingOptions.keySet()) {
      Collection<OptionDefinition> fields = diff.differingOptions.get(clazz);
      LinkedHashMap<String, Object> valueMap = new LinkedHashMap<>(fields.size());
      for (OptionDefinition optionDefinition : fields) {
        Object secondValue;
        try {
          secondValue = Iterables.getOnlyElement(diff.second.get(optionDefinition));
        } catch (IllegalArgumentException e) {
          // TODO(janakr): Currently this exception should never be thrown since diff is never
          // constructed using the diff method that takes in a preexisting OptionsDiff. If this
          // changes, add a test verifying this error catching works properly.
          throw new IllegalStateException(
              "OptionsDiffForReconstruction can only handle a single first BuildOptions and a "
                  + "single second BuildOptions and has encountered multiple second BuildOptions");
        }
        valueMap.put(optionDefinition.getField().getName(), secondValue);
      }
      differingOptions.put(clazz, valueMap);
    }
    first.maybeInitializeFingerprintAndHashCode();
    return new OptionsDiffForReconstruction(
        differingOptions,
        ImmutableSet.copyOf(diff.extraFirstFragments),
        ImmutableList.copyOf(diff.extraSecondFragments),
        first.fingerprint,
        second.computeChecksum());
  }

  /**
   * A diff class for BuildOptions. Fields are meant to be populated and returned by {@link
   * BuildOptions#diff}
   */
  public static class OptionsDiff{
    private final Multimap<Class<? extends FragmentOptions>, OptionDefinition> differingOptions =
        ArrayListMultimap.create();
    // The keyset for the {@link first} and {@link second} maps are identical and indicate which
    // specific options differ between the first and second built options.
    private final Map<OptionDefinition, Object> first = new LinkedHashMap<>();
    // Since this class can be used to track the result of transitions, {@link second} is a multimap
    // to be able to handle [@link SplitTransition}s.
    private final Multimap<OptionDefinition, Object> second = OrderedSetMultimap.create();
    // List of "extra" fragments for each BuildOption aka fragments that were trimmed off one
    // BuildOption but not the other.
    private final Set<Class<? extends FragmentOptions>> extraFirstFragments = new HashSet<>();
    private final Set<FragmentOptions> extraSecondFragments = new HashSet<>();

    private void addExtraFirstFragment(Class<? extends FragmentOptions> options) {
      extraFirstFragments.add(options);
    }

    private void addExtraSecondFragment(FragmentOptions options) {
      extraSecondFragments.add(options);
    }

    /** Return the extra fragments classes from the first configuration. */
    public Set<Class<? extends FragmentOptions>> getExtraFirstFragmentClasses() {
      return extraFirstFragments;
    }

    /** Return the extra fragments classes from the second configuration. */
    public Set<Class<?>> getExtraSecondFragmentClasses() {
      return extraSecondFragments.stream().map(Object::getClass).collect(Collectors.toSet());
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

    /**
     * Note: it's not enough for first and second to be empty, with trimming, they must also contain
     * the same options classes.
     */
    boolean areSame() {
      return first.isEmpty()
          && second.isEmpty()
          && extraSecondFragments.isEmpty()
          && extraFirstFragments.isEmpty()
          && differingOptions.isEmpty();
    }

    public String prettyPrint() {
      StringBuilder toReturn = new StringBuilder();
      for (String diff : getPrettyPrintList()) {
        toReturn.append(diff).append(System.lineSeparator());
      }
      return toReturn.toString();
    }

    public List<String> getPrettyPrintList() {
      List<String> toReturn = new ArrayList<>();
        first.forEach(
            (option, value) ->
                toReturn.add(
                    option.getOptionName() + ":" + value + " -> " + second.get(option)));
      return toReturn;
    }
  }

  /**
   * An object that encapsulates the data needed to transform one {@link BuildOptions} object into
   * another: the full fragments of the second one, the fragment classes of the first that should be
   * omitted, and the values of any fields that should be changed.
   */
  public static class OptionsDiffForReconstruction {
    private final Map<Class<? extends FragmentOptions>, Map<String, Object>> differingOptions;
    private final ImmutableSet<Class<? extends FragmentOptions>> extraFirstFragmentClasses;
    private final ImmutableList<FragmentOptions> extraSecondFragments;
    private final byte[] baseFingerprint;
    private final String checksum;

    OptionsDiffForReconstruction(
        Map<Class<? extends FragmentOptions>, Map<String, Object>> differingOptions,
        ImmutableSet<Class<? extends FragmentOptions>> extraFirstFragmentClasses,
        ImmutableList<FragmentOptions> extraSecondFragments,
        byte[] baseFingerprint,
        String checksum) {
      this.differingOptions = differingOptions;
      this.extraFirstFragmentClasses = extraFirstFragmentClasses;
      this.extraSecondFragments = extraSecondFragments;
      this.baseFingerprint = baseFingerprint;
      this.checksum = checksum;
    }

    private static OptionsDiffForReconstruction getEmpty(byte[] baseFingerprint, String checksum) {
      return new OptionsDiffForReconstruction(
          ImmutableMap.of(), ImmutableSet.of(), ImmutableList.of(), baseFingerprint, checksum);
    }

    @Nullable
    @VisibleForTesting
    FragmentOptions transformOptions(FragmentOptions input) {
      Class<? extends FragmentOptions> clazz = input.getClass();
      if (extraFirstFragmentClasses.contains(clazz)) {
        return null;
      }
      Map<String, Object> changedOptions = differingOptions.get(clazz);
      if (changedOptions == null || changedOptions.isEmpty()) {
        return input;
      }
      FragmentOptions newOptions = input.clone();
      for (Map.Entry<String, Object> entry : changedOptions.entrySet()) {
        try {
          clazz.getField(entry.getKey()).set(newOptions, entry.getValue());
        } catch (IllegalAccessException | NoSuchFieldException e) {
          throw new IllegalStateException("Couldn't set " + entry + " for " + newOptions, e);
        }
      }
      return newOptions;
    }

    public String getChecksum() {
      return checksum;
    }

    private boolean isEmpty() {
      return differingOptions.isEmpty()
          && extraFirstFragmentClasses.isEmpty()
          && extraSecondFragments.isEmpty();
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (o == null || getClass() != o.getClass()) {
        return false;
      }
      OptionsDiffForReconstruction that = (OptionsDiffForReconstruction) o;
      return differingOptions.equals(that.differingOptions)
          && extraFirstFragmentClasses.equals(that.extraFirstFragmentClasses)
          && this.extraSecondFragments.equals(that.extraSecondFragments)
          && Arrays.equals(this.baseFingerprint, that.baseFingerprint)
          && this.checksum.equals(that.checksum);
    }

    @Override
    public String toString() {
      return MoreObjects.toStringHelper(this)
          .add("differingOptions", differingOptions)
          .add("extraFirstFragmentClasses", extraFirstFragmentClasses)
          .add("extraSecondFragments", extraSecondFragments).toString();
    }

    @Override
    public int hashCode() {
      return Objects.hash(
          differingOptions,
          extraFirstFragmentClasses,
          extraSecondFragments,
          Arrays.hashCode(baseFingerprint),
          checksum);
    }
  }

  /**
   * Hand-rolled Codec so we can cache the byte representation of a {@link
   * BuildOptions.OptionsDiffForReconstruction} object because serialization is expensive.
   */
  @VisibleForTesting
  static class OptionsDiffForReconstructionCodec
      implements ObjectCodec<OptionsDiffForReconstruction> {

    @Override
    public void serialize(
        SerializationContext context,
        BuildOptions.OptionsDiffForReconstruction input,
        CodedOutputStream codedOut)
        throws SerializationException, IOException {
      context = context.getNewNonMemoizingContext();
      // We get this cache from our context because there can be different ObjectCodecRegistry's for
      // SkyKeys and SkyValues.
      @SuppressWarnings("unchecked")
      IdentityHashMap<OptionsDiffForReconstruction, byte[]> cache =
          context.getDependency(IdentityHashMap.class);
      if (cache.containsKey(input)) {
        byte[] rawBytes = cache.get(input);
        codedOut.writeInt32NoTag(rawBytes.length);
        codedOut.writeRawBytes(cache.get(input));
      } else {
        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
        CodedOutputStream codedOutputStream = CodedOutputStream.newInstance(byteArrayOutputStream);
        context.serialize(input.differingOptions, codedOutputStream);
        context.serialize(input.extraFirstFragmentClasses, codedOutputStream);
        context.serialize(input.extraSecondFragments, codedOutputStream);
        if (input.baseFingerprint != null) {
          codedOutputStream.writeBoolNoTag(true);
          codedOutputStream.writeInt32NoTag(input.baseFingerprint.length);
          codedOutputStream.writeRawBytes(input.baseFingerprint);
        } else {
          codedOutputStream.writeBoolNoTag(false);
        }
        context.serialize(input.checksum, codedOutputStream);
        codedOutputStream.flush();
        byteArrayOutputStream.flush();
        byte[] serializedBytes = byteArrayOutputStream.toByteArray();
        cache.put(input, serializedBytes);
        codedOut.writeInt32NoTag(serializedBytes.length);
        codedOut.writeRawBytes(serializedBytes);
        codedOut.flush();
      }
    }

    @Override
    public BuildOptions.OptionsDiffForReconstruction deserialize(
        DeserializationContext context, CodedInputStream codedIn)
        throws SerializationException, IOException {
      byte[] serializedBytes = codedIn.readRawBytes(codedIn.readInt32());
      CodedInputStream codedInputStream = CodedInputStream.newInstance(serializedBytes);
      context = context.getNewNonMemoizingContext();
      Map<Class<? extends FragmentOptions>, Map<String, Object>> differingOptions =
          context.deserialize(codedInputStream);
      ImmutableSet<Class<? extends FragmentOptions>> extraFirstFragmentClasses =
          context.deserialize(codedInputStream);
      ImmutableList<FragmentOptions> extraSecondFragments = context.deserialize(codedInputStream);
      byte[] baseFingerprint = null;
      if (codedInputStream.readBool()) {
        baseFingerprint = codedInputStream.readRawBytes(codedInputStream.readInt32());
      }
      String checksum = context.deserialize(codedInputStream);
      return new OptionsDiffForReconstruction(
          differingOptions,
          extraFirstFragmentClasses,
          extraSecondFragments,
          baseFingerprint,
          checksum);
    }

    @Override
    public Class<BuildOptions.OptionsDiffForReconstruction> getEncodedClass() {
      return BuildOptions.OptionsDiffForReconstruction.class;
    }
  }
}
