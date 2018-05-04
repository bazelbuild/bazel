// Copyright 2015 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.cpp;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.base.Strings;
import com.google.common.base.Supplier;
import com.google.common.base.Throwables;
import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableMultimap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;
import com.google.common.collect.Sets.SetView;
import com.google.common.collect.Streams;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.ArtifactExpander;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.VisibleForSerialization;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CToolchain;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.Stack;
import java.util.concurrent.ExecutionException;
import javax.annotation.Nullable;

/**
 * Provides access to features supported by a specific toolchain.
 *
 * <p>This class can be generated from the CToolchain protocol buffer.
 *
 * <p>TODO(bazel-team): Implement support for specifying the toolchain configuration directly from
 * the BUILD file.
 *
 * <p>TODO(bazel-team): Find a place to put the public-facing documentation and link to it from
 * here.
 *
 * <p>TODO(bazel-team): Split out Feature as CcToolchainFeature, which will modularize the crosstool
 * configuration into one part that is about handling a set of features (including feature
 * selection) and one part that is about how to apply a single feature (parsing flags and expanding
 * them from build variables).
 */
@Immutable
public class CcToolchainFeatures implements Serializable {

  /**
   * Thrown when a flag value cannot be expanded under a set of build variables.
   *
   * <p>This happens for example when a flag references a variable that is not provided by the
   * action, or when a flag group implicitly references multiple variables of sequence type.
   */
  public static class ExpansionException extends RuntimeException {
    ExpansionException(String message) {
      super(message);
    }
  }

  /** Thrown when multiple features provide the same string symbol. */
  public static class CollidingProvidesException extends Exception {
    CollidingProvidesException(String message) {
      super(message);
    }
  }

  /** Error message thrown when a toolchain does not provide a required artifact_name_pattern. */
  public static final String MISSING_ARTIFACT_NAME_PATTERN_ERROR_TEMPLATE =
      "Toolchain must provide artifact_name_pattern for category %s";

  /** Error message thrown when a toolchain enables two features that provide the same string. */
  public static final String COLLIDING_PROVIDES_ERROR =
      "Symbol %s is provided by all of the following features: %s";

  /**
   * A piece of a single string value.
   *
   * <p>A single value can contain a combination of text and variables (for example "-f
   * %{var1}/%{var2}"). We split the string into chunks, where each chunk represents either a text
   * snippet, or a variable that is to be replaced.
   */
  interface StringChunk {
    /**
     * Expands this chunk.
     *
     * @param variables binding of variable names to their values for a single flag expansion.
     */
    String expand(Variables variables);
  }

  /** A plain text chunk of a string (containing no variables). */
  @Immutable
  @AutoCodec
  @VisibleForSerialization
  static class StringLiteralChunk implements StringChunk, Serializable {
    private final String text;

    @VisibleForSerialization
    StringLiteralChunk(String text) {
      this.text = text;
    }

    @Override
    public String expand(Variables variables) {
      return text;
    }

    @Override
    public boolean equals(@Nullable Object object) {
      if (this == object) {
        return true;
      }
      if (object instanceof StringLiteralChunk) {
        StringLiteralChunk that = (StringLiteralChunk) object;
        return text.equals(that.text);
      }
      return false;
    }

    @Override
    public int hashCode() {
      return Objects.hash(text);
    }
  }

  /** A chunk of a string value into which a variable should be expanded. */
  @Immutable
  @AutoCodec
  static class VariableChunk implements StringChunk, Serializable {
    private final String variableName;

    @VisibleForSerialization
    VariableChunk(String variableName) {
      this.variableName = variableName;
    }

    @Override
    public String expand(Variables variables) {
      // We check all variables in FlagGroup.expandCommandLine.
      // If we arrive here with the variable not being available, the variable was provided, but
      // the nesting level of the NestedSequence was deeper than the nesting level of the flag
      // groups.
      return variables.getStringVariable(variableName);
    }

    @Override
    public boolean equals(@Nullable Object object) {
      if (this == object) {
        return true;
      }
      if (object instanceof VariableChunk) {
        VariableChunk that = (VariableChunk) object;
        return variableName.equals(that.variableName);
      }
      return false;
    }

    @Override
    public int hashCode() {
      return Objects.hash(variableName);
    }
  }
  
  /**
   * Parser for toolchain string values.
   *
   * <p>A string value contains a snippet of text supporting variable expansion. For example, a
   * string value "-f %{var1}/%{var2}" will expand the values of the variables "var1" and "var2" in
   * the corresponding places in the string.
   *
   * <p>The {@code StringValueParser} takes a string and parses it into a list of {@link
   * StringChunk} objects, where each chunk represents either a snippet of text or a variable to be
   * expanded. In the above example, the resulting chunks would be ["-f ", var1, "/", var2].
   *
   * <p>In addition to the list of chunks, the {@link StringValueParser} also provides the set of
   * variables necessary for the expansion of this flag via {@link #getUsedVariables}.
   *
   * <p>To get a literal percent character, "%%" can be used in the string.
   */
  static class StringValueParser {

    private final String value;
    
    /**
     * The current position in {@value} during parsing.
     */
    private int current = 0;
    
    private final ImmutableList.Builder<StringChunk> chunks = ImmutableList.builder();
    private final ImmutableSet.Builder<String> usedVariables = ImmutableSet.builder();
    
    StringValueParser(String value) throws InvalidConfigurationException {
      this.value = value;
      parse();
    }
    
    /** @return the parsed chunks for this string. */
    ImmutableList<StringChunk> getChunks() {
      return chunks.build();
    }
    
    /** @return all variable names needed to expand this string. */
    ImmutableSet<String> getUsedVariables() {
      return usedVariables.build();
    }
    
    /**
     * Parses the string.
     * 
     * @throws InvalidConfigurationException if there is a parsing error.
     */
    private void parse() throws InvalidConfigurationException {
      while (current < value.length()) {
        if (atVariableStart()) {
          parseVariableChunk();
        } else {
          parseStringChunk();
        }
      }
    }
    
    /**
     * @return whether the current position is the start of a variable.
     */
    private boolean atVariableStart() {
      // We parse a variable when value starts with '%', but not '%%'.
      return value.charAt(current) == '%'
          && (current + 1 >= value.length() || value.charAt(current + 1) != '%');
    }
    
    /**
     * Parses a chunk of text until the next '%', which indicates either an escaped literal '%'
     * or a variable. 
     */
    private void parseStringChunk() {
      int start = current;
      // We only parse string chunks starting with '%' if they also start with '%%'.
      // In that case, we want to have a single '%' in the string, so we start at the second
      // character.
      // Note that for strings like "abc%%def" this will lead to two string chunks, the first
      // referencing the subtring "abc", and a second referencing the substring "%def".
      if (value.charAt(current) == '%') {
        current = current + 1;
        start = current;
      }
      current = value.indexOf('%', current + 1);
      if (current == -1) {
        current = value.length();
      }
      final String text = value.substring(start, current);
      chunks.add(new StringLiteralChunk(text));
    }
    
    /**
     * Parses a variable to be expanded.
     * 
     * @throws InvalidConfigurationException if there is a parsing error.
     */
    private void parseVariableChunk() throws InvalidConfigurationException {
      current = current + 1;
      if (current >= value.length() || value.charAt(current) != '{') {
        abort("expected '{'");
      }
      current = current + 1;
      if (current >= value.length() || value.charAt(current) == '}') {
        abort("expected variable name");
      }
      int end = value.indexOf('}', current);
      final String name = value.substring(current, end);
      usedVariables.add(name);
      chunks.add(new VariableChunk(name));
      current = end + 1;
    }
    
    /**
     * @throws InvalidConfigurationException with the given error text, adding information about
     * the current position in the string.
     */
    private void abort(String error) throws InvalidConfigurationException {
      throw new InvalidConfigurationException("Invalid toolchain configuration: " + error
          + " at position " + current + " while parsing a flag containing '" + value + "'");
    }
  }

  /** A flag or flag group that can be expanded under a set of variables. */
  interface Expandable {
    /**
     * Expands the current expandable under the given {@code view}, adding new flags to {@code
     * commandLine}.
     *
     * <p>The {@code variables} controls which variables are visible during the expansion and allows
     * to recursively expand nested flag groups.
     */
    void expand(Variables variables, @Nullable ArtifactExpander expander, List<String> commandLine);
  }

  /**
   * A single flag to be expanded under a set of variables.
   */
  @Immutable
  @AutoCodec
  @VisibleForSerialization
  static class Flag implements Serializable, Expandable {
    private final ImmutableList<StringChunk> chunks;

    @VisibleForSerialization
    Flag(ImmutableList<StringChunk> chunks) {
      this.chunks = chunks;
    }

    /** Expand this flag into a single new entry in {@code commandLine}. */
    @Override
    public void expand(
        Variables variables, @Nullable ArtifactExpander expander, List<String> commandLine) {
      StringBuilder flag = new StringBuilder();
      for (StringChunk chunk : chunks) {
        flag.append(chunk.expand(variables));
      }
      commandLine.add(flag.toString());
    }

    @Override
    public boolean equals(@Nullable Object object) {
      if (this == object) {
        return true;
      }
      if (object instanceof Flag) {
        Flag that = (Flag) object;
        return Iterables.elementsEqual(chunks, that.chunks);
      }
      return false;
    }

    @Override
    public int hashCode() {
      return Objects.hash(chunks);
    }

    /** A single environment key/value pair to be expanded under a set of variables. */
    private static Expandable create(ImmutableList<StringChunk> chunks) {
      if (chunks.size() == 1) {
        return new SingleChunkFlag(chunks.get(0));
      }
      return new Flag(chunks);
    }

    /** Optimization for single-chunk case */
    @Immutable
    @AutoCodec
    @VisibleForSerialization
    static class SingleChunkFlag implements Serializable, Expandable {
      private final StringChunk chunk;

      @VisibleForSerialization
      SingleChunkFlag(StringChunk chunk) {
        this.chunk = chunk;
      }

      @Override
      public void expand(
          Variables variables,
          @Nullable ArtifactExpander artifactExpander,
          List<String> commandLine) {
        commandLine.add(chunk.expand(variables));
      }

      @Override
      public boolean equals(Object o) {
        if (this == o) {
          return true;
        }
        if (o == null || getClass() != o.getClass()) {
          return false;
        }
        SingleChunkFlag that = (SingleChunkFlag) o;
        return chunk.equals(that.chunk);
      }

      @Override
      public int hashCode() {
        return chunk.hashCode();
      }
    }
  }

  /** A single environment key/value pair to be expanded under a set of variables. */
  @Immutable
  @AutoCodec
  @VisibleForSerialization
  static class EnvEntry implements Serializable {
    private final String key;
    private final ImmutableList<StringChunk> valueChunks;

    private EnvEntry(CToolchain.EnvEntry envEntry) throws InvalidConfigurationException {
      this.key = envEntry.getKey();
      StringValueParser parser = new StringValueParser(envEntry.getValue());
      this.valueChunks = parser.getChunks();
    }

    @AutoCodec.Instantiator
    @VisibleForSerialization
    EnvEntry(String key, ImmutableList<StringChunk> valueChunks) {
      this.key = key;
      this.valueChunks = valueChunks;
    }

    /**
     * Adds the key/value pair this object represents to the given map of environment variables.
     * The value of the entry is expanded with the given {@code variables}.
     */
    public void addEnvEntry(Variables variables, ImmutableMap.Builder<String, String> envBuilder) {
      StringBuilder value = new StringBuilder();
      for (StringChunk chunk : valueChunks) {
        value.append(chunk.expand(variables));
      }
      envBuilder.put(key, value.toString());
    }

    @Override
    public boolean equals(@Nullable Object object) {
      if (this == object) {
        return true;
      }
      if (object instanceof EnvEntry) {
        EnvEntry that = (EnvEntry) object;
        return Objects.equals(key, that.key)
            && Iterables.elementsEqual(valueChunks, that.valueChunks);
      }
      return false;
    }

    @Override
    public int hashCode() {
      return Objects.hash(key, valueChunks);
    }
  }

  @Immutable
  @AutoCodec
  @VisibleForSerialization
  static class VariableWithValue {
    public final String variable;
    public final String value;

    public VariableWithValue(String variable, String value) {
      this.variable = variable;
      this.value = value;
    }
  }

  /**
   * A group of flags. When iterateOverVariable is specified, we assume the variable is a sequence
   * and the flag_group will be expanded repeatedly for every value in the sequence.
   */
  @Immutable
  @AutoCodec
  @VisibleForSerialization
  static class FlagGroup implements Serializable, Expandable {
    private final ImmutableList<Expandable> expandables;
    private String iterateOverVariable;
    private final ImmutableSet<String> expandIfAllAvailable;
    private final ImmutableSet<String> expandIfNoneAvailable;
    private final String expandIfTrue;
    private final String expandIfFalse;
    private final VariableWithValue expandIfEqual;

    private FlagGroup(CToolchain.FlagGroup flagGroup) throws InvalidConfigurationException {
      ImmutableList.Builder<Expandable> expandables = ImmutableList.builder();
      Collection<String> flags = flagGroup.getFlagList();
      Collection<CToolchain.FlagGroup> groups = flagGroup.getFlagGroupList();
      if (!flags.isEmpty() && !groups.isEmpty()) {
        // If both flags and flag_groups are available, the original order is not preservable.
        throw new ExpansionException(
            "Invalid toolchain configuration: a flag_group must not contain both a flag "
                + "and another flag_group.");
      }
      for (String flag : flags) {
        StringValueParser parser = new StringValueParser(flag);
        expandables.add(Flag.create(parser.getChunks()));
      }
      for (CToolchain.FlagGroup group : groups) {
        FlagGroup subgroup = new FlagGroup(group);
        expandables.add(subgroup);
      }
      if (flagGroup.hasIterateOver()) {
        this.iterateOverVariable = flagGroup.getIterateOver();
      }
      this.expandables = expandables.build();
      this.expandIfAllAvailable = ImmutableSet.copyOf(flagGroup.getExpandIfAllAvailableList());
      this.expandIfNoneAvailable = ImmutableSet.copyOf(flagGroup.getExpandIfNoneAvailableList());
      this.expandIfTrue = Strings.emptyToNull(flagGroup.getExpandIfTrue());
      this.expandIfFalse = Strings.emptyToNull(flagGroup.getExpandIfFalse());
      if (flagGroup.hasExpandIfEqual()) {
        this.expandIfEqual = new VariableWithValue(
            flagGroup.getExpandIfEqual().getVariable(),
            flagGroup.getExpandIfEqual().getValue());
      } else {
        this.expandIfEqual = null;
      }
    }

    @AutoCodec.Instantiator
    @VisibleForSerialization
    public FlagGroup(
        ImmutableList<Expandable> expandables,
        String iterateOverVariable,
        ImmutableSet<String> expandIfAllAvailable,
        ImmutableSet<String> expandIfNoneAvailable,
        String expandIfTrue,
        String expandIfFalse,
        VariableWithValue expandIfEqual) {
      this.expandables = expandables;
      this.iterateOverVariable = iterateOverVariable;
      this.expandIfAllAvailable = expandIfAllAvailable;
      this.expandIfNoneAvailable = expandIfNoneAvailable;
      this.expandIfTrue = expandIfTrue;
      this.expandIfFalse = expandIfFalse;
      this.expandIfEqual = expandIfEqual;
    }

    @Override
    public void expand(
        Variables variables, @Nullable ArtifactExpander expander, final List<String> commandLine) {
      if (!canBeExpanded(variables, expander)) {
        return;
      }
      if (iterateOverVariable != null) {
        for (Variables.VariableValue variableValue :
            variables.getSequenceVariable(iterateOverVariable, expander)) {
          Variables nestedVariables =
              new SingleVariables(variables, iterateOverVariable, variableValue);
          for (Expandable expandable : expandables) {
            expandable.expand(nestedVariables, expander, commandLine);
          }
        }
      } else {
        for (Expandable expandable : expandables) {
          expandable.expand(variables, expander, commandLine);
        }
      }
    }

    private boolean canBeExpanded(Variables variables, @Nullable ArtifactExpander expander) {
      for (String variable : expandIfAllAvailable) {
        if (!variables.isAvailable(variable, expander)) {
          return false;
        }
      }
      for (String variable : expandIfNoneAvailable) {
        if (variables.isAvailable(variable, expander)) {
          return false;
        }
      }
      if (expandIfTrue != null
          && (!variables.isAvailable(expandIfTrue, expander)
              || !variables.getVariable(expandIfTrue).isTruthy())) {
        return false;
      }
      if (expandIfFalse != null
          && (!variables.isAvailable(expandIfFalse, expander)
              || variables.getVariable(expandIfFalse).isTruthy())) {
        return false;
      }
      if (expandIfEqual != null
          && (!variables.isAvailable(expandIfEqual.variable, expander)
              || !variables
                  .getVariable(expandIfEqual.variable)
                  .getStringValue(expandIfEqual.variable)
                  .equals(expandIfEqual.value))) {
        return false;
      }
      return true;
    }

    /**
     * Expands all flags in this group and adds them to {@code commandLine}.
     *
     * <p>The flags of the group will be expanded either:
     *
     * <ul>
     *   <li>once, if there is no variable of sequence type in any of the group's flags, or
     *   <li>for each element in the sequence, if there is 'iterate_over' variable specified
     *       (preferred, explicit way), or
     *   <li>for each element in the sequence, if there is only one sequence variable used in the
     *       body of the flag_group (deprecated, implicit way). Having more than a single variable
     *       of sequence type in a single flag group with implicit iteration is not supported. Use
     *       explicit 'iterate_over' instead.
     * </ul>
     */
    private void expandCommandLine(
        Variables variables, @Nullable ArtifactExpander expander, final List<String> commandLine) {
      expand(variables, expander, commandLine);
    }

    @Override
    public boolean equals(@Nullable Object object) {
      if (this == object) {
        return true;
      }
      if (object instanceof FlagGroup) {
        FlagGroup that = (FlagGroup) object;
        return Iterables.elementsEqual(expandables, that.expandables)
            && Objects.equals(iterateOverVariable, that.iterateOverVariable)
            && Iterables.elementsEqual(expandIfAllAvailable, that.expandIfAllAvailable)
            && Iterables.elementsEqual(expandIfNoneAvailable, that.expandIfNoneAvailable)
            && Objects.equals(expandIfTrue, that.expandIfTrue)
            && Objects.equals(expandIfFalse, that.expandIfFalse)
            && Objects.equals(expandIfEqual, that.expandIfEqual);
      }
      return false;
    }

    @Override
    public int hashCode() {
      return Objects.hash(
          expandables,
          iterateOverVariable,
          expandIfAllAvailable,
          expandIfNoneAvailable,
          expandIfTrue,
          expandIfFalse,
          expandIfEqual);
    }
  }

  private static boolean isWithFeaturesSatisfied(
      Collection<WithFeatureSet> withFeatureSets, Set<String> enabledFeatureNames) {
    if (withFeatureSets.isEmpty()) {
      return true;
    }
    for (WithFeatureSet featureSet : withFeatureSets) {
      boolean negativeMatch =
          featureSet
              .getNotFeatures()
              .stream()
              .anyMatch(notFeature -> enabledFeatureNames.contains(notFeature));
      boolean positiveMatch = enabledFeatureNames.containsAll(featureSet.getFeatures());

      if (!negativeMatch && positiveMatch) {
        return true;
      }
    }
    return false;
  }

  /** Groups a set of flags to apply for certain actions. */
  @Immutable
  @AutoCodec
  @VisibleForSerialization
  static class FlagSet implements Serializable {
    private final ImmutableSet<String> actions;
    private final ImmutableSet<String> expandIfAllAvailable;
    private final ImmutableSet<WithFeatureSet> withFeatureSets;
    private final ImmutableList<FlagGroup> flagGroups;

    private FlagSet(CToolchain.FlagSet flagSet) throws InvalidConfigurationException {
      this(flagSet, ImmutableSet.copyOf(flagSet.getActionList()));
    }

    /**
     * Constructs a FlagSet for the given set of actions.
     */
    private FlagSet(CToolchain.FlagSet flagSet, ImmutableSet<String> actions)
        throws InvalidConfigurationException {
      this.actions = actions;
      this.expandIfAllAvailable = ImmutableSet.copyOf(flagSet.getExpandIfAllAvailableList());
      ImmutableSet.Builder<WithFeatureSet> featureSetBuilder = ImmutableSet.builder();
      for (CToolchain.WithFeatureSet withFeatureSet : flagSet.getWithFeatureList()) {
        featureSetBuilder.add(new WithFeatureSet(withFeatureSet));
      }
      this.withFeatureSets = featureSetBuilder.build();
      ImmutableList.Builder<FlagGroup> builder = ImmutableList.builder();
      for (CToolchain.FlagGroup flagGroup : flagSet.getFlagGroupList()) {
        builder.add(new FlagGroup(flagGroup));
      }
      this.flagGroups = builder.build();
    }

    @AutoCodec.Instantiator
    @VisibleForSerialization
    FlagSet(
        ImmutableSet<String> actions,
        ImmutableSet<String> expandIfAllAvailable,
        ImmutableSet<WithFeatureSet> withFeatureSets,
        ImmutableList<FlagGroup> flagGroups) {
      this.actions = actions;
      this.expandIfAllAvailable = expandIfAllAvailable;
      this.withFeatureSets = withFeatureSets;
      this.flagGroups = flagGroups;
    }

    /** Adds the flags that apply to the given {@code action} to {@code commandLine}. */
    private void expandCommandLine(
        String action,
        Variables variables,
        Set<String> enabledFeatureNames,
        @Nullable ArtifactExpander expander,
        List<String> commandLine) {
      for (String variable : expandIfAllAvailable) {
        if (!variables.isAvailable(variable, expander)) {
          return;
        }
      }
      if (!isWithFeaturesSatisfied(withFeatureSets, enabledFeatureNames)) {
        return;
      }
      if (!actions.contains(action)) {
        return;
      }
      for (FlagGroup flagGroup : flagGroups) {
        flagGroup.expandCommandLine(variables, expander, commandLine);
      }
    }

    @Override
    public boolean equals(@Nullable Object object) {
      if (object instanceof FlagSet) {
        FlagSet that = (FlagSet) object;
        return Iterables.elementsEqual(actions, that.actions)
            && Iterables.elementsEqual(expandIfAllAvailable, that.expandIfAllAvailable)
            && Iterables.elementsEqual(withFeatureSets, that.withFeatureSets)
            && Iterables.elementsEqual(flagGroups, that.flagGroups);
      }
      return false;
    }

    @Override
    public int hashCode() {
      return Objects.hash(actions, expandIfAllAvailable, withFeatureSets, flagGroups);
    }
  }

  @Immutable
  @AutoCodec
  @VisibleForSerialization
  static class WithFeatureSet implements Serializable {
    private final ImmutableSet<String> features;
    private final ImmutableSet<String> notFeatures;

    private WithFeatureSet(CToolchain.WithFeatureSet withFeatureSet) {
      this.features = ImmutableSet.copyOf(withFeatureSet.getFeatureList());
      this.notFeatures = ImmutableSet.copyOf(withFeatureSet.getNotFeatureList());
    }

    @AutoCodec.Instantiator
    @VisibleForSerialization
    WithFeatureSet(ImmutableSet<String> features, ImmutableSet<String> notFeatures) {
      this.features = features;
      this.notFeatures = notFeatures;
    }

    public ImmutableSet<String> getFeatures() {
      return features;
    }

    public ImmutableSet<String> getNotFeatures() {
      return notFeatures;
    }

    @Override
    public boolean equals(@Nullable Object object) {
      if (this == object) {
        return true;
      }
      if (object instanceof WithFeatureSet) {
        WithFeatureSet that = (WithFeatureSet) object;
        return Iterables.elementsEqual(features, that.features)
            && Iterables.elementsEqual(notFeatures, that.notFeatures);
      }
      return false;
    }

    @Override
    public int hashCode() {
      return Objects.hash(features, notFeatures);
    }
  }

  /** Groups a set of environment variables to apply for certain actions. */
  @Immutable
  @AutoCodec
  @VisibleForSerialization
  static class EnvSet implements Serializable {
    private final ImmutableSet<String> actions;
    private final ImmutableList<EnvEntry> envEntries;
    private final ImmutableSet<WithFeatureSet> withFeatureSets;

    private EnvSet(CToolchain.EnvSet envSet) throws InvalidConfigurationException {
      this.actions = ImmutableSet.copyOf(envSet.getActionList());
      ImmutableList.Builder<EnvEntry> builder = ImmutableList.builder();
      for (CToolchain.EnvEntry envEntry : envSet.getEnvEntryList()) {
        builder.add(new EnvEntry(envEntry));
      }
      ImmutableSet.Builder<WithFeatureSet> withFeatureSetsBuilder = ImmutableSet.builder();
      for (CToolchain.WithFeatureSet withFeatureSet : envSet.getWithFeatureList()) {
        withFeatureSetsBuilder.add(new WithFeatureSet(withFeatureSet));
      }

      this.envEntries = builder.build();
      this.withFeatureSets = withFeatureSetsBuilder.build();
    }

    @AutoCodec.Instantiator
    @VisibleForSerialization
    EnvSet(
        ImmutableSet<String> actions,
        ImmutableList<EnvEntry> envEntries,
        ImmutableSet<WithFeatureSet> withFeatureSets) {
      this.actions = actions;
      this.envEntries = envEntries;
      this.withFeatureSets = withFeatureSets;
    }

    /**
     * Adds the environment key/value pairs that apply to the given {@code action} to
     * {@code envBuilder}.
     */
    private void expandEnvironment(
        String action,
        Variables variables,
        Set<String> enabledFeatureNames,
        ImmutableMap.Builder<String, String> envBuilder) {
      if (!actions.contains(action)) {
        return;
      }
      if (!isWithFeaturesSatisfied(withFeatureSets, enabledFeatureNames)) {
        return;
      }
      for (EnvEntry envEntry : envEntries) {
        envEntry.addEnvEntry(variables, envBuilder);
      }
    }

    @Override
    public boolean equals(@Nullable Object object) {
      if (this == object) {
        return true;
      }
      if (object instanceof EnvSet) {
        EnvSet that = (EnvSet) object;
        return Iterables.elementsEqual(actions, that.actions)
            && Iterables.elementsEqual(envEntries, that.envEntries)
            && Iterables.elementsEqual(withFeatureSets, that.withFeatureSets);
      }
      return false;
    }

    @Override
    public int hashCode() {
      return Objects.hash(actions, envEntries, withFeatureSets);
    }
  }

  /**
   * An interface for classes representing crosstool messages that can activate each other using
   * 'requires' and 'implies' semantics.
   *
   * <p>Currently there are two types of CrosstoolActivatable: Feature and ActionConfig.
   */
  interface CrosstoolSelectable {

    /**
     * Returns the name of this selectable.
     */
    String getName();
  }

  /** Contains flags for a specific feature. */
  @Immutable
  @AutoCodec
  @VisibleForSerialization
  static class Feature implements Serializable, CrosstoolSelectable {
    private final String name;
    private final ImmutableList<FlagSet> flagSets;
    private final ImmutableList<EnvSet> envSets;
    
    private Feature(CToolchain.Feature feature) throws InvalidConfigurationException {
      this.name = feature.getName();
      ImmutableList.Builder<FlagSet> flagSetBuilder = ImmutableList.builder();
      for (CToolchain.FlagSet flagSet : feature.getFlagSetList()) {
        flagSetBuilder.add(new FlagSet(flagSet));
      }
      this.flagSets = flagSetBuilder.build();
      
      ImmutableList.Builder<EnvSet> envSetBuilder = ImmutableList.builder();
      for (CToolchain.EnvSet flagSet : feature.getEnvSetList()) {
        envSetBuilder.add(new EnvSet(flagSet));
      }
      this.envSets = envSetBuilder.build();
    }

    @AutoCodec.Instantiator
    @VisibleForSerialization
    Feature(String name, ImmutableList<FlagSet> flagSets, ImmutableList<EnvSet> envSets) {
      this.name = name;
      this.flagSets = flagSets;
      this.envSets = envSets;
    }

    @Override
    public String getName() {
      return name;
    }

    /** Adds environment variables for the given action to the provided builder. */
    private void expandEnvironment(
        String action,
        Variables variables,
        Set<String> enabledFeatureNames,
        ImmutableMap.Builder<String, String> envBuilder) {
      for (EnvSet envSet : envSets) {
        envSet.expandEnvironment(action, variables, enabledFeatureNames, envBuilder);
      }
    }

    /** Adds the flags that apply to the given {@code action} to {@code commandLine}. */
    private void expandCommandLine(
        String action,
        Variables variables,
        Set<String> enabledFeatureNames,
        @Nullable ArtifactExpander expander,
        List<String> commandLine) {
      for (FlagSet flagSet : flagSets) {
        flagSet.expandCommandLine(action, variables, enabledFeatureNames, expander, commandLine);
      }
    }

    @Override
    public boolean equals(@Nullable Object object) {
      if (this == object) {
        return true;
      }
      if (object instanceof Feature) {
        Feature that = (Feature) object;
        return name.equals(that.name)
            && Iterables.elementsEqual(flagSets, that.flagSets)
            && Iterables.elementsEqual(envSets, that.envSets);
      }
      return false;
    }

    @Override
    public int hashCode() {
      return Objects.hash(name, flagSets, envSets);
    }
  }

  /**
   * An executable to be invoked by a blaze action. Can carry information on its platform
   * restrictions.
   */
  @Immutable
  public static class Tool {
    private final PathFragment toolPathFragment;
    private final ImmutableSet<String> executionRequirements;
    private final ImmutableSet<WithFeatureSet> withFeatureSetSets;

    private Tool(
        CToolchain.Tool tool,
        PathFragment crosstoolTop,
        ImmutableSet<WithFeatureSet> withFeatureSetSets) {
      this.withFeatureSetSets = withFeatureSetSets;
      toolPathFragment = crosstoolTop.getRelative(tool.getToolPath());
      executionRequirements = ImmutableSet.copyOf(tool.getExecutionRequirementList());
    }

    @VisibleForTesting
    public Tool(
        PathFragment toolPathFragment,
        ImmutableSet<String> executionRequirements,
        ImmutableSet<WithFeatureSet> withFeatureSetSets) {
      this.toolPathFragment = toolPathFragment;
      this.executionRequirements = executionRequirements;
      this.withFeatureSetSets = withFeatureSetSets;
    }

    /** Returns the path to this action's tool relative to the provided crosstool path. */
    public PathFragment getToolPathFragment() {
      return toolPathFragment;
    }

    /**
     * Returns a list of requirement hints that apply to the execution of this tool.
     */
    ImmutableSet<String> getExecutionRequirements() {
      return executionRequirements;
    }

    /**
     * Returns a set of {@link WithFeatureSet} instances used to decide whether to use this tool
     * given a set of enabled features.
     */
    public ImmutableSet<WithFeatureSet> getWithFeatureSetSets() {
      return withFeatureSetSets;
    }
  }

  /**
   * A container for information on a particular blaze action.
   *
   * <p>An ActionConfig can select a tool for its blaze action based on the set of active features.
   * Internally, an ActionConfig maintains an ordered list (the order being that of the list of
   * tools in the crosstool action_config message) of such tools and the feature sets for which they
   * are valid. For a given feature configuration, the ActionConfig will consider the first tool in
   * that list with a feature set that matches the configuration to be the tool for its blaze
   * action.
   *
   * <p>ActionConfigs can be activated by features. That is, a particular feature can cause an
   * ActionConfig to be applied in its "implies" field. Blaze may include certain actions in the
   * action graph only if a corresponding ActionConfig is activated in the toolchain - this provides
   * the crosstool with a mechanism for adding certain actions to the action graph based on feature
   * configuration.
   *
   * <p>It is invalid for a toolchain to contain two action configs for the same blaze action. In
   * that case, blaze will throw an error when it consumes the crosstool.
   */
  @Immutable
  @AutoCodec
  static class ActionConfig implements Serializable, CrosstoolSelectable {
    public static final String FLAG_SET_WITH_ACTION_ERROR =
        "action_config %s specifies actions.  An action_config's flag sets automatically apply "
            + "to the configured action.  Thus, you must not specify action lists in an "
            + "action_config's flag set.";

    private final String configName;
    private final String actionName;
    private final ImmutableList<Tool> tools;
    private final ImmutableList<FlagSet> flagSets;

    private ActionConfig(CToolchain.ActionConfig actionConfig, PathFragment crosstoolTop)
        throws InvalidConfigurationException {
      this.configName = actionConfig.getConfigName();
      this.actionName = actionConfig.getActionName();
      this.tools =
          actionConfig
              .getToolList()
              .stream()
              .map(
                  t ->
                      new Tool(
                          t,
                          crosstoolTop,
                          t.getWithFeatureList()
                              .stream()
                              .map(f -> new WithFeatureSet(f))
                              .collect(ImmutableSet.toImmutableSet())))
              .collect(ImmutableList.toImmutableList());

      ImmutableList.Builder<FlagSet> flagSetBuilder = ImmutableList.builder();
      for (CToolchain.FlagSet flagSet : actionConfig.getFlagSetList()) {
        if (!flagSet.getActionList().isEmpty()) {
          throw new InvalidConfigurationException(
              String.format(FLAG_SET_WITH_ACTION_ERROR, configName));
        }

        flagSetBuilder.add(new FlagSet(flagSet, ImmutableSet.of(actionName)));
      }
      this.flagSets = flagSetBuilder.build();
    }

    @AutoCodec.Instantiator
    @VisibleForSerialization
    ActionConfig(
        String configName,
        String actionName,
        ImmutableList<Tool> tools,
        ImmutableList<FlagSet> flagSets) {
      this.configName = configName;
      this.actionName = actionName;
      this.tools = tools;
      this.flagSets = flagSets;
    }

    @Override
    public String getName() {
      return configName;
    }

    /**
     * Returns the name of the blaze action this action config applies to.
     */
    String getActionName() {
      return actionName;
    }

    /**
     * Returns the path to this action's tool relative to the provided crosstool path given a set
     * of enabled features.
     */
    private Tool getTool(final Set<String> enabledFeatureNames) {
      Optional<Tool> tool =
          tools
              .stream()
              .filter(t -> isWithFeaturesSatisfied(t.getWithFeatureSetSets(), enabledFeatureNames))
              .findFirst();
      if (tool.isPresent()) {
        return tool.get();
      } else {
        throw new IllegalArgumentException(
            "Matching tool for action "
                + getActionName()
                + " not "
                + "found for given feature configuration");
      }
    }

    /** Adds the flags that apply to this action to {@code commandLine}. */
    private void expandCommandLine(
        Variables variables,
        Set<String> enabledFeatureNames,
        @Nullable ArtifactExpander expander,
        List<String> commandLine) {
      for (FlagSet flagSet : flagSets) {
        flagSet.expandCommandLine(
            actionName, variables, enabledFeatureNames, expander, commandLine);
      }
    }
  }

  /** A description of how artifacts of a certain type are named. */
  @Immutable
  private static class ArtifactNamePattern {

    private final ArtifactCategory artifactCategory;
    private final ImmutableList<StringChunk> chunks;

    private ArtifactNamePattern(CToolchain.ArtifactNamePattern artifactNamePattern)
        throws InvalidConfigurationException {

      ArtifactCategory foundCategory = null;
      for (ArtifactCategory artifactCategory : ArtifactCategory.values()) {
        if (artifactNamePattern.getCategoryName().equals(artifactCategory.getCategoryName())) {
          foundCategory = artifactCategory;
        }
      }
      if (foundCategory == null) {
        throw new InvalidConfigurationException(
            String.format(
                "Invalid toolchain configuration: Artifact category %s not recognized",
                artifactNamePattern.getCategoryName()));
      }
      this.artifactCategory = foundCategory;
      
      StringValueParser parser = new StringValueParser(artifactNamePattern.getPattern());
      this.chunks = parser.getChunks();
    }

    /** Returns the ArtifactCategory for this ArtifactNamePattern. */
    ArtifactCategory getArtifactCategory() {
      return this.artifactCategory;
    }

    /**
     * Returns the artifact name that this pattern selects.
     */
    public String getArtifactName(Map<String, String> variables) {
      StringBuilder resultBuilder = new StringBuilder();
      Variables artifactNameVariables =
          new Variables.Builder().addAllStringVariables(variables).build();
      for (StringChunk chunk : chunks) {
        resultBuilder.append(chunk.expand(artifactNameVariables));
      }
      String result = resultBuilder.toString();
      return result.charAt(0) == '/' ? result.substring(1) : result;
    }
  }

  /**
   * Configured build variables usable by the toolchain configuration.
   *
   * <p>TODO(b/32655571): Investigate cleanup once implicit iteration is not needed. Variables
   * instance could serve as a top level View used to expand all flag_groups.
   */
  @SkylarkModule(
    name = "variables",
    documented = false,
    category = SkylarkModuleCategory.BUILTIN,
    doc = "Class encapsulating build variables."
  )
  public abstract static class Variables {
    /** An empty variables instance. */
    public static final Variables EMPTY = new Variables.Builder().build();

    /**
     * Retrieves a {@link StringSequence} variable named {@code variableName} from {@code variables}
     * and converts it into a list of plain strings.
     *
     * <p>Throws {@link ExpansionException} when the variable is not a {@link StringSequence}.
     */
    public static final ImmutableList<String> toStringList(
        CcToolchainFeatures.Variables variables, String variableName) {
      return Streams
          .stream(variables.getSequenceVariable(variableName))
          .map(variable -> variable.getStringValue(variableName))
          .collect(ImmutableList.toImmutableList());
    }

    /**
     * Get a variable value named @param name. Supports accessing fields in structures (e.g.
     * 'libraries_to_link.interface_libraries')
     *
     * @throws ExpansionException when no such variable or no such field are present, or when
     *     accessing a field of non-structured variable
     */
    VariableValue getVariable(String name) {
      return lookupVariable(name, true, null);
    }

    VariableValue getVariable(String name, @Nullable ArtifactExpander expander) {
      return lookupVariable(name, true, expander);
    }

    /**
     * Lookup a variable named @param name or return a reason why the variable was not found.
     * Supports accessing fields in structures.
     *
     * @return Pair<VariableValue, String> returns either (variable value, null) or (null, string
     *     reason why variable was not found)
     */
    private VariableValue lookupVariable(
        String name, boolean throwOnMissingVariable, @Nullable ArtifactExpander expander) {
      VariableValue nonStructuredVariable = getNonStructuredVariable(name);
      if (nonStructuredVariable != null) {
        return nonStructuredVariable;
      }
      VariableValue structuredVariable =
          getStructureVariable(name, throwOnMissingVariable, expander);
      if (structuredVariable != null) {
        return structuredVariable;
      } else if (throwOnMissingVariable) {
        throw new ExpansionException(
            String.format(
                "Invalid toolchain configuration: Cannot find variable named '%s'.", name));
      } else {
        return null;
      }
    }

    private VariableValue getStructureVariable(
        String name, boolean throwOnMissingVariable, @Nullable ArtifactExpander expander) {
      if (!name.contains(".")) {
        return null;
      }

      Stack<String> fieldsToAccess = new Stack<>();
      String structPath = name;
      VariableValue variable;

      do {
        fieldsToAccess.push(structPath.substring(structPath.lastIndexOf('.') + 1));
        structPath = structPath.substring(0, structPath.lastIndexOf('.'));
        variable = getNonStructuredVariable(structPath);
      } while (variable == null && structPath.contains("."));

      if (variable == null) {
        return null;
      }

      while (!fieldsToAccess.empty()) {
        String field = fieldsToAccess.pop();
        variable = variable.getFieldValue(structPath, field, expander);
        if (variable == null) {
          if (throwOnMissingVariable) {
            throw new ExpansionException(
                String.format(
                    "Invalid toolchain configuration: Cannot expand variable '%s.%s': structure %s "
                        + "doesn't have a field named '%s'",
                    structPath, field, structPath, field));
          } else {
            return null;
          }
        }
      }
      return variable;
    }

    public String getStringVariable(String variableName) {
      return getVariable(variableName, null).getStringValue(variableName);
    }

    public Iterable<? extends VariableValue> getSequenceVariable(String variableName) {
      return getVariable(variableName, null).getSequenceValue(variableName);
    }

    public Iterable<? extends VariableValue> getSequenceVariable(
        String variableName, @Nullable ArtifactExpander expander) {
      return getVariable(variableName, expander).getSequenceValue(variableName);
    }

    /** Returns whether {@code variable} is set. */
    boolean isAvailable(String variable) {
      return isAvailable(variable, null);
    }

    boolean isAvailable(String variable, @Nullable ArtifactExpander expander) {
      return lookupVariable(variable, false, expander) != null;
    }

    abstract Map<String, VariableValue> getVariablesMap();

    abstract Map<String, String> getStringVariablesMap();

    @Nullable
    abstract VariableValue getNonStructuredVariable(String name);

    /**
     * Value of a build variable exposed to the CROSSTOOL used for flag expansion.
     *
     * <p>{@link VariableValue} represent either primitive values or an arbitrarily deeply nested
     * recursive structures or sequences. Since there are builds with millions of values, some
     * implementations might exist only to optimize memory usage.
     *
     * <p>Implementations must be immutable and without any side-effects. They will be expanded and
     * queried multiple times.
     */
    interface VariableValue {
      /**
       * Returns string value of the variable, if the variable type can be converted to string (e.g.
       * StringValue), or throw exception if it cannot (e.g. Sequence).
       *
       * @param variableName name of the variable value at hand, for better exception message.
       */
      String getStringValue(String variableName);

      /**
       * Returns Iterable value of the variable, if the variable type can be converted to a Iterable
       * (e.g. Sequence), or throw exception if it cannot (e.g. StringValue).
       *
       * @param variableName name of the variable value at hand, for better exception message.
       */
      Iterable<? extends VariableValue> getSequenceValue(String variableName);

      /**
       * Returns value of the field, if the variable is of struct type or throw exception if it is
       * not or no such field exists.
       *
       * @param variableName name of the variable value at hand, for better exception message.
       */
      VariableValue getFieldValue(String variableName, String field);

      VariableValue getFieldValue(
          String variableName, String field, @Nullable ArtifactExpander expander);

      /** Returns true if the variable is truthy */
      boolean isTruthy();
    }

    /**
     * Adapter for {@link VariableValue} predefining error handling methods. Override {@link
     * #getVariableTypeName()}, {@link #isTruthy()}, and one of {@link #getFieldValue(String,
     * String)}, {@link #getSequenceValue(String)}, or {@link #getStringValue(String)}, and you'll
     * get error handling for the other methods for free.
     */
    abstract static class VariableValueAdapter implements VariableValue {

      /** Returns human-readable variable type name to be used in error messages. */
      public abstract String getVariableTypeName();

      @Override
      public abstract boolean isTruthy();

      @Override
      public VariableValue getFieldValue(String variableName, String field) {
        return getFieldValue(variableName, field, null);
      }

      @Override
      public VariableValue getFieldValue(
          String variableName, String field, @Nullable ArtifactExpander expander) {
        throw new ExpansionException(
            String.format(
                "Invalid toolchain configuration: Cannot expand variable '%s.%s': variable '%s' is "
                    + "%s, expected structure",
                variableName, field, variableName, getVariableTypeName()));
      }

      @Override
      public String getStringValue(String variableName) {
        throw new ExpansionException(
            String.format(
                "Invalid toolchain configuration: Cannot expand variable '%s': expected string, "
                    + "found %s",
                variableName, getVariableTypeName()));
      }

      @Override
      public Iterable<? extends VariableValue> getSequenceValue(String variableName) {
        throw new ExpansionException(
            String.format(
                "Invalid toolchain configuration: Cannot expand variable '%s': expected sequence, "
                    + "found %s",
                variableName, getVariableTypeName()));
      }
    }

    /** Interface for VariableValue builders */
    public interface VariableValueBuilder {
      VariableValue build();
    }

    /** Builder for StringSequence. */
    public static class StringSequenceBuilder implements VariableValueBuilder {

      private final ImmutableList.Builder<String> values = ImmutableList.builder();

      /** Adds a value to the sequence. */
      public StringSequenceBuilder addValue(String value) {
        values.add(value);
        return this;
      }

      /** Returns an immutable string sequence. */
      @Override
      public StringSequence build() {
        return new StringSequence(values.build());
      }
    }

    /** Builder for Sequence. */
    public static class SequenceBuilder implements VariableValueBuilder {

      private final ImmutableList.Builder<VariableValue> values = ImmutableList.builder();

      /** Adds a value to the sequence. */
      public SequenceBuilder addValue(VariableValue value) {
        values.add(value);
        return this;
      }

      /** Adds a value to the sequence. */
      public SequenceBuilder addValue(VariableValueBuilder value) {
        Preconditions.checkArgument(value != null, "Cannot use null builder for a sequence value");
        values.add(value.build());
        return this;
      }

      /** Returns an immutable sequence. */
      @Override
      public Sequence build() {
        return new Sequence(values.build());
      }
    }

    /** Builder for StructureValue. */
    public static class StructureBuilder implements VariableValueBuilder {

      private final ImmutableMap.Builder<String, VariableValue> fields = ImmutableMap.builder();

      /** Adds a field to the structure. */
      public StructureBuilder addField(String name, VariableValue value) {
        fields.put(name, value);
        return this;
      }

      /** Adds a field to the structure. */
      public StructureBuilder addField(String name, VariableValueBuilder valueBuilder) {
        Preconditions.checkArgument(
            valueBuilder != null,
            "Cannot use null builder to get a field value for field '%s'",
            name);
        fields.put(name, valueBuilder.build());
        return this;
      }

      /** Adds a field to the structure. */
      public StructureBuilder addField(String name, String value) {
        fields.put(name, new StringValue(value));
        return this;
      }

      /** Adds a field to the structure. */
      public StructureBuilder addField(String name, ImmutableList<String> values) {
        fields.put(name, new StringSequence(values));
        return this;
      }

      /** Returns an immutable structure. */
      @Override
      public StructureValue build() {
        return new StructureValue(fields.build());
      }
    }

    /**
     * Lazily computed string sequence. Exists as a memory optimization. Make sure the {@param
     * supplier} doesn't capture anything that shouldn't outlive analysis phase (e.g. {@link
     * RuleContext}).
     */
    @AutoCodec
    @VisibleForSerialization
    static final class LazyStringSequence extends VariableValueAdapter {
      private final Supplier<ImmutableList<String>> supplier;

      @VisibleForSerialization
      LazyStringSequence(Supplier<ImmutableList<String>> supplier) {
        this.supplier = Preconditions.checkNotNull(supplier);
      }

      @Override
      public Iterable<? extends VariableValue> getSequenceValue(String variableName) {
        return supplier
            .get()
            .stream()
            .map(flag -> new StringValue(flag))
            .collect(ImmutableList.toImmutableList());
      }

      @Override
      public String getVariableTypeName() {
        return Sequence.SEQUENCE_VARIABLE_TYPE_NAME;
      }

      @Override
      public boolean isTruthy() {
        return !supplier.get().isEmpty();
      }
    }

    /**
     * A sequence of structure values. Exists as a memory optimization - a typical build can contain
     * millions of feature values, so getting rid of the overhead of {@code StructureValue} objects
     * significantly reduces memory overhead.
     */
    @Immutable
    @AutoCodec
    public static class LibraryToLinkValue extends VariableValueAdapter {
      public static final String OBJECT_FILES_FIELD_NAME = "object_files";
      public static final String NAME_FIELD_NAME = "name";
      public static final String TYPE_FIELD_NAME = "type";
      public static final String IS_WHOLE_ARCHIVE_FIELD_NAME = "is_whole_archive";

      private static final String LIBRARY_TO_LINK_VARIABLE_TYPE_NAME = "structure (LibraryToLink)";

      @VisibleForSerialization
      enum Type {
        OBJECT_FILE("object_file"),
        OBJECT_FILE_GROUP("object_file_group"),
        INTERFACE_LIBRARY("interface_library"),
        STATIC_LIBRARY("static_library"),
        DYNAMIC_LIBRARY("dynamic_library"),
        VERSIONED_DYNAMIC_LIBRARY("versioned_dynamic_library");

        private final String name;

        Type(String name) {
          this.name = name;
        }
      }

      private final String name;
      private final ImmutableList<Artifact> objectFiles;
      private final boolean isWholeArchive;
      private final Type type;

      public static LibraryToLinkValue forDynamicLibrary(String name) {
        return new LibraryToLinkValue(
            Preconditions.checkNotNull(name), null, false, Type.DYNAMIC_LIBRARY);
      }

      public static LibraryToLinkValue forVersionedDynamicLibrary(
          String name) {
        return new LibraryToLinkValue(
            Preconditions.checkNotNull(name), null, false, Type.VERSIONED_DYNAMIC_LIBRARY);
      }

      public static LibraryToLinkValue forInterfaceLibrary(String name) {
        return new LibraryToLinkValue(
            Preconditions.checkNotNull(name), null, false, Type.INTERFACE_LIBRARY);
      }

      public static LibraryToLinkValue forStaticLibrary(String name, boolean isWholeArchive) {
        return new LibraryToLinkValue(
            Preconditions.checkNotNull(name), null, isWholeArchive, Type.STATIC_LIBRARY);
      }

      public static LibraryToLinkValue forObjectFile(String name, boolean isWholeArchive) {
        return new LibraryToLinkValue(
            Preconditions.checkNotNull(name), null, isWholeArchive, Type.OBJECT_FILE);
      }

      public static LibraryToLinkValue forObjectFileGroup(
          ImmutableList<Artifact> objects, boolean isWholeArchive) {
        Preconditions.checkNotNull(objects);
        Preconditions.checkArgument(!objects.isEmpty());
        return new LibraryToLinkValue(null, objects, isWholeArchive, Type.OBJECT_FILE_GROUP);
      }

      @VisibleForSerialization
      LibraryToLinkValue(
          String name, ImmutableList<Artifact> objectFiles, boolean isWholeArchive, Type type) {
        this.name = name;
        this.objectFiles = objectFiles;
        this.isWholeArchive = isWholeArchive;
        this.type = type;
      }

      @Override
      public VariableValue getFieldValue(
          String variableName, String field, @Nullable ArtifactExpander expander) {
        Preconditions.checkNotNull(field);
        if (NAME_FIELD_NAME.equals(field) && !type.equals(Type.OBJECT_FILE_GROUP)) {
          return new StringValue(name);
        } else if (OBJECT_FILES_FIELD_NAME.equals(field) && type.equals(Type.OBJECT_FILE_GROUP)) {
          ImmutableList.Builder<String> expandedObjectFiles = ImmutableList.builder();
          for (Artifact objectFile : objectFiles) {
            if (objectFile.isTreeArtifact() && (expander != null)) {
              List<Artifact> artifacts = new ArrayList<>();
              expander.expand(objectFile, artifacts);
              expandedObjectFiles.addAll(
                  Iterables.transform(artifacts, artifact -> artifact.getExecPathString()));
            } else {
              expandedObjectFiles.add(objectFile.getExecPathString());
            }
          }
          return new StringSequence(expandedObjectFiles.build());
        } else if (TYPE_FIELD_NAME.equals(field)) {
          return new StringValue(type.name);
        } else if (IS_WHOLE_ARCHIVE_FIELD_NAME.equals(field)) {
          return new IntegerValue(isWholeArchive ? 1 : 0);
        } else {
          return null;
        }
      }

      @Override
      public String getVariableTypeName() {
        return LIBRARY_TO_LINK_VARIABLE_TYPE_NAME;
      }

      @Override
      public boolean isTruthy() {
        return true;
      }
    }

    /** Sequence of arbitrary VariableValue objects. */
    @Immutable
    @AutoCodec
    @VisibleForSerialization
    static final class Sequence extends VariableValueAdapter {
      private static final String SEQUENCE_VARIABLE_TYPE_NAME = "sequence";

      private final ImmutableList<VariableValue> values;

      public Sequence(ImmutableList<VariableValue> values) {
        this.values = values;
      }

      @Override
      public Iterable<? extends VariableValue> getSequenceValue(String variableName) {
        return values;
      }

      @Override
      public String getVariableTypeName() {
        return SEQUENCE_VARIABLE_TYPE_NAME;
      }

      @Override
      public boolean isTruthy() {
        return values.isEmpty();
      }
    }

    /**
     * A sequence of structure values. Exists as a memory optimization - a typical build can contain
     * millions of feature values, so getting rid of the overhead of {@code StructureValue} objects
     * significantly reduces memory overhead.
     */
    @Immutable
    @AutoCodec
    @VisibleForSerialization
    static final class StructureSequence extends VariableValueAdapter {
      private final ImmutableList<ImmutableMap<String, VariableValue>> values;

      @VisibleForSerialization
      StructureSequence(ImmutableList<ImmutableMap<String, VariableValue>> values) {
        Preconditions.checkNotNull(values);
        this.values = values;
      }

      @Override
      public Iterable<? extends VariableValue> getSequenceValue(String variableName) {
        final ImmutableList.Builder<VariableValue> sequences = ImmutableList.builder();
        for (ImmutableMap<String, VariableValue> value : values) {
          sequences.add(new StructureValue(value));
        }
        return sequences.build();
      }

      @Override
      public String getVariableTypeName() {
        return Sequence.SEQUENCE_VARIABLE_TYPE_NAME;
      }

      @Override
      public boolean isTruthy() {
        return !values.isEmpty();
      }
    }

    /**
     * A sequence of simple string values. Exists as a memory optimization - a typical build can
     * contain millions of feature values, so getting rid of the overhead of {@code StringValue}
     * objects significantly reduces memory overhead.
     */
    @Immutable
    @AutoCodec
    static final class StringSequence extends VariableValueAdapter {
      private final Iterable<String> values;

      public StringSequence(Iterable<String> values) {
        Preconditions.checkNotNull(values);
        this.values = values;
      }

      @Override
      public Iterable<? extends VariableValue> getSequenceValue(String variableName) {
        final ImmutableList.Builder<VariableValue> sequences = ImmutableList.builder();
        for (String value : values) {
          sequences.add(new StringValue(value));
        }
        return sequences.build();
      }

      @Override
      public String getVariableTypeName() {
        return Sequence.SEQUENCE_VARIABLE_TYPE_NAME;
      }

      @Override
      public boolean isTruthy() {
        return !Iterables.isEmpty(values);
      }
    }

    /**
     * Single structure value. Be careful not to create sequences of single structures, as the
     * memory overhead is prohibitively big. Use optimized {@link StructureSequence} instead.
     */
    @Immutable
    @AutoCodec
    @VisibleForSerialization
    static final class StructureValue extends VariableValueAdapter {
      private static final String STRUCTURE_VARIABLE_TYPE_NAME = "structure";

      private final ImmutableMap<String, VariableValue> value;

      public StructureValue(ImmutableMap<String, VariableValue> value) {
        this.value = value;
      }

      @Override
      public VariableValue getFieldValue(
          String variableName, String field, @Nullable ArtifactExpander expander) {
        if (value.containsKey(field)) {
          return value.get(field);
        } else {
          return null;
        }
      }

      @Override
      public String getVariableTypeName() {
        return STRUCTURE_VARIABLE_TYPE_NAME;
      }

      @Override
      public boolean isTruthy() {
        return !value.isEmpty();
      }
    }

    /**
     * The leaves in the variable sequence node tree are simple string values. Note that this should
     * never live outside of {@code expand}, as the object overhead is prohibitively expensive.
     */
    @Immutable
    @AutoCodec
    @VisibleForSerialization
    static final class StringValue extends VariableValueAdapter {
      private static final String STRING_VARIABLE_TYPE_NAME = "string";

      private final String value;

      public StringValue(String value) {
        Preconditions.checkNotNull(value, "Cannot create StringValue from null");
        this.value = value;
      }

      @Override
      public String getStringValue(String variableName) {
        return value;
      }

      @Override
      public String getVariableTypeName() {
        return STRING_VARIABLE_TYPE_NAME;
      }

      @Override
      public boolean isTruthy() {
        return !value.isEmpty();
      }
    }

    /**
     * The leaves in the variable sequence node tree are simple integer values. Note that this
     * should never live outside of {@code expand}, as the object overhead is prohibitively
     * expensive.
     */
    @Immutable
    @AutoCodec
    static final class IntegerValue extends VariableValueAdapter {
      private static final String INTEGER_VALUE_TYPE_NAME = "integer";
      private final int value;

      public IntegerValue(int value) {
        this.value = value;
      }

      @Override
      public String getStringValue(String variableName) {
        return Integer.toString(value);
      }

      @Override
      public String getVariableTypeName() {
        return INTEGER_VALUE_TYPE_NAME;
      }

      @Override
      public boolean isTruthy() {
        return value != 0;
      }
    }

    /**
     * Builder for {@code Variables}.
     */
    // TODO(b/65472725): Forbid sequences with empty string in them.
    public static class Builder {
      private final Map<String, VariableValue> variablesMap = new LinkedHashMap<>();
      private final Map<String, String> stringVariablesMap = new LinkedHashMap<>();
      private final Variables parent;

      public Builder() {
        parent = null;
      }

      public Builder(@Nullable Variables parent) {
        this.parent = parent;
      }

      /** Add an integer variable that expands {@code name} to {@code value}. */
      public Builder addIntegerVariable(String name, int value) {
        variablesMap.put(name, new IntegerValue(value));
        return this;
      }

      /** Add a string variable that expands {@code name} to {@code value}. */
      public Builder addStringVariable(String name, String value) {
        checkVariableNotPresentAlready(name);
        Preconditions.checkNotNull(
            value, "Cannot set null as a value for variable '%s'", name);
        stringVariablesMap.put(name, value);
        return this;
      }

      /** Overrides a variable to expands {@code name} to {@code value} instead. */
      public Builder overrideStringVariable(String name, String value) {
        Preconditions.checkNotNull(
            value, "Cannot set null as a value for variable '%s'", name);
        stringVariablesMap.put(name, value);
        return this;
      }

      /** Overrides a variable to expands {@code name} to {@code value} instead. */
      public Builder overrideLazyStringSequenceVariable(
          String name, Supplier<ImmutableList<String>> supplier) {
        Preconditions.checkNotNull(supplier, "Cannot set null as a value for variable '%s'", name);
        variablesMap.put(name, new LazyStringSequence(supplier));
        return this;
      }

      /**
       * Add a sequence variable that expands {@code name} to {@code values}.
       *
       * <p>Accepts values as ImmutableSet. As ImmutableList has smaller memory footprint, we copy
       * the values into a new list.
       */
      public Builder addStringSequenceVariable(String name, ImmutableSet<String> values) {
        checkVariableNotPresentAlready(name);
        Preconditions.checkNotNull(values, "Cannot set null as a value for variable '%s'", name);
        ImmutableList.Builder<String> builder = ImmutableList.builder();
        builder.addAll(values);
        variablesMap.put(name, new StringSequence(builder.build()));
        return this;
      }

      /**
       * Add a sequence variable that expands {@code name} to {@code values}.
       *
       * <p>Accepts values as NestedSet. Nested set is stored directly, not cloned, not flattened.
       */
      public Builder addStringSequenceVariable(String name, NestedSet<String> values) {
        checkVariableNotPresentAlready(name);
        Preconditions.checkNotNull(values, "Cannot set null as a value for variable '%s'", name);
        variablesMap.put(name, new StringSequence(values));
        return this;
      }

      /**
       * Add a sequence variable that expands {@code name} to {@code values}.
       *
       * <p>Accepts values as Iterable. The iterable is stored directly, not cloned, not iterated.
       * Be mindful of memory consumption of the particular Iterable. Prefer ImmutableList, or
       * be sure that the iterable always returns the same elements in the same order, without any
       * side effects.
       */
      public Builder addStringSequenceVariable(String name, Iterable<String> values) {
        checkVariableNotPresentAlready(name);
        Preconditions.checkNotNull(values, "Cannot set null as a value for variable '%s'", name);
        variablesMap.put(name, new StringSequence(values));
        return this;
      }

      public Builder addLazyStringSequenceVariable(
          String name, Supplier<ImmutableList<String>> supplier) {
        checkVariableNotPresentAlready(name);
        Preconditions.checkNotNull(supplier, "Cannot set null as a value for variable '%s'", name);
        variablesMap.put(name, new LazyStringSequence(supplier));
        return this;
      }

      /**
       * Add a variable built using {@code VariableValueBuilder} api that expands {@code name} to
       * the value returned by the {@code builder}.
       */
      public Builder addCustomBuiltVariable(String name, Variables.VariableValueBuilder builder) {
        checkVariableNotPresentAlready(name);
        Preconditions.checkNotNull(
            builder,
            "Cannot use null builder to get variable value for variable '%s'",
            name);
        variablesMap.put(name, builder.build());
        return this;
      }

      /** Add all string variables in a map. */
      public Builder addAllStringVariables(Map<String, String> variables) {
        for (String name : variables.keySet()) {
          checkVariableNotPresentAlready(name);
        }
        stringVariablesMap.putAll(variables);
        return this;
      }

      private void checkVariableNotPresentAlready(String name) {
        Preconditions.checkNotNull(name);
        Preconditions.checkArgument(
            !variablesMap.containsKey(name), "Cannot overwrite variable '%s'", name);
        Preconditions.checkArgument(
            !stringVariablesMap.containsKey(name), "Cannot overwrite variable '%s'", name);
      }

      /**
       * Adds all variables to this builder. Cannot override already added variables. Does not add
       * variables defined in the {@code parent} variables.
       */
      public Builder addAllNonTransitive(Variables variables) {
        SetView<String> intersection =
            Sets.intersection(variables.getVariablesMap().keySet(), variablesMap.keySet());
        SetView<String> stringIntersection =
            Sets.intersection(
                variables.getStringVariablesMap().keySet(), stringVariablesMap.keySet());
        Preconditions.checkArgument(
            intersection.isEmpty(), "Cannot overwrite existing variables: %s", intersection);
        Preconditions.checkArgument(
            stringIntersection.isEmpty(),
            "Cannot overwrite existing variables: %s", stringIntersection);
        this.variablesMap.putAll(variables.getVariablesMap());
        this.stringVariablesMap.putAll(variables.getStringVariablesMap());
        return this;
      }

      /** @return a new {@link Variables} object. */
      public Variables build() {
        if (stringVariablesMap.isEmpty() && variablesMap.size() == 1) {
          return new SingleVariables(
              parent,
              variablesMap.keySet().iterator().next(),
              variablesMap.values().iterator().next());
        }
        return new MapVariables(
            parent, ImmutableMap.copyOf(variablesMap), ImmutableMap.copyOf(stringVariablesMap));
      }
    }
    
    /**
     * A group of extra {@code Variable} instances, packaged as logic for adding to a
     * {@code Builder}
     */
    public interface VariablesExtension {
      void addVariables(Builder builder);
    }
  }

  @Immutable
  @AutoCodec.VisibleForSerialization
  @AutoCodec
  static class MapVariables extends Variables {
    private final ImmutableMap<String, VariableValue> variablesMap;
    private final ImmutableMap<String, String> stringVariablesMap;
    private final Variables parent;

    @AutoCodec.Instantiator
    @VisibleForSerialization
    MapVariables(
        Variables parent,
        ImmutableMap<String, VariableValue> variablesMap,
        ImmutableMap<String, String> stringVariablesMap) {
      this.variablesMap = variablesMap;
      this.stringVariablesMap = stringVariablesMap;
      this.parent = parent;
    }

    @Override
    Map<String, VariableValue> getVariablesMap() {
      return variablesMap;
    }

    @Override
    Map<String, String> getStringVariablesMap() {
      return stringVariablesMap;
    }

    @Override
    VariableValue getNonStructuredVariable(String name) {
      if (variablesMap.containsKey(name)) {
        return variablesMap.get(name);
      }
      if (stringVariablesMap.containsKey(name)) {
        return new StringValue(stringVariablesMap.get(name));
      }

      if (parent != null) {
        return parent.getNonStructuredVariable(name);
      }

      return null;
    }
  }

  @Immutable
  private static class SingleVariables extends Variables {
    private final Variables parent;
    private final String name;
    private final VariableValue variableValue;

    SingleVariables(Variables parent, String name, VariableValue variableValue) {
      this.parent = parent;
      this.name = name;
      this.variableValue = variableValue;
    }

    @Override
    Map<String, VariableValue> getVariablesMap() {
      return ImmutableMap.of(name, variableValue);
    }

    @Override
    Map<String, String> getStringVariablesMap() {
      return ImmutableMap.of();
    }

    @Override
    VariableValue getNonStructuredVariable(String name) {
      if (this.name.equals(name)) {
        return variableValue;
      }
      return parent == null ? null : parent.getNonStructuredVariable(name);
    }
  }

  /** Captures the set of enabled features and action configs for a rule. */
  @Immutable
  @AutoCodec
  @SkylarkModule(
    name = "feature_configuration",
    documented = false,
    category = SkylarkModuleCategory.BUILTIN,
    doc = "Class used to construct command lines from CROSSTOOL features."
  )
  public static class FeatureConfiguration {
    private final ImmutableSet<String> enabledFeatureNames;
    private final ImmutableList<Feature> enabledFeatures;
    private final ImmutableSet<String> enabledActionConfigActionNames;

    private final ImmutableMap<String, ActionConfig> actionConfigByActionName;

    /**
     * {@link FeatureConfiguration} instance that doesn't produce any command lines. This is to be
     * used when creation of the real {@link FeatureConfiguration} failed, the rule error was
     * reported, but the analysis continues to collect more rule errors.
     */
    public static final FeatureConfiguration EMPTY = new FeatureConfiguration();

    protected FeatureConfiguration() {
      this(ImmutableList.of(), ImmutableSet.of(), ImmutableMap.of());
    }

    @AutoCodec.Instantiator
    FeatureConfiguration(
        ImmutableList<Feature> enabledFeatures,
        ImmutableSet<String> enabledActionConfigActionNames,
        ImmutableMap<String, ActionConfig> actionConfigByActionName) {
      this.enabledFeatures = enabledFeatures;

      this.actionConfigByActionName = actionConfigByActionName;
      ImmutableSet.Builder<String> featureBuilder = ImmutableSet.builder();
      for (Feature feature : enabledFeatures) {
        featureBuilder.add(feature.getName());
      }
      this.enabledFeatureNames = featureBuilder.build();
      this.enabledActionConfigActionNames = enabledActionConfigActionNames;
    }

    /**
     * @return whether the given {@code feature} is enabled.
     */
    public boolean isEnabled(String feature) {
      return enabledFeatureNames.contains(feature);
    }

    /** @return true if tool_path in action_config points to a real tool, not a dummy placeholder */
    public boolean hasConfiguredLinkerPathInActionConfig() {
      return isEnabled("has_configured_linker_path");
    }

    /** @return whether an action config for the blaze action with the given name is enabled. */
    boolean actionIsConfigured(String actionName) {
      return enabledActionConfigActionNames.contains(actionName);
    }

    /** @return the command line for the given {@code action}. */
    public List<String> getCommandLine(String action, Variables variables) {
      return getCommandLine(action, variables, null);
    }

    public List<String> getCommandLine(
        String action, Variables variables, @Nullable ArtifactExpander expander) {
      List<String> commandLine = new ArrayList<>();
      if (actionIsConfigured(action)) {
        actionConfigByActionName
            .get(action)
            .expandCommandLine(variables, enabledFeatureNames, expander, commandLine);
      }

      for (Feature feature : enabledFeatures) {
        feature.expandCommandLine(action, variables, enabledFeatureNames, expander, commandLine);
      }

      return commandLine;
    }

    /** @return the flags expanded for the given {@code action} in per-feature buckets. */
    public ImmutableList<Pair<String, List<String>>> getPerFeatureExpansions(
        String action, Variables variables) {
      return getPerFeatureExpansions(action, variables, null);
    }

    public ImmutableList<Pair<String, List<String>>> getPerFeatureExpansions(
        String action, Variables variables, @Nullable ArtifactExpander expander) {
      ImmutableList.Builder<Pair<String, List<String>>> perFeatureExpansions =
          ImmutableList.builder();
      if (actionIsConfigured(action)) {
        List<String> commandLine = new ArrayList<>();
        ActionConfig actionConfig = actionConfigByActionName.get(action);
        actionConfig.expandCommandLine(variables, enabledFeatureNames, expander, commandLine);
        perFeatureExpansions.add(Pair.of(actionConfig.getName(), commandLine));
      }

      for (Feature feature : enabledFeatures) {
        List<String> commandLine = new ArrayList<>();
        feature.expandCommandLine(action, variables, enabledFeatureNames, expander, commandLine);
        perFeatureExpansions.add(Pair.of(feature.getName(), commandLine));
      }

      return perFeatureExpansions.build();
    }

    /** @return the environment variables (key/value pairs) for the given {@code action}. */
    public ImmutableMap<String, String> getEnvironmentVariables(
        String action, Variables variables) {
      ImmutableMap.Builder<String, String> envBuilder = ImmutableMap.builder();
      for (Feature feature : enabledFeatures) {
        feature.expandEnvironment(action, variables, enabledFeatureNames, envBuilder);
      }
      return envBuilder.build();
    }

    /** Returns a given action's tool under this FeatureConfiguration. */
    public Tool getToolForAction(String actionName) {
      Preconditions.checkArgument(
          actionConfigByActionName.containsKey(actionName),
          "Action %s does not have an enabled configuration in the toolchain.",
          actionName);
      ActionConfig actionConfig = actionConfigByActionName.get(actionName);
      return actionConfig.getTool(enabledFeatureNames);
    }

    @Override
    public boolean equals(Object object) {
      if (object == this) {
        return true;
      }
      if (object instanceof FeatureConfiguration) {
        FeatureConfiguration that = (FeatureConfiguration) object;
        return Objects.equals(actionConfigByActionName, that.actionConfigByActionName)
            && Iterables.elementsEqual(
                enabledActionConfigActionNames, that.enabledActionConfigActionNames)
            && Iterables.elementsEqual(enabledFeatureNames, that.enabledFeatureNames)
            && Iterables.elementsEqual(enabledFeatures, that.enabledFeatures);
      }
      return false;
    }

    @Override
    public int hashCode() {
      return Objects.hash(
          actionConfigByActionName,
          enabledActionConfigActionNames,
          enabledFeatureNames,
          enabledFeatures);
    }

    public ImmutableSet<String> getEnabledFeatureNames() {
      return enabledFeatureNames;
    }
  }

  /** All artifact name patterns defined in this feature configuration. */
  private final ImmutableList<ArtifactNamePattern> artifactNamePatterns;

  /**
   * All features and action configs in the order in which they were specified in the configuration.
   *
   * <p>We guarantee the command line to be in the order in which the flags were specified in the
   * configuration.
   */
  private final ImmutableList<CrosstoolSelectable> selectables;

  /**
   * Maps the selectables's name to the selectable.
   */
  private final ImmutableMap<String, CrosstoolSelectable> selectablesByName;

  /**
   * Maps an action's name to the ActionConfig.
   */
  private final ImmutableMap<String, ActionConfig> actionConfigsByActionName;

  /**
   * Maps from a selectable to a set of all the selectables it has a direct 'implies' edge to.
   */
  private final ImmutableMultimap<CrosstoolSelectable, CrosstoolSelectable> implies;

  /**
   * Maps from a selectable to all features that have an direct 'implies' edge to this
   * selectable.
   */
  private final ImmutableMultimap<CrosstoolSelectable, CrosstoolSelectable> impliedBy;

  /**
   * Maps from a selectable to a set of selecatable sets, where:
   * <ul>
   * <li>a selectable set satisfies the 'requires' condition, if all selectables in the
   *        selectable set are enabled</li>
   * <li>the 'requires' condition is satisfied, if at least one of the selectable sets satisfies
   *        the 'requires' condition.</li>
   * </ul>
   */
  private final ImmutableMultimap<CrosstoolSelectable, ImmutableSet<CrosstoolSelectable>>
      requires;

  /**
   * Maps from a string to the set of selectables that 'provide' it.
   */
  private final ImmutableMultimap<String, CrosstoolSelectable> provides;

  /**
   * Maps from a selectable to all selectables that have a requirement referencing it.
   *
   * <p>This will be used to determine which selectables need to be re-checked after a selectable
   * was disabled.
   */
  private final ImmutableMultimap<CrosstoolSelectable, CrosstoolSelectable> requiredBy;

  private final ImmutableList<String> defaultSelectables;

  /**
   * A cache of feature selection results, so we do not recalculate the feature selection for all
   * actions.
   */
  private transient LoadingCache<ImmutableSet<String>, FeatureConfiguration> configurationCache =
      buildConfigurationCache();

  /**
   * Constructs the feature configuration from a {@code CToolchain} protocol buffer.
   *
   * @param toolchain the toolchain configuration as specified by the user.
   * @throws InvalidConfigurationException if the configuration has logical errors.
   */
  @VisibleForTesting
  public CcToolchainFeatures(CToolchain toolchain, PathFragment crosstoolTop)
      throws InvalidConfigurationException {
    // Build up the feature/action config graph.  We refer to features/action configs as
    // 'selectables'.
    // First, we build up the map of name -> selectables in one pass, so that earlier selectables
    // can reference later features in their configuration.
    ImmutableList.Builder<CrosstoolSelectable> selectablesBuilder = ImmutableList.builder();
    HashMap<String, CrosstoolSelectable> selectablesByName = new HashMap<>();

    // Also build a map from action -> action_config, for use in tool lookups
    ImmutableMap.Builder<String, ActionConfig> actionConfigsByActionName = ImmutableMap.builder();

    ImmutableList.Builder<String> defaultSelectablesBuilder = ImmutableList.builder();
    for (CToolchain.Feature toolchainFeature : toolchain.getFeatureList()) {
      Feature feature = new Feature(toolchainFeature);
      selectablesBuilder.add(feature);
      selectablesByName.put(feature.getName(), feature);
      if (toolchainFeature.getEnabled()) {
        defaultSelectablesBuilder.add(feature.getName());
      }
    }

    for (CToolchain.ActionConfig toolchainActionConfig : toolchain.getActionConfigList()) {
      ActionConfig actionConfig = new ActionConfig(toolchainActionConfig, crosstoolTop);
      selectablesBuilder.add(actionConfig);
      selectablesByName.put(actionConfig.getName(), actionConfig);
      actionConfigsByActionName.put(actionConfig.getActionName(), actionConfig);
      if (toolchainActionConfig.getEnabled()) {
        defaultSelectablesBuilder.add(actionConfig.getName());
      }
    }
    this.defaultSelectables = defaultSelectablesBuilder.build();
       
    this.selectables = selectablesBuilder.build();
    this.selectablesByName = ImmutableMap.copyOf(selectablesByName);

    checkForActionNameDups(toolchain.getActionConfigList());
    checkForActivatableDups(this.selectables);

    this.actionConfigsByActionName = actionConfigsByActionName.build();

    ImmutableList.Builder<ArtifactNamePattern> artifactNamePatternsBuilder =
        ImmutableList.builder();
    for (CToolchain.ArtifactNamePattern artifactNamePattern :
        toolchain.getArtifactNamePatternList()) {
      artifactNamePatternsBuilder.add(new ArtifactNamePattern(artifactNamePattern));
    }
    this.artifactNamePatterns = artifactNamePatternsBuilder.build();

    // Next, we build up all forward references for 'implies', 'requires', and 'provides' edges.
    ImmutableMultimap.Builder<CrosstoolSelectable, CrosstoolSelectable> implies =
        ImmutableMultimap.builder();
    ImmutableMultimap.Builder<CrosstoolSelectable, ImmutableSet<CrosstoolSelectable>> requires =
        ImmutableMultimap.builder();
    ImmutableMultimap.Builder<CrosstoolSelectable, String> provides = ImmutableMultimap.builder();
    // We also store the reverse 'implied by' and 'required by' edges during this pass.
    ImmutableMultimap.Builder<CrosstoolSelectable, CrosstoolSelectable> impliedBy =
        ImmutableMultimap.builder();
    ImmutableMultimap.Builder<CrosstoolSelectable, CrosstoolSelectable> requiredBy =
        ImmutableMultimap.builder();

    for (CToolchain.Feature toolchainFeature : toolchain.getFeatureList()) {
      String name = toolchainFeature.getName();
      CrosstoolSelectable selectable = selectablesByName.get(name);
      for (CToolchain.FeatureSet requiredFeatures : toolchainFeature.getRequiresList()) {
        ImmutableSet.Builder<CrosstoolSelectable> allOf = ImmutableSet.builder();
        for (String requiredName : requiredFeatures.getFeatureList()) {
          CrosstoolSelectable required = getActivatableOrFail(requiredName, name);
          allOf.add(required);
          requiredBy.put(required, selectable);
        }
        requires.put(selectable, allOf.build());
      }
      for (String impliedName : toolchainFeature.getImpliesList()) {
        CrosstoolSelectable implied = getActivatableOrFail(impliedName, name);
        impliedBy.put(implied, selectable);
        implies.put(selectable, implied);
      }
      for (String providesName : toolchainFeature.getProvidesList()) {
        provides.put(selectable, providesName);
      }
    }

    for (CToolchain.ActionConfig toolchainActionConfig : toolchain.getActionConfigList()) {
      String name = toolchainActionConfig.getConfigName();
      CrosstoolSelectable selectable = selectablesByName.get(name);
      for (String impliedName : toolchainActionConfig.getImpliesList()) {
        CrosstoolSelectable implied = getActivatableOrFail(impliedName, name);
        impliedBy.put(implied, selectable);
        implies.put(selectable, implied);
      }
    }

    this.implies = implies.build();
    this.requires = requires.build();
    this.provides = provides.build().inverse();
    this.impliedBy = impliedBy.build();
    this.requiredBy = requiredBy.build();
  }

  private static void checkForActivatableDups(Iterable<CrosstoolSelectable> selectables)
      throws InvalidConfigurationException {
    Collection<String> names = new HashSet<>();
    for (CrosstoolSelectable selectable : selectables) {
      if (!names.add(selectable.getName())) {
        throw new InvalidConfigurationException(
            "Invalid toolchain configuration: feature or "
                + "action config '"
                + selectable.getName()
                + "' was specified multiple times.");
      }
    }
  }

  private static void checkForActionNameDups(Iterable<CToolchain.ActionConfig> actionConfigs)
      throws InvalidConfigurationException {
    Collection<String> actionNames = new HashSet<>();
    for (CToolchain.ActionConfig actionConfig : actionConfigs) {
      if (!actionNames.add(actionConfig.getActionName())) {
        throw new InvalidConfigurationException(
            "Invalid toolchain configuration: multiple action "
                + "configs for action '"
                + actionConfig.getActionName()
                + "'");
      }
    }
  }

  /**
   * Assign an empty cache after default-deserializing all non-transient members.
   */
  private void readObject(ObjectInputStream in) throws ClassNotFoundException, IOException {
    in.defaultReadObject();
    this.configurationCache = buildConfigurationCache();
  }

  /** @return an empty {@code FeatureConfiguration} cache. */
  private LoadingCache<ImmutableSet<String>, FeatureConfiguration> buildConfigurationCache() {
    return CacheBuilder.newBuilder()
        // TODO(klimek): Benchmark and tweak once we support a larger configuration.
        .maximumSize(10000)
        .build(
            new CacheLoader<ImmutableSet<String>, FeatureConfiguration>() {
              @Override
              public FeatureConfiguration load(ImmutableSet<String> requestedFeatures)
                  throws CollidingProvidesException {
                return computeFeatureConfiguration(requestedFeatures);
              }
            });
  }

  /**
   * Given a list of {@code requestedSelectables}, returns all features that are enabled by the
   * toolchain configuration.
   *
   * <p>A requested feature will not be enabled if the toolchain does not support it (which may
   * depend on other requested features).
   *
   * <p>Additional features will be enabled if the toolchain supports them and they are implied by
   * requested features.
   */
  public FeatureConfiguration getFeatureConfiguration(ImmutableSet<String> requestedSelectables)
      throws CollidingProvidesException {
    try {
      return configurationCache.get(requestedSelectables);
    } catch (ExecutionException e) {
      Throwables.throwIfInstanceOf(e.getCause(), CollidingProvidesException.class);
      Throwables.throwIfUnchecked(e.getCause());
      throw new IllegalStateException("Unexpected checked exception encountered", e);
    }
  }

  /**
   * Given {@code featureSpecification}, returns a FeatureConfiguration with all requested features
   * enabled.
   *
   * <p>A requested feature will not be enabled if the toolchain does not support it (which may
   * depend on other requested features).
   *
   * <p>Additional features will be enabled if the toolchain supports them and they are implied by
   * requested features.
   */
  public FeatureConfiguration computeFeatureConfiguration(ImmutableSet<String> requestedSelectables)
      throws CollidingProvidesException {
    // Command line flags will be output in the order in which they are specified in the toolchain
    // configuration.
    return new FeatureSelection(
            requestedSelectables,
            selectablesByName,
            selectables,
            provides,
            implies,
            impliedBy,
            requires,
            requiredBy,
            actionConfigsByActionName)
        .run();
  }

  public ImmutableList<String> getDefaultFeaturesAndActionConfigs() {
    return defaultSelectables;
  }

  /**
   * @return the selectable with the given {@code name}.s
   *
   * @throws InvalidConfigurationException if no selectable with the given name was configured.
   */
  private CrosstoolSelectable getActivatableOrFail(String name, String reference)
      throws InvalidConfigurationException {
    if (!selectablesByName.containsKey(name)) {
      throw new InvalidConfigurationException("Invalid toolchain configuration: feature '" + name
          + "', which is referenced from feature '" + reference + "', is not defined.");
    }
    return selectablesByName.get(name);
  }
  
  @VisibleForTesting
  Collection<String> getActivatableNames() {
    Collection<String> featureNames = new HashSet<>();
    for (CrosstoolSelectable selectable : selectables) {
      featureNames.add(selectable.getName());
    }
    return featureNames;
  }

  /**
   * Returns the artifact selected by the toolchain for the given action type and action category.
   *
   * @throws InvalidConfigurationException if the category is not supported by the action config.
   */
  String getArtifactNameForCategory(ArtifactCategory artifactCategory, String outputName)
      throws InvalidConfigurationException {
    PathFragment output = PathFragment.create(outputName);

    ArtifactNamePattern patternForCategory = null;
    for (ArtifactNamePattern artifactNamePattern : artifactNamePatterns) {
      if (artifactNamePattern.getArtifactCategory() == artifactCategory) {
        patternForCategory = artifactNamePattern;
      }
    }
    if (patternForCategory == null) {
      throw new InvalidConfigurationException(
          String.format(
              MISSING_ARTIFACT_NAME_PATTERN_ERROR_TEMPLATE, artifactCategory.getCategoryName()));
    }

    return patternForCategory.getArtifactName(ImmutableMap.of(
        "output_name", outputName,
        "base_name", output.getBaseName(),
        "output_directory", output.getParentDirectory().getPathString()));
  }

  /** Returns true if the toolchain defines an ArtifactNamePattern for the given category. */
  boolean hasPatternForArtifactCategory(ArtifactCategory artifactCategory) {
    for (ArtifactNamePattern artifactNamePattern : artifactNamePatterns) {
      if (artifactNamePattern.getArtifactCategory() == artifactCategory) {
        return true;
      }
    }
    return false;
  }
}
