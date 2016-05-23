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
import com.google.common.base.Optional;
import com.google.common.base.Preconditions;
import com.google.common.base.Predicate;
import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableMultimap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CToolchain;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CToolchain.Tool;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.Serializable;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Set;
import java.util.Stack;

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
 * <p>TODO(bazel-team): Split out Feature as CcToolchainFeature, which will modularize the
 * crosstool configuration into one part that is about handling a set of features (including feature
 * selection) and one part that is about how to apply a single feature (parsing flags and expanding
 * them from build variables).
 */
@Immutable
public class CcToolchainFeatures implements Serializable {
  
  /**
   * Thrown when a flag value cannot be expanded under a set of build variables.
   * 
   * <p>This happens for example when a flag references a variable that is not provided by the
   * action, or when a flag group references multiple variables of sequence type.
   */
  public static class ExpansionException extends RuntimeException {
    ExpansionException(String message) {
      super(message);
    }
  }
  
  /**
   * A piece of a single string value.
   * 
   * <p>A single value can contain a combination of text and variables (for example
   * "-f %{var1}/%{var2}"). We split the string into chunks, where each chunk represents either a
   * text snippet, or a variable that is to be replaced.
   */
  interface StringChunk {
    
    /**
     * Expands this chunk.
     * 
     * @param variables variable names mapped to their values for a single flag expansion.
     * @param flag the flag content to append to.
     */
    void expand(Map<String, String> variables, StringBuilder flag);
  }
  
  /**
   * A plain text chunk of a string (containing no variables).
   */
  @Immutable
  private static class StringLiteralChunk implements StringChunk, Serializable {
    private final String text;
    
    private StringLiteralChunk(String text) {
      this.text = text;
    }
    
    @Override
    public void expand(Map<String, String> variables, StringBuilder flag) {
      flag.append(text);
    }
  }
  
  /**
   * A chunk of a string value into which a variable should be expanded.
   */
  @Immutable
  private static class VariableChunk implements StringChunk, Serializable {
    private final String variableName;
    
    private VariableChunk(String variableName) {
      this.variableName = variableName;
    }
    
    @Override
    public void expand(Map<String, String> variables, StringBuilder stringBuilder) {
      // We check all variables in FlagGroup.expandCommandLine.
      // If we arrive here with the variable not being available, the variable was provided, but
      // the nesting level of the NestedSequence was deeper than the nesting level of the flag
      // groups.
      if (!variables.containsKey(variableName)) {
        throw new ExpansionException(
            "Invalid toolchain configuration: the flag group referencing '"
                + variableName
                + "' is not nested deeply enough to fully expand it.");
      }
      String value = variables.get(variableName);
      if (value == null) {
        throw new ExpansionException(
            "Internal blaze error: build variable '" + variableName + "'was set to 'null'.");
      }
      stringBuilder.append(variables.get(variableName));
    }
  }
  
  /**
   * Parser for toolchain string values.
   * 
   * <p>A string value contains a snippet of text supporting variable expansion. For example, a
   * string value "-f %{var1}/%{var2}" will expand the values of the variables "var1" and "var2"
   * in the corresponding places in the string.
   * 
   * <p>The {@code StringValueParser} takes a string and parses it into a list of
   * {@link StringChunk} objects, where each chunk represents either a snippet of text or a
   * variable to be expanded. In the above example, the resulting chunks would be
   * ["-f ", var1, "/", var2].
   * 
   * <p>In addition to the list of chunks, the {@link StringValueParser} also provides the set of
   * variables necessary for the expansion of this flag via {@link #getUsedVariables}.
   * 
   * <p>To get a literal percent character, "%%" can be used in the string.
   */
  private static class StringValueParser {

    private final String value;
    
    /**
     * The current position in {@value} during parsing.
     */
    private int current = 0;
    
    private final ImmutableList.Builder<StringChunk> chunks = ImmutableList.builder();
    private final ImmutableSet.Builder<String> usedVariables = ImmutableSet.builder();
    
    private StringValueParser(String value) throws InvalidConfigurationException {
      this.value = value;
      parse();
    }
    
    /**
     * @return the parsed chunks for this string.
     */
    private ImmutableList<StringChunk> getChunks() {
      return chunks.build();
    }
    
    /**
     * @return all variable names needed to expand this string.
     */
    private ImmutableSet<String> getUsedVariables() {
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
  
  /**
   * A flag or flag group that can be expanded under a set of variables.
   */
  interface Expandable {

    /**
     * Returns the set of variables that can control the expansion of this expandable.
     * Returns null if the expandable is a simple flag that will only be expanded once.
     */
    Set<String> getControllingVariables();

    /**
     * Expands the current expandable under the given {@code view}, adding new flags to
     * {@code commandLine}.
     *
     * The {@code view} controls which variables are visible during the expansion and allows
     * to recursively expand nested flag groups.
     */
    void expand(Variables.View view, List<String> commandLine);
  }

  /**
   * A single flag to be expanded under a set of variables.
   *
   * <p>TODO(bazel-team): Consider specializing Flag for the simple case that a flag is just a bit
   * of text.
   */
  @Immutable
  private static class Flag implements Serializable, Expandable {
    private final ImmutableList<StringChunk> chunks;
    
    private Flag(ImmutableList<StringChunk> chunks) {
      this.chunks = chunks;
    }
    
    /**
     * Expand this flag into a single new entry in {@code commandLine}.
     */
    @Override
    public void expand(Variables.View view, List<String> commandLine) {
      StringBuilder flag = new StringBuilder();
      for (StringChunk chunk : chunks) {
        chunk.expand(view.getVariables(), flag);
      }
      commandLine.add(flag.toString());
    }

    @Override
    public Set<String> getControllingVariables() {
      // A simple flag will only ever be expanded once.
      return null;
    }
  }
  
  /**
   * A single environment key/value pair to be expanded under a set of variables.
   */
  @Immutable
  private static class EnvEntry implements Serializable {
    private final String key;
    private final ImmutableList<StringChunk> valueChunks;
    private final ImmutableSet<String> usedVariables;
    
    private EnvEntry(CToolchain.EnvEntry envEntry) throws InvalidConfigurationException {
      this.key = envEntry.getKey();
      StringValueParser parser = new StringValueParser(envEntry.getValue());
      this.valueChunks = parser.getChunks();
      this.usedVariables = parser.getUsedVariables();
    }

    /**
     * Adds the key/value pair this object represents to the given map of environment variables.
     * The value of the entry is expanded with the given {@code variables}.
     */
    public void addEnvEntry(Variables variables, ImmutableMap.Builder<String, String> envBuilder) {
      Variables.View view = variables.getView(usedVariables);
      StringBuilder value = new StringBuilder();
      for (StringChunk chunk : valueChunks) {
        chunk.expand(view.getVariables(), value);
      }
      envBuilder.put(key, value.toString());
    }
  }

  /**
   * A group of flags.
   */
  @Immutable
  private static class FlagGroup implements Serializable, Expandable {
    private final ImmutableList<Expandable> expandables;
    private final ImmutableSet<String> usedVariables;
    
    private FlagGroup(CToolchain.FlagGroup flagGroup) throws InvalidConfigurationException {
      ImmutableList.Builder<Expandable> expandables = ImmutableList.builder();
      ImmutableSet.Builder<String> usedVariables = ImmutableSet.builder();
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
        expandables.add(new Flag(parser.getChunks()));
        usedVariables.addAll(parser.getUsedVariables());
      }
      for (CToolchain.FlagGroup group : groups) {
        FlagGroup subgroup = new FlagGroup(group);
        expandables.add(subgroup);
        usedVariables.addAll(subgroup.getControllingVariables());
      }
      this.expandables = expandables.build();
      this.usedVariables = usedVariables.build();
    }
    
    @Override
    public void expand(Variables.View view, final List<String> commandLine) {
      for (Expandable expandable : expandables) {
        view.expand(expandable, commandLine);
      }
    }

    @Override
    public Set<String> getControllingVariables() {
      // Any of the used variables can be used to control this flag group's expansion.
      return usedVariables;
    }

    /**
     * Expands all flags in this group and adds them to {@code commandLine}.
     * 
     * <p>The flags of the group will be expanded either:
     * <ul>
     * <li>once, if there is no variable of sequence type in any of the group's flags, or</li>
     * <li>for each element in the sequence, if there is one variable of sequence type within
     * the flags.</li>
     * </ul>
     * 
     * <p>Having more than a single variable of sequence type in a single flag group is not
     * supported.
     */
    private void expandCommandLine(Variables variables, final List<String> commandLine) {
      Variables.View view = variables.getView(getControllingVariables());
      view.expand(this, commandLine);
    }
  }
  
  /**
   * Groups a set of flags to apply for certain actions.
   */
  @Immutable
  private static class FlagSet implements Serializable {
    private final ImmutableSet<String> actions;
    private final ImmutableSet<String> expandIfAllAvailable;
    private final ImmutableList<FlagGroup> flagGroups;
    
    private FlagSet(CToolchain.FlagSet flagSet) throws InvalidConfigurationException {
      this.actions = ImmutableSet.copyOf(flagSet.getActionList());
      this.expandIfAllAvailable = ImmutableSet.copyOf(flagSet.getExpandIfAllAvailableList());
      ImmutableList.Builder<FlagGroup> builder = ImmutableList.builder();
      for (CToolchain.FlagGroup flagGroup : flagSet.getFlagGroupList()) {
        builder.add(new FlagGroup(flagGroup));
      }
      this.flagGroups = builder.build();
    }

    /**
     * Adds the flags that apply to the given {@code action} to {@code commandLine}.
     */
    private void expandCommandLine(String action, Variables variables, List<String> commandLine) {
      for (String variable : expandIfAllAvailable) {
        if (!variables.isAvailable(variable)) {
          return;
        }
      }
      if (!actions.contains(action)) {
        return;
      }
      for (FlagGroup flagGroup : flagGroups) {
        flagGroup.expandCommandLine(variables, commandLine);
      }
    }
  }
  
  /**
   * Groups a set of environment variables to apply for certain actions.
   */
  @Immutable
  private static class EnvSet implements Serializable {
    private final ImmutableSet<String> actions;
    private final ImmutableList<EnvEntry> envEntries;
    
    private EnvSet(CToolchain.EnvSet envSet) throws InvalidConfigurationException {
      this.actions = ImmutableSet.copyOf(envSet.getActionList());
      ImmutableList.Builder<EnvEntry> builder = ImmutableList.builder();
      for (CToolchain.EnvEntry envEntry : envSet.getEnvEntryList()) {
        builder.add(new EnvEntry(envEntry));
      }
      this.envEntries = builder.build();
    }

    /**
     * Adds the environment key/value pairs that apply to the given {@code action} to
     * {@code envBuilder}.
     */
    private void expandEnvironment(String action, Variables variables,
        ImmutableMap.Builder<String, String> envBuilder) {
      if (!actions.contains(action)) {
        return;
      }
      for (EnvEntry envEntry : envEntries) {
        envEntry.addEnvEntry(variables, envBuilder);
      }
    }
  }

  /**
   * An interface for classes representing crosstool messages that can activate eachother
   * using 'requires' and 'implies' semantics.
   *
   * <p>Currently there are two types of CrosstoolActivatable: Feature and ActionConfig.
   */
  private interface CrosstoolSelectable {

    /**
     * Returns the name of this selectable.
     */
    String getName();
  }

  /**
   * Contains flags for a specific feature.
   */
  @Immutable
  private static class Feature implements Serializable, CrosstoolSelectable {
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

    @Override
    public String getName() {
      return name;
    }

    /**
     * Adds environment variables for the given action to the provided builder.
     */
    private void expandEnvironment(
        String action, Variables variables, ImmutableMap.Builder<String, String> envBuilder) {
      for (EnvSet envSet : envSets) {
        envSet.expandEnvironment(action, variables, envBuilder);
      }
    }

    /**
     * Adds the flags that apply to the given {@code action} to {@code commandLine}.
     */
    private void expandCommandLine(String action, Variables variables,
        List<String> commandLine) {
      for (FlagSet flagSet : flagSets) {
        flagSet.expandCommandLine(action, variables, commandLine);
      }
    }
  }

  /**
   * An executable to be invoked by a blaze action.  Can carry information on its platform
   * restrictions.
   */
  @Immutable
  static class Tool {
    private final String toolPathString;
    private final ImmutableSet<String> executionRequirements;

    private Tool(CToolchain.Tool tool) {
      toolPathString = tool.getToolPath();
      executionRequirements = ImmutableSet.copyOf(tool.getExecutionRequirementList());
    }

    @VisibleForTesting
    public Tool(String toolPathString, ImmutableSet<String> executionRequirements) {
      this.toolPathString = toolPathString;
      this.executionRequirements = executionRequirements;
    }

    /**
     * Returns the path to this action's tool relative to the provided crosstool path.
     */
    PathFragment getToolPath(PathFragment crosstoolTopPathFragment) {
      return crosstoolTopPathFragment.getRelative(toolPathString);
    }

    /**
     * Returns a list of requirement hints that apply to the execution of this tool.
     */
    ImmutableSet<String> getExecutionRequirements() {
      return executionRequirements;
    }
  }
  
  
  /**
   * A container for information on a particular blaze action. 
   * 
   * <p>An ActionConfig can select a tool for its blaze action based on the set of active
   * features.  Internally, an ActionConfig maintains an ordered list (the order being that of the
   * list of tools in the crosstool action_config message) of such tools and the feature sets for 
   * which they are valid.  For a given feature configuration, the ActionConfig will consider the
   * first tool in that list with a feature set that matches the configuration to be the tool for
   * its blaze action.
   * 
   * <p>ActionConfigs can be activated by features.  That is, a particular feature can cause an
   * ActionConfig to be applied in its "implies" field.  Blaze may include certain actions in 
   * the action graph only if a corresponding ActionConfig is activated in the toolchain - this 
   * provides the crosstool with a mechanism for adding certain actions to the action graph based 
   * on feature configuration.
   * 
   * <p>It is invalid for a toolchain to contain two action configs for the same blaze action.  In
   * that case, blaze will throw an error when it consumes the crosstool.
   */
  @Immutable
  private static class ActionConfig implements Serializable, CrosstoolSelectable {
    private final String configName;
    private final String actionName;
    private final List<CToolchain.Tool> tools;

    private ActionConfig(CToolchain.ActionConfig actionConfig) {
      this.configName = actionConfig.getConfigName();
      this.actionName = actionConfig.getActionName();
      this.tools = actionConfig.getToolList();
    }

    @Override
    public String getName() {
      return configName;
    }

    /**
     * Returns the name of the blaze action this action config applies to.
     */
    private String getActionName() {
      return actionName;
    }

    /**
     * Returns the path to this action's tool relative to the provided crosstool path given a set
     * of enabled features.
     */
    private Tool getTool(final Set<String> enabledFeatureNames) {
      Optional<CToolchain.Tool> tool =
          Iterables.tryFind(
              tools,
              new Predicate<CToolchain.Tool>() {
                // We select the first listed tool for which all specified features are activated
                // in this configuration
                @Override
                public boolean apply(CToolchain.Tool input) {
                  Collection<String> featureNamesForTool = input.getWithFeature().getFeatureList();
                  return enabledFeatureNames.containsAll(featureNamesForTool);
                }
              });
      if (tool.isPresent()) {
        return new Tool(tool.get());
      } else {
        throw new IllegalArgumentException(
            "Matching tool for action "
                + getActionName()
                + " not "
                + "found for given feature configuration");
      }
    }
  }
  
  /**
   * Configured build variables usable by the toolchain configuration.
   */
  @Immutable
  public static class Variables {
    
    /**
     * Variables can be set as an arbitrarily deeply nested recursive sequence, which
     * we represent as a tree of {@code Sequence} nodes.
     * The nodes are {@code NestedSequence} objects, while the leafs are {@code ValueSequence}
     * objects. We do not allow {@code Value} objects in the tree, as the object memory overhead
     * is too large when we have millions of values.
     * If we find single element {@code ValueSequence} in memory profiles in the future, we
     * can introduce another special case type.
     */
    interface Sequence {

      /**
       * Expands {@code expandable} under the given nested {@code view}, appending flags to
       * {@code commandLine}.
       */
      void expand(NestedView view, Expandable expandable, List<String> commandLine);
    }

    /**
     * A sequence of simple string values.
     * Exists as a memory optimization - a typical build can contain millions of feature values,
     * so getting rid of the overhead of {@code Value} objects significantly reduces memory
     * overhead.
     */
    public static class ValueSequence implements Sequence {
      private final List<String> values;

      /** Builder for value sequences. */
      public static class Builder {
        private final ImmutableList.Builder<String> values = ImmutableList.builder();

        /** Adds a value to the sequence. */
        public Builder addValue(String value) {
          values.add(value);
          return this;
        }

        /** Returns an immutable value sequence. */
        public ValueSequence build() {
          return new ValueSequence(values.build());
        }
      }

      private ValueSequence(List<String> values) {
        this.values = values;
      }

      @Override
      public void expand(NestedView view, Expandable expandable, List<String> commandLine) {
        final ImmutableList.Builder<Sequence> sequences = ImmutableList.builder();
        for (String value : values) {
          sequences.add(new Value(value));
        }
        view.expandSequence(sequences.build(), expandable, commandLine);
      }

      /**
       * The leaves in the variable sequence node tree are simple values. Note that this should
       * never live outside of {@code expand}, as the object overhead is prohibitively expensive.
       */
      private static class Value implements Sequence {
        private final String value;

        private Value(String value) {
          this.value = value;
        }

        @Override
        public void expand(NestedView view, Expandable expandable, List<String> commandLine) {
          view.expandValue(value, expandable, commandLine);
        }
      }
    }

    /** An internal node in the sequence node tree. */
    static class NestedSequence implements Sequence {

      /**
       * Builder for nested sequences.
       */
      static class Builder {
        private final ImmutableList.Builder<Sequence> sequences = ImmutableList.builder();

        /**
         * Adds a sub-sequence to the sequence.
         */
        Builder addSequence(Sequence sequence) {
          sequences.add(sequence);
          return this;
        }

        /**
         * Returns an immutable nested sequence.
         */
        NestedSequence build() {
          return new NestedSequence(sequences.build());
        }
      }

      private final List<Sequence> sequences;

      private NestedSequence(List<Sequence> expandables) {
        this.sequences = expandables;
      }

      @Override
      public void expand(NestedView view, Expandable expandable, List<String> commandLine) {
        view.expandSequence(sequences, expandable, commandLine);
      }
    }

    /**
     * Builder for {@code Variables}.
     */
    public static class Builder {
      private final ImmutableMap.Builder<String, String> variables = ImmutableMap.builder();
      private final ImmutableMap.Builder<String, Sequence> expandables = ImmutableMap.builder();
      
      /**
       * Add a variable that expands {@code name} to {@code value}.
       */
      public Builder addVariable(String name, String value) {
        variables.put(name, value);
        return this;
      }
     
      /**
       * Add all variables in a map.
       */
      public Builder addAllVariables(Map<String, String> variableMap) {
        variables.putAll(variableMap);
        return this;
      }
      
      /**
       * Add a nested sequence that expands {@code name} recursively.
       */
      public Builder addSequence(String name, Sequence sequence) {
        expandables.put(name, sequence);
        return this;
      }

      /**
       * Add a variable that expands a flag group containing a reference to {@code name} for each
       * entry in {@code values}.
       */
      public Builder addSequenceVariable(String name, Collection<String> values) {
        ValueSequence.Builder sequenceBuilder = new ValueSequence.Builder();
        for (String value : values) {
          sequenceBuilder.addValue(value);
        }
        return addSequence(name, sequenceBuilder.build());
      }
      
      /**
       * @return a new {@Variables} object.
       */
      Variables build() {
        return new Variables(variables.build(), expandables.build());
      }
    }
    
    /**
     * A group of extra {@code Variable} instances, packaged as logic for adding to a
     * {@code Builder}
     */
    public interface VariablesExtension {
      void addVariables(Builder builder);
    }
    
    /**
     * Interface for a set of variables visible during an expansion of a variable.
     */
    interface View {

      /**
       * Returns all bounds variables in the current view.
       */
      Map<String, String> getVariables();

      /**
       * Expands the given {@code expandable} under the current view, adding flags to
       * {@code commandLine}.
       */
      void expand(Expandable expandable, List<String> commandLine);
    }

    /**
     * An simple view that contains a fixed mapping.
     */
    private static class FixedView implements View {
      private final Map<String, String> viewMap;

      FixedView(Map<String, String> viewMap) {
        this.viewMap = viewMap;
      }

      @Override
      public Map<String, String> getVariables() {
        return viewMap;
      }

      @Override
      public void expand(Expandable expandable, List<String> commandLine) {
        expandable.expand(this, commandLine);
      }
    }

    /**
     * A nested view that is controlled by a nested build variable and expanded recursively.
     */
    private static class NestedView implements View {

      /**
       * The view map will contain all mapped variables when a leaf node is reached.
       * During traversal it will not contain the control variable.
       */
      private final Map<String, String> viewMap;

      /**
       * The name of the control variable (a variable of nested structure) that controls
       * the recursive expansion.
       */
      String controlVariable;

      /**
       * The stack of sequences that are currently being expanded. Each level represents
       * one level in the control variables nesting.
       */
      Stack<Sequence> sequenceStack = new Stack<>();

      NestedView(Map<String, String> viewMap, String controlVariable, Sequence sequence) {
        this.viewMap = viewMap;
        this.controlVariable = controlVariable;
        this.sequenceStack.push(sequence);
      }

      @Override
      public Map<String, String> getVariables() {
        return viewMap;
      }

      @Override
      public void expand(Expandable expandable, List<String> commandLine) {
        Sequence sequence = sequenceStack.peek();
        sequence.expand(this, expandable, commandLine);
      }

      /**
       * Expands {@code expandable} with the control variable set to {@code value}.
       */
      void expandValue(String value, Expandable expandable, List<String> commandLine) {
        viewMap.put(controlVariable, value);
        expandable.expand(this, commandLine);
        viewMap.remove(controlVariable);
      }

      /**
       * Expands {@code expandable}. If the {@code expandable} is controlled by the current view's
       * control variable, it will be expanded for each given sequence. Otherwise it will be
       * expanded once.
       */
      void expandSequence(
          Collection<Sequence> sequences, Expandable expandable, List<String> commandLine) {
        if (!controls(expandable)) {
          // If an expandable does not reference the control variable, we only want to expand
          // it once.
          expandable.expand(this, commandLine);
          return;
        }
        // Expandables that are controlled by the control variable will be expanded for each
        // sequence at the current nesting level of the control variable's content.
        for (Sequence sequence : sequences) {
          sequenceStack.push(sequence);
          expandable.expand(this, commandLine);
          sequenceStack.pop();
        }
      }

      /**
       * Returns whether the expansion of {@code expandable} is controlled by the controlling
       * variable of the current view.
       */
      boolean controls(Expandable expandable) {
        if (expandable.getControllingVariables() == null) {
          return false;
        }
        return expandable.getControllingVariables().contains(controlVariable);
      }
    }
    
    private final ImmutableMap<String, String> variables;
    private final ImmutableMap<String, Sequence> sequenceVariables;

    private Variables(
        ImmutableMap<String, String> variables, ImmutableMap<String, Sequence> sequenceVariables) {
      this.variables = variables;
      this.sequenceVariables = sequenceVariables;
    }
    
    /**
     * Returns a view of the current variables under the given {@code controllingVariables}.
     * Verifies that all controlling variables are available.
     */
    View getView(Collection<String> controllingVariables) {
      Map<String, String> viewMap = new HashMap<>();
      String sequenceName = null; 
      for (String name : controllingVariables) {
        if (sequenceVariables.containsKey(name)) {
          if (variables.containsKey(name)) {
            throw new ExpansionException("Internal error: variable '" + name
                + "' provided both as sequence and standard variable.");
          } else if (sequenceName != null) {
            throw new ExpansionException(
                "Invalid toolchain configuration: trying to expand two variable list in one "
                + "flag group: '" + sequenceName + "' and '" + name + "'");
          } else {
            sequenceName = name;
          }
        } else if (variables.containsKey(name)) {
          viewMap.put(name, variables.get(name));
        } else {
          throw new ExpansionException("Invalid toolchain configuration: unknown variable '" + name
              + "' can not be expanded.");
        }
      }
      if (sequenceName == null) {
        return new FixedView(viewMap);
      }
      return new NestedView(viewMap, sequenceName, sequenceVariables.get(sequenceName));
    }

    /**
     * Returns whether {@code variable} is set.
     */
    private boolean isAvailable(String variable) {
      return variables.containsKey(variable) || sequenceVariables.containsKey(variable);
    }
  }
  
  /**
   * Captures the set of enabled features and action configs for a rule.
   */
  @Immutable
  public static class FeatureConfiguration {
    private final ImmutableSet<String> enabledFeatureNames;
    private final Iterable<Feature> enabledFeatures;
    private final ImmutableSet<String> enabledActionConfigActionNames;
    
    private final ImmutableMap<String, ActionConfig> actionConfigByActionName;
    
    public FeatureConfiguration() {
      this(
          ImmutableList.<Feature>of(),
          ImmutableList.<ActionConfig>of(),
          ImmutableMap.<String, ActionConfig>of());
    }

    private FeatureConfiguration(
        Iterable<Feature> enabledFeatures,
        Iterable<ActionConfig> enabledActionConfigs,
        ImmutableMap<String, ActionConfig> actionConfigByActionName) {
      this.enabledFeatures = enabledFeatures;
      
      this.actionConfigByActionName = actionConfigByActionName;
      ImmutableSet.Builder<String> featureBuilder = ImmutableSet.builder();
      for (Feature feature : enabledFeatures) {
        featureBuilder.add(feature.getName());
      }
      this.enabledFeatureNames = featureBuilder.build();
      
      ImmutableSet.Builder<String> actionConfigBuilder = ImmutableSet.builder();
      for (ActionConfig actionConfig : enabledActionConfigs) {
        actionConfigBuilder.add(actionConfig.getActionName());
      }
      this.enabledActionConfigActionNames = actionConfigBuilder.build();
    }
    
    /**
     * @return whether the given {@code feature} is enabled.
     */
    public boolean isEnabled(String feature) {
      return enabledFeatureNames.contains(feature);
    }

    /**
     * @return whether an action config for the blaze action with the given name is enabled.
     */
    boolean actionIsConfigured(String actionName) {
      return enabledActionConfigActionNames.contains(actionName);
    }

    /**
     * @return the command line for the given {@code action}.
     */
    List<String> getCommandLine(String action, Variables variables) {
      List<String> commandLine = new ArrayList<>();
      for (Feature feature : enabledFeatures) {
        feature.expandCommandLine(action, variables, commandLine);
      }
      return commandLine;
    }

    /**
     * @return the environment variables (key/value pairs) for the given {@code action}.
     */
    Map<String, String> getEnvironmentVariables(String action, Variables variables) {
      ImmutableMap.Builder<String, String> envBuilder = ImmutableMap.builder();
      for (Feature feature : enabledFeatures) {
        feature.expandEnvironment(action, variables, envBuilder);
      }
      return envBuilder.build();
    }
  
    /**
     * Returns a given action's tool under this FeatureConfiguration.
     */
    Tool getToolForAction(String actionName) {
      Preconditions.checkArgument(
          actionConfigByActionName.containsKey(actionName),
          "Action %s does not have an enabled configuration in the toolchain.",
          actionName);
      ActionConfig actionConfig = actionConfigByActionName.get(actionName);
      return actionConfig.getTool(enabledFeatureNames);
    }
  }
  
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
   * Maps from a selectable to all selectables that have a requirement referencing it.
   *
   * <p>This will be used to determine which selectables need to be re-checked after a selectable
   * was disabled.
   */
  private final ImmutableMultimap<CrosstoolSelectable, CrosstoolSelectable> requiredBy;
 
  /**
   * A cache of feature selection results, so we do not recalculate the feature selection for
   * all actions.
   */
  private transient LoadingCache<Collection<String>, FeatureConfiguration>
      configurationCache = buildConfigurationCache();

  /**
   * Constructs the feature configuration from a {@code CToolchain} protocol buffer.
   *
   * @param toolchain the toolchain configuration as specified by the user.
   * @throws InvalidConfigurationException if the configuration has logical errors.
   */
  @VisibleForTesting
  public CcToolchainFeatures(CToolchain toolchain) throws InvalidConfigurationException {
    // Build up the feature/action config graph.  We refer to features/action configs as
    // 'selectables'.
    // First, we build up the map of name -> selectables in one pass, so that earlier selectables
    // can reference later features in their configuration.
    ImmutableList.Builder<CrosstoolSelectable> selectablesBuilder = ImmutableList.builder();
    HashMap<String, CrosstoolSelectable> selectablesByName = new HashMap<>();

    // Also build a map from action -> action_config, for use in tool lookups
    ImmutableMap.Builder<String, ActionConfig> actionConfigsByActionName = ImmutableMap.builder();

    for (CToolchain.Feature toolchainFeature : toolchain.getFeatureList()) {
      Feature feature = new Feature(toolchainFeature);
      selectablesBuilder.add(feature);
      selectablesByName.put(feature.getName(), feature);
    }
    
    for (CToolchain.ActionConfig toolchainActionConfig : toolchain.getActionConfigList()) {
      ActionConfig actionConfig = new ActionConfig(toolchainActionConfig);
      selectablesBuilder.add(actionConfig);
      selectablesByName.put(actionConfig.getName(), actionConfig);
      actionConfigsByActionName.put(actionConfig.getActionName(), actionConfig);
    }

    this.selectables = selectablesBuilder.build();
    this.selectablesByName = ImmutableMap.copyOf(selectablesByName);

    checkForActionNameDups(toolchain.getActionConfigList());
    checkForActivatableDups(this.selectables);

    this.actionConfigsByActionName = actionConfigsByActionName.build();

    // Next, we build up all forward references for 'implies' and 'requires' edges.
    ImmutableMultimap.Builder<CrosstoolSelectable, CrosstoolSelectable> implies =
        ImmutableMultimap.builder();
    ImmutableMultimap.Builder<CrosstoolSelectable, ImmutableSet<CrosstoolSelectable>> requires =
        ImmutableMultimap.builder();
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
    this.impliedBy = impliedBy.build();
    this.requiredBy = requiredBy.build();
  }

  private static void checkForActivatableDups(Iterable<CrosstoolSelectable> selectables)
      throws InvalidConfigurationException {
    Collection<String> names = new HashSet<>();
    for (CrosstoolSelectable selectable : selectables) {
      if (!names.add(selectable.getName())) {
        throw new InvalidConfigurationException(
            "Invalid toolcahin configuration: feature or "
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
  
  /**
   * @return an empty {@code FeatureConfiguration} cache. 
   */
  private LoadingCache<Collection<String>, FeatureConfiguration> buildConfigurationCache() {
    return CacheBuilder.newBuilder()
        // TODO(klimek): Benchmark and tweak once we support a larger configuration. 
        .maximumSize(10000)
        .build(new CacheLoader<Collection<String>, FeatureConfiguration>() {
          @Override
          public FeatureConfiguration load(Collection<String> requestedFeatures) {
            return computeFeatureConfiguration(requestedFeatures);
          }
        });
  }

  /**
   * Given a list of {@code requestedFeatures}, returns all features that are enabled by the
   * toolchain configuration.
   *
   * <p>A requested feature will not be enabled if the toolchain does not support it (which may
   * depend on other requested features).
   *
   * <p>Additional features will be enabled if the toolchain supports them and they are implied by
   * requested features.
   */
  public FeatureConfiguration getFeatureConfiguration(Collection<String> requestedFeatures) {
    return configurationCache.getUnchecked(requestedFeatures);
  }
      
  private FeatureConfiguration computeFeatureConfiguration(Collection<String> requestedFeatures) { 
    // Command line flags will be output in the order in which they are specified in the toolchain
    // configuration.
    return new FeatureSelection(requestedFeatures).run();
  }
  
  /**
   * Given a list of {@code requestedFeatures}, returns all features that are enabled by the
   * toolchain configuration.
   *
   * <p>A requested feature will not be enabled if the toolchain does not support it (which may
   * depend on other requested features).
   *
   * <p>Additional features will be enabled if the toolchain supports them and they are implied by
   * requested features.
   */ 
  public FeatureConfiguration getFeatureConfiguration(String... requestedFeatures) {
    return getFeatureConfiguration(Arrays.asList(requestedFeatures));
  }

  /**
   * @return the selectable with the given {@code name}.
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
   * Implements the feature selection algorithm.
   * 
   * <p>Feature selection is done by first enabling all features reachable by an 'implies' edge,
   * and then iteratively pruning features that have unmet requirements.
   */
  private class FeatureSelection {
    
    /**
     * The selectables Bazel would like to enable; either because they are supported and generally
     * useful, or because the user required them (for example through the command line).
     */
    private final ImmutableSet<CrosstoolSelectable> requestedSelectables;
    
    /**
     * The currently enabled selectable; during feature selection, we first put all selectables
     * reachable via an 'implies' edge into the enabled selectable set, and than prune that set
     * from selectables that have unmet requirements.
     */
    private final Set<CrosstoolSelectable> enabled = new HashSet<>();
    
    private FeatureSelection(Collection<String> requestedSelectables) {
      ImmutableSet.Builder<CrosstoolSelectable> builder = ImmutableSet.builder();
      for (String name : requestedSelectables) {
        if (selectablesByName.containsKey(name)) {
          builder.add(selectablesByName.get(name));
        }
      }
      this.requestedSelectables = builder.build();
    }

    /**
     * @return a {@code FeatureConfiguration} that reflects the set of activated features and
     * action configs.
     */
    private FeatureConfiguration run() {
      for (CrosstoolSelectable selectable : requestedSelectables) {
        enableAllImpliedBy(selectable);
      }

      disableUnsupportedActivatables();
      ImmutableList.Builder<CrosstoolSelectable> enabledActivatablesInOrderBuilder =
          ImmutableList.builder();
      for (CrosstoolSelectable selectable : selectables) {
        if (enabled.contains(selectable)) {
          enabledActivatablesInOrderBuilder.add(selectable);
        }
      }
      
      ImmutableList<CrosstoolSelectable> enabledActivatablesInOrder =
          enabledActivatablesInOrderBuilder.build();
      Iterable<Feature> enabledFeaturesInOrder =
          Iterables.filter(enabledActivatablesInOrder, Feature.class);
      Iterable<ActionConfig> enabledActionConfigsInOrder =
          Iterables.filter(enabledActivatablesInOrder, ActionConfig.class);

      return new FeatureConfiguration(
          enabledFeaturesInOrder, enabledActionConfigsInOrder, actionConfigsByActionName);
    }
    
    /**
     * Transitively and unconditionally enable all selectables implied by the given selectable
     * and the selectable itself to the enabled selectable set.
     */
    private void enableAllImpliedBy(CrosstoolSelectable selectable) {
      if (enabled.contains(selectable)) {
        return;
      }
      enabled.add(selectable);
      for (CrosstoolSelectable implied : implies.get(selectable)) {
        enableAllImpliedBy(implied);
      }
    }
    
    /**
     * Remove all unsupported features from the enabled feature set.
     */
    private void disableUnsupportedActivatables() {
      Queue<CrosstoolSelectable> check = new ArrayDeque<>(enabled);
      while (!check.isEmpty()) {
        checkActivatable(check.poll());
      }
    }
    
    /**
     * Check if the given selectable is still satisfied within the set of currently enabled
     * selectables.
     *
     * <p>If it is not, remove the selectable from the set of enabled selectables, and re-check
     * all selectables that may now also become disabled.
     */
    private void checkActivatable(CrosstoolSelectable selectable) {
      if (!enabled.contains(selectable) || isSatisfied(selectable)) {
        return;
      }
      enabled.remove(selectable);

      // Once we disable a selectable, we have to re-check all selectables that can be affected
      // by that removal.
      // 1. A selectable that implied the current selectable is now going to be disabled.
      for (CrosstoolSelectable impliesCurrent : impliedBy.get(selectable)) {
        checkActivatable(impliesCurrent);
      }
      // 2. A selectable that required the current selectable may now be disabled, depending on
      // whether the requirement was optional.
      for (CrosstoolSelectable requiresCurrent : requiredBy.get(selectable)) {
        checkActivatable(requiresCurrent);
      }
      // 3. A selectable that this selectable implied may now be disabled if no other selectables
      // also implies it.
      for (CrosstoolSelectable implied : implies.get(selectable)) {
        checkActivatable(implied);
      }
    }

    /**
     * @return whether all requirements of the selectable are met in the set of currently enabled
     * selectables.
     */
    private boolean isSatisfied(CrosstoolSelectable selectable) {
      return (requestedSelectables.contains(selectable)
              || isImpliedByEnabledActivatable(selectable))
          && allImplicationsEnabled(selectable)
          && allRequirementsMet(selectable);
    }
    
    /**
     * @return whether a currently enabled selectable implies the given selectable.
     */
    private boolean isImpliedByEnabledActivatable(CrosstoolSelectable selectable) {
      return !Collections.disjoint(impliedBy.get(selectable), enabled);
    }
        
    /**
     * @return whether all implications of the given feature are enabled.
     */
    private boolean allImplicationsEnabled(CrosstoolSelectable selectable) {
      for (CrosstoolSelectable implied : implies.get(selectable)) {
        if (!enabled.contains(implied)) {
          return false;
        }
      }
      return true;
    }
    
    /**
     * @return whether all requirements are enabled.
     *
     * <p>This implies that for any of the selectable sets all of the specified selectable
     *   are enabled.
     */
    private boolean allRequirementsMet(CrosstoolSelectable feature) {
      if (!requires.containsKey(feature)) {
        return true;
      }
      for (ImmutableSet<CrosstoolSelectable> requiresAllOf : requires.get(feature)) {
        boolean requirementMet = true;
        for (CrosstoolSelectable required : requiresAllOf) {
          if (!enabled.contains(required)) {
            requirementMet = false;
            break;
          }
        }
        if (requirementMet) {
          return true;
        }
      }
      return false;
    }
  }
}
