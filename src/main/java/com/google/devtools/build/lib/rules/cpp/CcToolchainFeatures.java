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
import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableMultimap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CToolchain;

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
   * Contains flags for a specific feature.
   */
  @Immutable
  private static class Feature implements Serializable {
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

    /**
     * @return the features's name.
     */
    private String getName() {
      return name;
    }

    private void expandEnvironment(String action, Variables variables,
        ImmutableMap.Builder<String, String> envBuilder) {
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
    static class ValueSequence implements Sequence {
      private final List<String> values;

      /** Builder for value sequences. */
      static class Builder {
        private final ImmutableList.Builder<String> values = ImmutableList.builder();

        /** Adds a value to the sequence. */
        Builder addValue(String value) {
          values.add(value);
          return this;
        }

        /** Returns an immutable value sequence. */
        ValueSequence build() {
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
    static class Builder {
      private final ImmutableMap.Builder<String, String> variables = ImmutableMap.builder();
      private final ImmutableMap.Builder<String, Sequence> expandables = ImmutableMap.builder();
      
      /**
       * Add a variable that expands {@code name} to {@code value}.
       */
      Builder addVariable(String name, String value) {
        variables.put(name, value);
        return this;
      }
     
      /**
       * Add all variables in a map.
       */
      Builder addAllVariables(Map<String, String> variableMap) {
        variables.putAll(variableMap);
        return this;
      }
      
      /**
       * Add a nested sequence that expands {@code name} recursively.
       */
      Builder addSequence(String name, Sequence sequence) {
        expandables.put(name, sequence);
        return this;
      }

      /**
       * Add a variable that expands a flag group containing a reference to {@code name} for each
       * entry in {@code values}.
       */
      Builder addSequenceVariable(String name, Collection<String> values) {
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
   * Captures the set of enabled features for a rule.
   */
  @Immutable
  public static class FeatureConfiguration {
    private final ImmutableSet<String> enabledFeatureNames;
    private final ImmutableList<Feature> enabledFeatures;
    
    public FeatureConfiguration() {
      enabledFeatureNames = ImmutableSet.of();
      enabledFeatures = ImmutableList.of();
    }
    
    private FeatureConfiguration(ImmutableList<Feature> enabledFeatures) {
      this.enabledFeatures = enabledFeatures;
      ImmutableSet.Builder<String> builder = ImmutableSet.builder();
      for (Feature feature : enabledFeatures) {
        builder.add(feature.getName());
      }
      this.enabledFeatureNames = builder.build();
    }
    
    /**
     * @return whether the given {@code feature} is enabled.
     */
    boolean isEnabled(String feature) {
      return enabledFeatureNames.contains(feature);
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
  }
  
  /**
   * All features in the order in which they were specified in the configuration.
   *
   * <p>We guarantee the command line to be in the order in which the flags were specified in the
   * configuration.
   */
  private final ImmutableList<Feature> features;
  
  /**
   * Maps from the feature's name to the feature.
   */
  private final ImmutableMap<String, Feature> featuresByName;
  
  /**
   * Maps from a feature to a set of all the features it has a direct 'implies' edge to.
   */
  private final ImmutableMultimap<Feature, Feature> implies;
  
  /**
   * Maps from a feature to all features that have an direct 'implies' edge to this feature. 
   */
  private final ImmutableMultimap<Feature, Feature> impliedBy;
  
  /**
   * Maps from a feature to a set of feature sets, where:
   * <ul>
   * <li>a feature set satisfies the 'requires' condition, if all features in the feature set are
   *     enabled</li>
   * <li>the 'requires' condition is satisfied, if at least one of the feature sets satisfies the
   *     'requires' condition.</li>
   * </ul> 
   */
  private final ImmutableMultimap<Feature, ImmutableSet<Feature>> requires;
  
  /**
   * Maps from a feature to all features that have a requirement referencing it.
   * 
   * <p>This will be used to determine which features need to be re-checked after a feature was
   * disabled.
   */
  private final ImmutableMultimap<Feature, Feature> requiredBy;
  
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
    // Build up the feature graph.
    // First, we build up the map of name -> features in one pass, so that earlier features can
    // reference later features in their configuration.
    ImmutableList.Builder<Feature> features = ImmutableList.builder();
    HashMap<String, Feature> featuresByName = new HashMap<>();
    for (CToolchain.Feature toolchainFeature : toolchain.getFeatureList()) {
      Feature feature = new Feature(toolchainFeature);
      features.add(feature);
      if (featuresByName.put(feature.getName(), feature) != null) {
        throw new InvalidConfigurationException("Invalid toolchain configuration: feature '"
            + feature.getName() + "' was specified multiple times.");
      }
    }
    this.features = features.build();
    this.featuresByName = ImmutableMap.copyOf(featuresByName);
    
    // Next, we build up all forward references for 'implies' and 'requires' edges.
    ImmutableMultimap.Builder<Feature, Feature> implies = ImmutableMultimap.builder();
    ImmutableMultimap.Builder<Feature, ImmutableSet<Feature>> requires =
        ImmutableMultimap.builder();
    // We also store the reverse 'implied by' and 'required by' edges during this pass. 
    ImmutableMultimap.Builder<Feature, Feature> impliedBy = ImmutableMultimap.builder();
    ImmutableMultimap.Builder<Feature, Feature> requiredBy = ImmutableMultimap.builder();
    for (CToolchain.Feature toolchainFeature : toolchain.getFeatureList()) {
      String name = toolchainFeature.getName();
      Feature feature = featuresByName.get(name);
      for (CToolchain.FeatureSet requiredFeatures : toolchainFeature.getRequiresList()) {
        ImmutableSet.Builder<Feature> allOf = ImmutableSet.builder(); 
        for (String requiredName : requiredFeatures.getFeatureList()) {
          Feature required = getFeatureOrFail(requiredName, name);
          allOf.add(required);
          requiredBy.put(required, feature);
        }
        requires.put(feature, allOf.build());
      }
      for (String impliedName : toolchainFeature.getImpliesList()) {
        Feature implied = getFeatureOrFail(impliedName, name);
        impliedBy.put(implied, feature);
        implies.put(feature, implied);
      }
    }
    this.implies = implies.build();
    this.requires = requires.build();
    this.impliedBy = impliedBy.build();
    this.requiredBy = requiredBy.build();
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
  FeatureConfiguration getFeatureConfiguration(Collection<String> requestedFeatures) {
    return configurationCache.getUnchecked(requestedFeatures);
  }
      
  private FeatureConfiguration computeFeatureConfiguration(Collection<String> requestedFeatures) { 
    // Command line flags will be output in the order in which they are specified in the toolchain
    // configuration.
    return new FeatureSelection(requestedFeatures).run();
  }
  
  /**
   * Convenience method taking a variadic string argument list for testing.   
   */
  public FeatureConfiguration getFeatureConfiguration(String... requestedFeatures) {
    return getFeatureConfiguration(Arrays.asList(requestedFeatures));
  }

  /**
   * @return the feature with the given {@code name}.
   * 
   * @throws InvalidConfigurationException if no feature with the given name was configured.
   */
  private Feature getFeatureOrFail(String name, String reference)
      throws InvalidConfigurationException {
    if (!featuresByName.containsKey(name)) {
      throw new InvalidConfigurationException("Invalid toolchain configuration: feature '" + name
          + "', which is referenced from feature '" + reference + "', is not defined.");
    }
    return featuresByName.get(name);
  }
  
  @VisibleForTesting
  Collection<String> getFeatureNames() {
    Collection<String> featureNames = new HashSet<>();
    for (Feature feature : features) {
      featureNames.add(feature.getName());
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
     * The features Bazel would like to enable; either because they are supported and generally
     * useful, or because the user required them (for example through the command line). 
     */
    private final ImmutableSet<Feature> requestedFeatures;
    
    /**
     * The currently enabled feature; during feature selection, we first put all features reachable
     * via an 'implies' edge into the enabled feature set, and than prune that set from features
     * that have unmet requirements.
     */
    private Set<Feature> enabled = new HashSet<>();
    
    private FeatureSelection(Collection<String> requestedFeatures) {
      ImmutableSet.Builder<Feature> builder = ImmutableSet.builder();
      for (String name : requestedFeatures) {
        if (featuresByName.containsKey(name)) {
          builder.add(featuresByName.get(name));
        }
      }
      this.requestedFeatures = builder.build();
    }

    /**
     * @return all enabled features in the order in which they were specified in the configuration.
     */
    private FeatureConfiguration run() {
      for (Feature feature : requestedFeatures) {
        enableAllImpliedBy(feature);
      }
      disableUnsupportedFeatures();
      ImmutableList.Builder<Feature> enabledFeaturesInOrder = ImmutableList.builder(); 
      for (Feature feature : features) {
        if (enabled.contains(feature)) {
          enabledFeaturesInOrder.add(feature);
        }
      }
      return new FeatureConfiguration(enabledFeaturesInOrder.build());
    }
    
    /**
     * Transitively and unconditionally enable all features implied by the given feature and the
     * feature itself to the enabled feature set.
     */
    private void enableAllImpliedBy(Feature feature) {
      if (enabled.contains(feature)) {
        return;
      }
      enabled.add(feature);
      for (Feature implied : implies.get(feature)) {
        enableAllImpliedBy(implied);
      }
    }
    
    /**
     * Remove all unsupported features from the enabled feature set.
     */
    private void disableUnsupportedFeatures() {
      Queue<Feature> check = new ArrayDeque<>(enabled);
      while (!check.isEmpty()) {
        checkFeature(check.poll());
      }
    }
    
    /**
     * Check if the given feature is still satisfied within the set of currently enabled features.
     * 
     * <p>If it is not, remove the feature from the set of enabled features, and re-check all
     * features that may now also become disabled.
     */
    private void checkFeature(Feature feature) {
      if (!enabled.contains(feature) || isSatisfied(feature)) {
        return;
      }
      enabled.remove(feature);
      
      // Once we disable a feature, we have to re-check all features that can be affected by
      // that removal.
      // 1. A feature that implied the current feature is now going to be disabled.
      for (Feature impliesCurrent : impliedBy.get(feature)) {
        checkFeature(impliesCurrent);
      }
      // 2. A feature that required the current feature may now be disabled, depending on whether
      //    the requirement was optional.
      for (Feature requiresCurrent : requiredBy.get(feature)) {
        checkFeature(requiresCurrent);
      }
      // 3. A feature that this feature implied may now be disabled if no other feature also implies
      //    it.
      for (Feature implied : implies.get(feature)) {
        checkFeature(implied);
      }
    }

    /**
     * @return whether all requirements of the feature are met in the set of currently enabled
     * features.
     */
    private boolean isSatisfied(Feature feature) {
      return (requestedFeatures.contains(feature) || isImpliedByEnabledFeature(feature))
          && allImplicationsEnabled(feature) && allRequirementsMet(feature);
    }
    
    /**
     * @return whether a currently enabled feature implies the given feature.
     */
    private boolean isImpliedByEnabledFeature(Feature feature) {
      return !Collections.disjoint(impliedBy.get(feature), enabled);
    }
        
    /**
     * @return whether all implications of the given feature are enabled.
     */
    private boolean allImplicationsEnabled(Feature feature) {
      for (Feature implied : implies.get(feature)) {
        if (!enabled.contains(implied)) {
          return false;
        }
      }
      return true;
    }
    
    /**
     * @return whether all requirements are enabled.
     * 
     * <p>This implies that for any of the feature sets all of the specified features are enabled.
     */
    private boolean allRequirementsMet(Feature feature) {
      if (!requires.containsKey(feature)) {
        return true;
      }
      for (ImmutableSet<Feature> requiresAllOf : requires.get(feature)) {
        boolean requirementMet = true;
        for (Feature required : requiresAllOf) {
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
