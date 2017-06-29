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
import com.google.common.base.Joiner;
import com.google.common.base.Optional;
import com.google.common.base.Preconditions;
import com.google.common.base.Strings;
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
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.Variables.VariableValue;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CToolchain;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.Serializable;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
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

  /**
   * Thrown when multiple features provide the same string symbol.
   */
  public static class CollidingProvidesException extends RuntimeException {
    CollidingProvidesException(String message) {
      super(message);
    }
  }

  /** Error message thrown when a toolchain does not provide a required artifact_name_pattern. */
  public static final String MISSING_ARTIFACT_NAME_PATTERN_ERROR_TEMPLATE =
      "Toolchain must provide artifact_name_pattern for category %s";

  /** Error message thrown when a toolchain enables two features that provide the same string. */
  @VisibleForTesting static final String COLLIDING_PROVIDES_ERROR =
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
     * @param flag the flag content to append to.
     */
    void expand(Variables variables, StringBuilder flag);
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
    public void expand(Variables variables, StringBuilder flag) {
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
    public void expand(Variables variables, StringBuilder stringBuilder) {
      // We check all variables in FlagGroup.expandCommandLine.
      // If we arrive here with the variable not being available, the variable was provided, but
      // the nesting level of the NestedSequence was deeper than the nesting level of the flag
      // groups.
      stringBuilder.append(variables.getStringVariable(variableName));
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
  
  /**
   * A flag or flag group that can be expanded under a set of variables.
   */
  interface Expandable {
    /**
     * Expands the current expandable under the given {@code view}, adding new flags to {@code
     * commandLine}.
     *
     * <p>The {@code variables} controls which variables are visible during the expansion and allows
     * to recursively expand nested flag groups.
     */
    void expand(Variables variables, List<String> commandLine);
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
    
    /** Expand this flag into a single new entry in {@code commandLine}. */
    @Override
    public void expand(Variables variables, List<String> commandLine) {
      StringBuilder flag = new StringBuilder();
      for (StringChunk chunk : chunks) {
        chunk.expand(variables, flag);
      }
      commandLine.add(flag.toString());
    }
  }
  
  /**
   * A single environment key/value pair to be expanded under a set of variables.
   */
  @Immutable
  private static class EnvEntry implements Serializable {
    private final String key;
    private final ImmutableList<StringChunk> valueChunks;

    private EnvEntry(CToolchain.EnvEntry envEntry) throws InvalidConfigurationException {
      this.key = envEntry.getKey();
      StringValueParser parser = new StringValueParser(envEntry.getValue());
      this.valueChunks = parser.getChunks();
    }

    /**
     * Adds the key/value pair this object represents to the given map of environment variables.
     * The value of the entry is expanded with the given {@code variables}.
     */
    public void addEnvEntry(Variables variables, ImmutableMap.Builder<String, String> envBuilder) {
      StringBuilder value = new StringBuilder();
      for (StringChunk chunk : valueChunks) {
        chunk.expand(variables, value);
      }
      envBuilder.put(key, value.toString());
    }
  }

  @Immutable
  private static class VariableWithValue {
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
  private static class FlagGroup implements Serializable, Expandable {
    private final ImmutableList<Expandable> expandables;
    private final ImmutableSet<String> usedVariables;
    private String iterateOverVariable;
    private final ImmutableSet<String> expandIfAllAvailable;
    private final ImmutableSet<String> expandIfNoneAvailable;
    private final String expandIfTrue;
    private final String expandIfFalse;
    private final VariableWithValue expandIfEqual;

    /**
     * TODO(b/32655571): Cleanup and get rid of usedVariables field once implicit iteration is not
     * needed.
     *
     * @throws InvalidConfigurationException
     */
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
        usedVariables.addAll(subgroup.getUsedVariables());
      }
      if (flagGroup.hasIterateOver()) {
        this.iterateOverVariable = flagGroup.getIterateOver();
        usedVariables.add(this.iterateOverVariable);
      }
      this.usedVariables = usedVariables.build();
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
    
    @Override
    public void expand(Variables variables, final List<String> commandLine) {
      if (!canBeExpanded(variables)) {
        return;
      }
      if (iterateOverVariable == null) {
        // TODO(b/32655571): Remove branch once implicit iteration is not needed anymore.
        iterateOverVariable = variables.guessIteratedOverVariable(usedVariables);
      }
      if (iterateOverVariable != null) {
        for (VariableValue variableValue : variables.getSequenceVariable(iterateOverVariable)) {
          Variables nestedVariables = new Variables(variables, iterateOverVariable, variableValue);
          for (Expandable expandable : expandables) {
            expandable.expand(nestedVariables, commandLine);
          }
        }
      } else {
        for (Expandable expandable : expandables) {
          expandable.expand(variables, commandLine);
        }
      }
    }

    private boolean canBeExpanded(Variables variables) {
      for (String variable : expandIfAllAvailable) {
        if (!variables.isAvailable(variable)) {
          return false;
        }
      }
      for (String variable : expandIfNoneAvailable) {
        if (variables.isAvailable(variable)) {
          return false;
        }
      }
      if (expandIfTrue != null
          && (!variables.isAvailable(expandIfTrue)
              || !variables.getVariable(expandIfTrue).isTruthy())) {
        return false;
      }
      if (expandIfFalse != null
          && (!variables.isAvailable(expandIfFalse)
              || variables.getVariable(expandIfFalse).isTruthy())) {
        return false;
      }
      if (expandIfEqual != null
          && (!variables.isAvailable(expandIfEqual.variable)
              || !variables
                .getVariable(expandIfEqual.variable)
                .getStringValue(expandIfEqual.variable)
                .equals(expandIfEqual.value))) {
        return false;
      }
      return true;
    }

    private Set<String> getUsedVariables() {
      return usedVariables;
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
    private void expandCommandLine(Variables variables, final List<String> commandLine) {
      expand(variables, commandLine);
    }
  }

  private static boolean isWithFeaturesSatisfied(
      Set<CToolchain.FeatureSet> withFeatureSets, Set<String> enabledFeatureNames) {
    if (withFeatureSets.isEmpty()) {
      return true;
    }
    for (CToolchain.FeatureSet featureSet : withFeatureSets) {
      if (enabledFeatureNames.containsAll(featureSet.getFeatureList())) {
        return true;
      }
    }
    return false;
  }

  /**
   * Groups a set of flags to apply for certain actions.
   */
  @Immutable
  private static class FlagSet implements Serializable {
    private final ImmutableSet<String> actions;
    private final ImmutableSet<String> expandIfAllAvailable;
    private final ImmutableSet<CToolchain.FeatureSet> withFeatureSets;
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
      this.withFeatureSets = ImmutableSet.copyOf(flagSet.getWithFeatureList());
      ImmutableList.Builder<FlagGroup> builder = ImmutableList.builder();
      for (CToolchain.FlagGroup flagGroup : flagSet.getFlagGroupList()) {
        builder.add(new FlagGroup(flagGroup));
      }
      this.flagGroups = builder.build();
    }

    /** Adds the flags that apply to the given {@code action} to {@code commandLine}. */
    private void expandCommandLine(
        String action,
        Variables variables,
        Set<String> enabledFeatureNames,
        List<String> commandLine) {
      for (String variable : expandIfAllAvailable) {
        if (!variables.isAvailable(variable)) {
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

    /** Adds the flags that apply to the given {@code action} to {@code commandLine}. */
    private void expandCommandLine(
        String action,
        Variables variables,
        Set<String> enabledFeatureNames,
        List<String> commandLine) {
      for (FlagSet flagSet : flagSets) {
        flagSet.expandCommandLine(action, variables, enabledFeatureNames, commandLine);
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
  static class ActionConfig implements Serializable, CrosstoolSelectable {

    public static final String FLAG_SET_WITH_ACTION_ERROR =
        "action_config %s specifies actions.  An action_config's flag sets automatically apply "
            + "to the configured action.  Thus, you must not specify action lists in an "
            + "action_config's flag set.";

    private final String configName;
    private final String actionName;
    private final List<CToolchain.Tool> tools;
    private final ImmutableList<FlagSet> flagSets;

    private ActionConfig(CToolchain.ActionConfig actionConfig)
        throws InvalidConfigurationException {
      this.configName = actionConfig.getConfigName();
      this.actionName = actionConfig.getActionName();
      this.tools = actionConfig.getToolList();

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
              input -> {
                Collection<String> featureNamesForTool = input.getWithFeature().getFeatureList();
                return enabledFeatureNames.containsAll(featureNamesForTool);
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

    /** Adds the flags that apply to this action to {@code commandLine}. */
    private void expandCommandLine(
        Variables variables, Set<String> enabledFeatureNames, List<String> commandLine) {
      for (FlagSet flagSet : flagSets) {
        flagSet.expandCommandLine(actionName, variables, enabledFeatureNames, commandLine);
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
        throw new ExpansionException(
            String.format(
                "Artifact category %s not recognized", artifactNamePattern.getCategoryName()));
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
        chunk.expand(artifactNameVariables, resultBuilder);
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
  @Immutable
  public static class Variables {
   
    /** An empty variables instance. */
    public static final Variables EMPTY = new Variables.Builder().build();

    /**
     * Variables can be either String values or an arbitrarily deeply nested recursive sequences,
     * which we represent as a tree of {@code VariableValue} nodes. The nodes are {@code Sequence}
     * objects, while the leafs are {@code StringSequence} objects. We do not allow {@code
     * StringValue} objects in the tree, as the object memory overhead is too large when we have
     * millions of values. If we find single element {@code StringSequence} in memory profiles in
     * the future, we can introduce another special case type.
     */
    interface VariableValue {

      /**
       * Return string value of the variable, if the variable type can be converted to string (e.g.
       * StringValue), or throw exception if it cannot (e.g. Sequence).
       *
       * @param variableName name of the variable value at hand, for better exception message.
       */
      String getStringValue(String variableName);

      /**
       * Return Iterable value of the variable, if the variable type can be converted to a Iterable
       * (e.g. Sequence), or throw exception if it cannot (e.g. StringValue).
       *
       * @param variableName name of the variable value at hand, for better exception message.
       */
      Iterable<? extends VariableValue> getSequenceValue(String variableName);

      // TODO(b/32655571): Remove once implicit iteration is not needed
      boolean isSequence();

      /**
       * Return value of the field, if the variable is of struct type or throw exception if it is
       * not or no such field exists.
       *
       * @param variableName name of the variable value at hand, for better exception message.
       */
      VariableValue getFieldValue(String variableName, String field);

      /** Return true if the variable is truthy */
      boolean isTruthy();
    }

    /** Interface for VariableValue builders */
    public interface VariableValueBuilder {
      VariableValue build();
    }

    /** Builder for StructureSequence. */
    public static class StructureSequenceBuilder implements VariableValueBuilder {

      private final ImmutableList.Builder<ImmutableMap<String, VariableValue>> values =
          ImmutableList.builder();

      /** Adds a structure to the sequence. */
      public StructureSequenceBuilder addValue(ImmutableMap<String, VariableValue> value) {
        values.add(value);
        return this;
      }

      /** Returns an immutable structure sequence. */
      @Override
      public StructureSequence build() {
        return new StructureSequence(values.build());
      }
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

      /** Adds a value to the sequence. */
      public SequenceBuilder addValues(ImmutableList<VariableValueBuilder> builders) {
        Preconditions.checkArgument(
            builders != null, "Cannot use null builders as a sequence value");
        for (VariableValueBuilder builder : builders) {
          addValue(builder);
        }
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
      public StructureBuilder addField(String name, int value) {
        fields.put(name, new IntegerValue(value));
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
     * A sequence of structure values. Exists as a memory optimization - a typical build can contain
     * millions of feature values, so getting rid of the overhead of {@code StructureValue} objects
     * significantly reduces memory overhead.
     */
    @Immutable
    public static class LibraryToLinkValue implements VariableValue {

      public static final String OBJECT_FILES_FIELD_NAME = "object_files";
      public static final String NAME_FIELD_NAME = "name";
      public static final String TYPE_FIELD_NAME = "type";
      public static final String IS_WHOLE_ARCHIVE_FIELD_NAME = "is_whole_archive";
      private enum Type {
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
      private final ImmutableList<String> objectFiles;
      private final boolean isWholeArchive;
      private final Type type;

      public static LibraryToLinkValue forDynamicLibrary(String name) {
        return new LibraryToLinkValue(name, null, false, Type.DYNAMIC_LIBRARY);
      }

      public static LibraryToLinkValue forVersionedDynamicLibrary(
          String name) {
        return new LibraryToLinkValue(name, null, false, Type.VERSIONED_DYNAMIC_LIBRARY);
      }

      public static LibraryToLinkValue forInterfaceLibrary(String name) {
        return new LibraryToLinkValue(name, null, false, Type.INTERFACE_LIBRARY);
      }

      public static LibraryToLinkValue forStaticLibrary(String name, boolean isWholeArchive) {
        return new LibraryToLinkValue(name, null, isWholeArchive, Type.STATIC_LIBRARY);
      }

      public static LibraryToLinkValue forObjectFile(String name, boolean isWholeArchive) {
        return new LibraryToLinkValue(name, null, isWholeArchive, Type.OBJECT_FILE);
      }

      public static LibraryToLinkValue forObjectFileGroup(
          ImmutableList<String> objects, boolean isWholeArchive) {
        return new LibraryToLinkValue(null, objects, isWholeArchive, Type.OBJECT_FILE_GROUP);
      }

      private LibraryToLinkValue(
          String name, ImmutableList<String> objectFiles, boolean isWholeArchive, Type type) {
        this.name = name;
        this.objectFiles = objectFiles;
        this.isWholeArchive = isWholeArchive;
        this.type = type;
      }

      @Override
      public Iterable<? extends VariableValue> getSequenceValue(String variableName) {
        throw new ExpansionException(
            String.format(
                "Invalid toolchain configuration: Cannot expand variable '%s': expected sequence, "
                    + "found structure (LibraryToLink)",
                variableName));
      }

      @Override
      public boolean isSequence() {
        return false;
      }

      @Override
      public VariableValue getFieldValue(String variableName, String field) {
        Preconditions.checkNotNull(field);
        if (NAME_FIELD_NAME.equals(field) && !type.equals(Type.OBJECT_FILE_GROUP)) {
          return new StringValue(name);
        } else if (OBJECT_FILES_FIELD_NAME.equals(field) && type.equals(Type.OBJECT_FILE_GROUP)) {
          return new StringSequence(objectFiles);
        } else if (TYPE_FIELD_NAME.equals(field)) {
          return new StringValue(type.name);
        } else if (IS_WHOLE_ARCHIVE_FIELD_NAME.equals(field)) {
          return new IntegerValue(isWholeArchive ? 1 : 0);
        } else {
          return null;
        }
      }

      @Override
      public String getStringValue(String variableName) {
        throw new ExpansionException(
            String.format(
                "Invalid toolchain configuration: Cannot expand variable '%s': expected string, "
                    + "found structure (LibraryToLink)",
                variableName));
      }

      @Override
      public boolean isTruthy() {
        return true;
      }
    }

    /**
     * A sequence of structure values. Exists as a memory optimization - a typical build can contain
     * millions of feature values, so getting rid of the overhead of {@code StructureValue} objects
     * significantly reduces memory overhead.
     */
    @Immutable
    private static final class StructureSequence implements VariableValue {

      private final ImmutableList<ImmutableMap<String, VariableValue>> values;

      private StructureSequence(ImmutableList<ImmutableMap<String, VariableValue>> values) {
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
      public boolean isSequence() {
        return true;
      }

      @Override
      public VariableValue getFieldValue(String variableName, String field) {
        throw new ExpansionException(
            String.format(
                "Invalid toolchain configuration: Cannot expand variable '%s.%s': variable '%s' is "
                    + "sequence, expected structure",
                variableName, field, variableName));
      }

      @Override
      public String getStringValue(String variableName) {
        throw new ExpansionException(
            String.format(
                "Invalid toolchain configuration: Cannot expand variable '%s': expected string, "
                    + "found sequence",
                variableName));
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
    private static final class StringSequence implements VariableValue {

      private final Iterable<String> values;

      public StringSequence(Iterable<String> values) {
        Preconditions.checkNotNull(values, "Cannot create StringSequence from null");
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
      public boolean isSequence() {
        return true;
      }

      @Override
      public VariableValue getFieldValue(String variableName, String field) {
        throw new ExpansionException(
            String.format(
                "Invalid toolchain configuration: Cannot expand variable '%s.%s': variable '%s' is "
                    + "sequence, expected structure",
                variableName, field, variableName));
      }

      @Override
      public String getStringValue(String variableName) {
        throw new ExpansionException(
            String.format(
                "Invalid toolchain configuration: Cannot expand variable '%s': expected string, "
                    + "found sequence",
                variableName));
      }

      @Override
      public boolean isTruthy() {
        return !Iterables.isEmpty(values);
      }
    }

    /** Sequence of arbitrary VariableValue objects. */
    @Immutable
    private static final class Sequence implements VariableValue {

      private final ImmutableList<VariableValue> values;

      public Sequence(ImmutableList<VariableValue> values) {
        this.values = values;
      }

      @Override
      public Iterable<? extends VariableValue> getSequenceValue(String variableName) {
        return values;
      }

      @Override
      public boolean isSequence() {
        return true;
      }

      @Override
      public VariableValue getFieldValue(String variableName, String field) {
        throw new ExpansionException(
            String.format(
                "Invalid toolchain configuration: Cannot expand variable '%s.%s': variable '%s' is "
                    + "sequence, expected structure",
                variableName, field, variableName));
      }

      @Override
      public String getStringValue(String variableName) {
        throw new ExpansionException(
            String.format(
                "Invalid toolchain configuration: Cannot expand variable '%s': expected string, "
                    + "found sequence",
                variableName));
      }

      @Override
      public boolean isTruthy() {
        return values.isEmpty();
      }
    }

    /**
     * Single structure value. Be careful not to create sequences of single structures, as the
     * memory overhead is prohibitively big. Use optimized {@link StructureSequence} instead.
     */
    @Immutable
    private static final class StructureValue implements VariableValue {

      private final ImmutableMap<String, VariableValue> value;

      public StructureValue(ImmutableMap<String, VariableValue> value) {
        this.value = value;
      }

      @Override
      public String getStringValue(String variableName) {
        throw new ExpansionException(
            String.format(
                "Invalid toolchain configuration: Cannot expand variable '%s': expected string, "
                    + "found structure",
                variableName));
      }

      @Override
      public Iterable<? extends VariableValue> getSequenceValue(String variableName) {
        throw new ExpansionException(
            String.format(
                "Invalid toolchain configuration: Cannot expand variable '%s': expected sequence, "
                    + "found structure",
                variableName));
      }

      @Override
      public VariableValue getFieldValue(String variableName, String field) {
        if (value.containsKey(field)) {
          return value.get(field);
        } else {
          return null;
        }
      }

      @Override
      public boolean isSequence() {
        return false;
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
    private static final class StringValue implements VariableValue {

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
      public Iterable<? extends VariableValue> getSequenceValue(String variableName) {
        throw new ExpansionException(
            String.format(
                "Invalid toolchain configuration: Cannot expand variable '%s': expected sequence, "
                    + "found string",
                variableName));
      }

      @Override
      public VariableValue getFieldValue(String variableName, String field) {
        throw new ExpansionException(
            String.format(
                "Invalid toolchain configuration: Cannot expand variable '%s.%s': variable '%s' is "
                    + "string, expected structure",
                variableName, field, variableName));
      }

      @Override
      public boolean isSequence() {
        return false;
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
    static final class IntegerValue implements VariableValue {

      private final int value;

      public IntegerValue(int value) {
        this.value = value;
      }

      @Override
      public String getStringValue(String variableName) {
        return Integer.toString(value);
      }

      @Override
      public Iterable<? extends VariableValue> getSequenceValue(String variableName) {
        throw new ExpansionException(
            String.format(
                "Invalid toolchain configuration: Cannot expand variable '%s': expected sequence, "
                    + "found integer",
                variableName));
      }

      @Override
      public VariableValue getFieldValue(String variableName, String field) {
        throw new ExpansionException(
            String.format(
                "Invalid toolchain configuration: Cannot expand variable '%s.%s': variable '%s' is "
                    + "integer, expected structure",
                variableName, field, variableName));
      }

      @Override
      public boolean isSequence() {
        return false;
      }

      @Override
      public boolean isTruthy() {
        return value != 0;
      }
    }

    /**
     * Builder for {@code Variables}.
     */
    public static class Builder {
      private final Map<String, VariableValue> variablesMap = new LinkedHashMap<>();
      private final Map<String, String> stringVariablesMap = new LinkedHashMap<>();

      public Builder() {};

      public Builder(Variables variables) {
        variablesMap.putAll(variables.variablesMap);
        stringVariablesMap.putAll(variables.stringVariablesMap);
      }

      /** Add an integer variable that expands {@code name} to {@code value}. */
      public Builder addIntegerVariable(String name, int value) {
        variablesMap.put(name, new IntegerValue(value));
        return this;
      }

      /** Add a string variable that expands {@code name} to {@code value}. */
      public Builder addStringVariable(String name, String value) {
        Preconditions.checkArgument(
            !variablesMap.containsKey(name), "Cannot overwrite variable '%s'", name);
        Preconditions.checkArgument(
            !stringVariablesMap.containsKey(name), "Cannot overwrite variable '%s'", name);
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

      /**
       * Add a sequence variable that expands {@code name} to {@code values}.
       *
       * <p>Accepts values as ImmutableSet. As ImmutableList has smaller memory footprint, we copy
       * the values into a new list.
       */
      public Builder addStringSequenceVariable(String name, ImmutableSet<String> values) {
        Preconditions.checkArgument(
            !variablesMap.containsKey(name), "Cannot overwrite variable '%s'", name);
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
        Preconditions.checkArgument(
            !variablesMap.containsKey(name), "Cannot overwrite variable '%s'", name);
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
        Preconditions.checkArgument(
            !variablesMap.containsKey(name), "Cannot overwrite variable '%s'", name);
        variablesMap.put(name, new StringSequence(values));
        return this;
      }

      /**
       * Add a variable built using {@code VariableValueBuilder} api that expands {@code name} to
       * the value returned by the {@code builder}.
       */
      public Builder addCustomBuiltVariable(String name, Variables.VariableValueBuilder builder) {
        Preconditions.checkArgument(
            !variablesMap.containsKey(name), "Cannot overwrite variable '%s'", name);
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
          Preconditions.checkArgument(
              !variablesMap.containsKey(name), "Cannot overwrite variable '%s'", name);
          Preconditions.checkArgument(
              !stringVariablesMap.containsKey(name), "Cannot overwrite variable '%s'", name);
        }
        stringVariablesMap.putAll(variables);
        return this;
      }

      /** Adds all variables to this builder. Note: cannot override already added variables. */
      public Builder addAll(Variables variables) {
        SetView<String> intersection =
            Sets.intersection(variables.variablesMap.keySet(), variablesMap.keySet());
        SetView<String> stringIntersection =
            Sets.intersection(variables.stringVariablesMap.keySet(), stringVariablesMap.keySet());
        Preconditions.checkArgument(
            intersection.isEmpty(), "Cannot overwrite existing variables: %s", intersection);
        Preconditions.checkArgument(
            stringIntersection.isEmpty(),
            "Cannot overwrite existing variables: %s", stringIntersection);
        this.variablesMap.putAll(variables.variablesMap);
        this.stringVariablesMap.putAll(variables.stringVariablesMap);
        return this;
      }

      /**
       * Add all variables to this builder, possibly overriding variables already present in the
       * builder. Use cautiously, prefer {@code addAll} if possible.
       * TODO(b/32893861) Clean 'module_files' to be registered only once and remove this method.
       */
      Builder addAndOverwriteAll(Variables overwrittenVariables) {
        this.variablesMap.putAll(overwrittenVariables.variablesMap);
        this.stringVariablesMap.putAll(overwrittenVariables.stringVariablesMap);
        return this;
      }

      /**
       * @return a new {@Variables} object.
       */
      Variables build() {
        return new Variables(
            ImmutableMap.copyOf(variablesMap), ImmutableMap.copyOf(stringVariablesMap));
      }
    }
    
    /**
     * A group of extra {@code Variable} instances, packaged as logic for adding to a
     * {@code Builder}
     */
    public interface VariablesExtension {
      void addVariables(Builder builder);
    }
    
    private final ImmutableMap<String, VariableValue> variablesMap;
    private final ImmutableMap<String, String> stringVariablesMap;
    private final Variables parent;

    private Variables(
        ImmutableMap<String, VariableValue> variablesMap,
        ImmutableMap<String, String> stringVariablesMap) {
      this.variablesMap = variablesMap;
      this.stringVariablesMap = stringVariablesMap;
      this.parent = null;
    }

    /**
     * Creates a variables instance nested under the @param parent, and binds variable named @param
     * name to @param value
     */
    private Variables(Variables parent, String name, VariableValue value) {
      this.variablesMap = ImmutableMap.of(name, value);
      this.stringVariablesMap = ImmutableMap.of();
      this.parent = parent;
    }

    /**
     * Get a variable value named @param name. Supports accessing fields in structures (e.g.
     * 'libraries_to_link.interface_libraries')
     *
     * @throws ExpansionException when no such variable or no such field are present, or when
     *     accessing a field of non-structured variable
     */
    public VariableValue getVariable(String name) {
      return lookupVariable(name, true);
    }

    /**
     * Lookup a variable named @param name or return a reason why the variable was not found.
     * Supports accessing fields in structures.
     *
     * @return Pair<VariableValue, String> returns either (variable value, null) or (null, string
     *     reason why variable was not found)
     */
    private VariableValue lookupVariable(String name, boolean throwOnMissingVariable) {
      VariableValue nonStructuredVariable = getNonStructuredVariable(name);
      if (nonStructuredVariable != null) {
        return nonStructuredVariable;
      }
      VariableValue structuredVariable = getStructureVariable(name, throwOnMissingVariable);
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

    private VariableValue getNonStructuredVariable(String name) {
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

    private VariableValue getStructureVariable(String name, boolean throwOnMissingVariable) {
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
        variable = variable.getFieldValue(structPath, field);
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
      return getVariable(variableName).getStringValue(variableName);
    }

    public Iterable<? extends VariableValue> getSequenceVariable(String variableName) {
      return getVariable(variableName).getSequenceValue(variableName);
    }

    private String guessIteratedOverVariable(ImmutableSet<String> usedVariables) {
      String sequenceName = null;
      for (String usedVariable : usedVariables) {
        VariableValue variableValue = lookupVariable(usedVariable, false);
        if (variableValue != null && variableValue.isSequence()) {
          if (sequenceName != null) {
            throw new ExpansionException(
                "Invalid toolchain configuration: trying to expand two variable list in one "
                    + "flag group: '"
                    + sequenceName
                    + "' and '"
                    + usedVariable
                    + "'");
          } else {
            sequenceName = usedVariable;
          }
        }
      }
      return sequenceName;
    }

    /** Returns whether {@code variable} is set. */
    boolean isAvailable(String variable) {
      return lookupVariable(variable, false) != null;
    }
  }
  
  /**
   * Captures the set of enabled features and action configs for a rule.
   */
  @Immutable
  public static class FeatureConfiguration {
    private final FeatureSpecification featureSpecification;
    private final ImmutableSet<String> enabledFeatureNames;
    private final Iterable<Feature> enabledFeatures;
    private final ImmutableSet<String> enabledActionConfigActionNames;
    
    private final ImmutableMap<String, ActionConfig> actionConfigByActionName;

    public FeatureConfiguration() {
      this(
          FeatureSpecification.EMPTY,
          ImmutableList.<Feature>of(),
          ImmutableList.<ActionConfig>of(),
          ImmutableMap.<String, ActionConfig>of());
    }

    private FeatureConfiguration(
        FeatureSpecification featureSpecification,
        Iterable<Feature> enabledFeatures,
        Iterable<ActionConfig> enabledActionConfigs,
        ImmutableMap<String, ActionConfig> actionConfigByActionName) {
      this.featureSpecification = featureSpecification;
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

    /** @return true if tool_path in action_config points to a real tool, not a dummy placeholder */
    public boolean hasConfiguredLinkerPathInActionConfig() {
      return isEnabled("has_configured_linker_path");
    }

    /** @return whether an action config for the blaze action with the given name is enabled. */
    boolean actionIsConfigured(String actionName) {
      return enabledActionConfigActionNames.contains(actionName);
    }

    /**
     * @return the command line for the given {@code action}.
     */
    List<String> getCommandLine(String action, Variables variables) {
      List<String> commandLine = new ArrayList<>();
      for (Feature feature : enabledFeatures) {
        feature.expandCommandLine(action, variables, enabledFeatureNames, commandLine);
      }
      
      if (actionIsConfigured(action)) {
        actionConfigByActionName
            .get(action)
            .expandCommandLine(variables, enabledFeatureNames, commandLine);
      }

      return commandLine;
    }

    /** @return the environment variables (key/value pairs) for the given {@code action}. */
    ImmutableMap<String, String> getEnvironmentVariables(String action, Variables variables) {
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

    public FeatureSpecification getFeatureSpecification() {
      return featureSpecification;
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

  private final ImmutableList<String> defaultFeatures;
 
  /**
   * A cache of feature selection results, so we do not recalculate the feature selection for all
   * actions.
   */
  private transient LoadingCache<FeatureSpecification, FeatureConfiguration> configurationCache =
      buildConfigurationCache();

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

    ImmutableList.Builder<String> defaultFeaturesBuilder = ImmutableList.builder();
    for (CToolchain.Feature toolchainFeature : toolchain.getFeatureList()) {
      Feature feature = new Feature(toolchainFeature);
      selectablesBuilder.add(feature);
      selectablesByName.put(feature.getName(), feature);
      if (toolchainFeature.getEnabled()) {
        defaultFeaturesBuilder.add(feature.getName());
      }
    }
    this.defaultFeatures = defaultFeaturesBuilder.build();
    
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
  private LoadingCache<FeatureSpecification, FeatureConfiguration> buildConfigurationCache() {
    return CacheBuilder.newBuilder()
        // TODO(klimek): Benchmark and tweak once we support a larger configuration.
        .maximumSize(10000)
        .build(
            new CacheLoader<FeatureSpecification, FeatureConfiguration>() {
              @Override
              public FeatureConfiguration load(FeatureSpecification featureSpecification) {
                return computeFeatureConfiguration(featureSpecification);
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
  public FeatureConfiguration getFeatureConfiguration(FeatureSpecification featureSpecification) {
    return configurationCache.getUnchecked(featureSpecification);
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
  public FeatureConfiguration computeFeatureConfiguration(
      FeatureSpecification featureSpecification) {
    // Command line flags will be output in the order in which they are specified in the toolchain
    // configuration.
    return new FeatureSelection(featureSpecification).run();
  }

  /** Returns the list of features that specify themselves as enabled by default. */
  public ImmutableList<String> getDefaultFeatures() {
    return defaultFeatures;
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
   * Returns the artifact selected by the toolchain for the given action type and action category,
   * or null if the category is not supported by the action config.
   */
  String getArtifactNameForCategory(ArtifactCategory artifactCategory, String outputName)
      throws ExpansionException {
    PathFragment output = PathFragment.create(outputName);

    ArtifactNamePattern patternForCategory = null;
    for (ArtifactNamePattern artifactNamePattern : artifactNamePatterns) {
      if (artifactNamePattern.getArtifactCategory() == artifactCategory) {
        patternForCategory = artifactNamePattern;
      }
    }
    if (patternForCategory == null) {
      throw new ExpansionException(
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

  /**
   * Implements the feature selection algorithm.
   *
   * <p>Feature selection is done by first enabling all features reachable by an 'implies' edge, and
   * then iteratively pruning features that have unmet requirements.
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
    private final FeatureSpecification featureSpecification;

    private FeatureSelection(FeatureSpecification featureSpecification) {
      this.featureSpecification = featureSpecification;
      ImmutableSet.Builder<CrosstoolSelectable> builder = ImmutableSet.builder();
      for (String name : featureSpecification.getRequestedFeatures()) {
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

      for (String provided : provides.keys()) {
        List<String> conflicts = new ArrayList<>();
        for (CrosstoolSelectable selectableProvidingString : provides.get(provided)) {
          if (enabledActivatablesInOrder.contains(selectableProvidingString)) {
            conflicts.add(selectableProvidingString.getName());
          }
        }

        if (conflicts.size() > 1) {
          throw new CollidingProvidesException(String.format(COLLIDING_PROVIDES_ERROR,
              provided, Joiner.on(" ").join(conflicts)));
        }
      }

      return new FeatureConfiguration(
          featureSpecification,
          enabledFeaturesInOrder,
          enabledActionConfigsInOrder,
          actionConfigsByActionName);
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
