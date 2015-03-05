// Copyright 2015 Google Inc. All rights reserved.
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
   * A piece of a single flag.
   * 
   * <p>A single flag can contain a combination of text and variables (for example
   * "-f %{var1}/%{var2}"). We split the flag into chunks, where each chunk represents either a
   * text snippet, or a variable that is to be replaced.
   */
  interface FlagChunk {
    
    /**
     * Expands this chunk.
     * 
     * @param variables variable names mapped to their values for a single flag expansion.
     * @param flag the flag content to append to.
     */
    void expand(Map<String, String> variables, StringBuilder flag);
  }
  
  /**
   * A plain text chunk of a flag.
   */
  @Immutable
  private static class StringChunk implements FlagChunk, Serializable {
    private final String text;
    
    private StringChunk(String text) {
      this.text = text;
    }
    
    @Override
    public void expand(Map<String, String> variables, StringBuilder flag) {
      flag.append(text);
    }
  }
  
  /**
   * A chunk of a flag into which a variable should be expanded.
   */
  @Immutable
  private static class VariableChunk implements FlagChunk, Serializable {
    private final String variableName;
    
    private VariableChunk(String variableName) {
      this.variableName = variableName;
    }
    
    @Override
    public void expand(Map<String, String> variables, StringBuilder flag) {
      String value = variables.get(variableName);
      if (value == null) {
        // We check all variables in FlagGroup.expandCommandLine, so if we arrive here with a
        // null value, the variable map originally handed to the feature selection must have
        // contained an explicit null value.
        throw new ExpansionException("Internal blaze error: build variable was set to 'null'.");
      }
      flag.append(variables.get(variableName));
    }
  }
  
  /**
   * Parser for toolchain flags.
   * 
   * <p>A flag contains a snippet of text supporting variable expansion. For example, a flag value
   * "-f %{var1}/%{var2}" will expand the values of the variables "var1" and "var2" in the
   * corresponding places in the string.
   * 
   * <p>The {@code FlagParser} takes a flag string and parses it into a list of {@code FlagChunk}
   * objects, where each chunk represents either a snippet of text or a variable to be expanded. In
   * the above example, the resulting chunks would be ["-f ", var1, "/", var2].
   * 
   * <p>In addition to the list of chunks, the {@code FlagParser} also provides the set of variables
   * necessary for the expansion of this flag via {@code getUsedVariables}.
   * 
   * <p>To get a literal percent character, "%%" can be used in the flag text.
   */
  private static class FlagParser {
    
    /**
     * The given flag value.
     */
    private final String value;
    
    /**
     * The current position in {@value} during parsing.
     */
    private int current = 0;
    
    private final ImmutableList.Builder<FlagChunk> chunks = ImmutableList.builder();
    private final ImmutableSet.Builder<String> usedVariables = ImmutableSet.builder();
    
    private FlagParser(String value) throws InvalidConfigurationException {
      this.value = value;
      parse();
    }
    
    /**
     * @return the parsed chunks for this flag.
     */
    private ImmutableList<FlagChunk> getChunks() {
      return chunks.build();
    }
    
    /**
     * @return all variable names needed to expand this flag.
     */
    private ImmutableSet<String> getUsedVariables() {
      return usedVariables.build();
    }
    
    /**
     * Parses the flag.
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
      // Note that for flags like "abc%%def" this will lead to two string chunks, the first
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
      chunks.add(new StringChunk(text));
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
     * the current position in the flag.
     */
    private void abort(String error) throws InvalidConfigurationException {
      throw new InvalidConfigurationException("Invalid toolchain configuration: " + error
          + " at position " + current + " while parsing a flag containing '" + value + "'");
    }
  }
  
  /**
   * A single flag to be expanded under a set of variables.
   * 
   * <p>TODO(bazel-team): Consider specializing Flag for the simple case that a flag is just a bit
   * of text.
   */
  @Immutable
  private static class Flag implements Serializable {
    private final ImmutableList<FlagChunk> chunks;
    
    private Flag(ImmutableList<FlagChunk> chunks) {
      this.chunks = chunks;
    }
    
    /**
     * Expand this flag into a single new entry in {@code commandLine}.
     */
    private void expandCommandLine(Map<String, String> variables, List<String> commandLine) {
      StringBuilder flag = new StringBuilder();
      for (FlagChunk chunk : chunks) {
        chunk.expand(variables, flag);
      }
      commandLine.add(flag.toString());      
    }
  }
  
  /**
   * A group of flags.
   */
  @Immutable
  private static class FlagGroup implements Serializable {
    private final ImmutableList<Flag> flags;
    private final ImmutableSet<String> usedVariables;
    
    private FlagGroup(CToolchain.FlagGroup flagGroup) throws InvalidConfigurationException {
      ImmutableList.Builder<Flag> flags = ImmutableList.builder();
      ImmutableSet.Builder<String> usedVariables = ImmutableSet.builder();
      for (String flag : flagGroup.getFlagList()) {
        FlagParser parser = new FlagParser(flag);        
        flags.add(new Flag(parser.getChunks()));
        usedVariables.addAll(parser.getUsedVariables());
      }
      this.flags = flags.build();
      this.usedVariables = usedVariables.build();
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
      variables.forEachExpansion(new Variables.ExpansionConsumer() {
        @Override
        public Set<String> getUsedVariables() {
          return usedVariables;
        }

        @Override
        public void expand(Map<String, String> variables) {
          for (Flag flag : flags) {
            flag.expandCommandLine(variables, commandLine);
          }
        }
      });
    }
  }
  
  /**
   * Groups a set of flags to apply for certain actions.
   */
  @Immutable
  private static class FlagSet implements Serializable {
    private final ImmutableSet<String> actions;
    private final ImmutableList<FlagGroup> flagGroups;
    
    private FlagSet(CToolchain.FlagSet flagSet) throws InvalidConfigurationException {
      this.actions = ImmutableSet.copyOf(flagSet.getActionList());
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
      if (!actions.contains(action)) {
        return;
      }
      for (FlagGroup flagGroup : flagGroups) {
        flagGroup.expandCommandLine(variables, commandLine);
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
    
    private Feature(CToolchain.Feature feature) throws InvalidConfigurationException {
      this.name = feature.getName();
      ImmutableList.Builder<FlagSet> builder = ImmutableList.builder();
      for (CToolchain.FlagSet flagSet : feature.getFlagSetList()) {
        builder.add(new FlagSet(flagSet));
      }
      this.flagSets = builder.build();
    }

    /**
     * @return the features's name.
     */
    private String getName() {
      return name;
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
     * Builder for {@code Variables}.
     */
    public static class Builder {
      private final ImmutableMap.Builder<String, String> variables = ImmutableMap.builder();
      private final ImmutableMap.Builder<String, ImmutableList<String>> sequenceVariables =
          ImmutableMap.builder();
      
      /**
       * Add a variable that expands {@code name} to {@code value}.
       */
      public Builder addVariable(String name, String value) {
        variables.put(name, value);
        return this;
      }
      
      /**
       * Add a variable that expands a flag group containing a reference to {@code name} for each
       * entry in {@code value}. 
       */
      public Builder addSequenceVariable(String name, Collection<String> value) {
        sequenceVariables.put(name, ImmutableList.copyOf(value));
        return this;
      }
      
      /**
       * @return a new {@Variables} object.
       */
      public Variables build() {
        return new Variables(variables.build(), sequenceVariables.build());
      }
    }
    
    /**
     * An {@code ExpansionConsumer} is a callback to be called for each expansion of a variable
     * configuration over its set of used variables. 
     */
    private interface ExpansionConsumer {
      
      /**
       * @return the used variables to be considered for the expansion.
       */
      Set<String> getUsedVariables();
      
      /**
       * Called either once if there are only normal variables in the used variables set, or
       * for each entry in the sequence variable in the used variables set.
       */
      void expand(Map<String, String> variables);
    }
    
    private final ImmutableMap<String, String> variables;
    private final ImmutableMap<String, ImmutableList<String>> sequenceVariables;

    private Variables(ImmutableMap<String, String> variables,
        ImmutableMap<String, ImmutableList<String>> sequenceVariables) {
      this.variables = variables;
      this.sequenceVariables = sequenceVariables;
    }

    /**
     * Calls {@code expand} on the {@code consumer} for each expansion of the {@code consumer}'s
     * used variable set.
     * 
     * <p>The {@code consumer}'s used variable set must contain at most one variable of sequence
     * type; additionally, all of the used variables must be available in the current variable
     * configuration. If any of the preconditions are violated, throws an
     * {@code ExpansionException}.
     */
    void forEachExpansion(ExpansionConsumer consumer) {
      Map<String, String> variableView = new HashMap<>();
      String sequenceName = null; 
      for (String name : consumer.getUsedVariables()) {
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
          variableView.put(name, variables.get(name));
        } else {
          throw new ExpansionException("Invalid toolchain configuration: unknown variable '" + name
              + "' can not be expanded.");
        }
      }
      if (sequenceName != null) {
        for (String value : sequenceVariables.get(sequenceName)) {
          variableView.put(sequenceName, value);
          consumer.expand(variableView);
        }
      } else {
        consumer.expand(variableView);
      }      
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
  CcToolchainFeatures(CToolchain toolchain) throws InvalidConfigurationException {
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
  FeatureConfiguration getFeatureConfiguration(String... requestedFeatures) {
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
