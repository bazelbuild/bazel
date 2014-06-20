// Copyright 2014 Google Inc. All rights reserved.
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

package com.google.devtools.build.lib.view.config;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.Attribute.ConfigurationTransition;
import com.google.devtools.build.lib.packages.Attribute.Configurator;
import com.google.devtools.build.lib.packages.InputFile;
import com.google.devtools.build.lib.packages.PackageGroup;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.Target;

import java.util.Collection;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Queue;
import java.util.Set;

/**
 * The primary container for all main {@link BuildConfiguration} instances,
 * currently "target", "data", and "host".
 *
 * <p>The target configuration is used for all targets specified on the command
 * line. Data dependencies of targets in the target configuration use the data
 * configuration instead.
 *
 * <p>The host configuration is used for tools that are executed during the
 * build, e. g, compilers.
 *
 * <p>The "related" configurations are also contained in this class.
 */
@ThreadSafe
public final class BuildConfigurationCollection {
  private final ImmutableList<BuildConfiguration> targetConfigurations;

  public BuildConfigurationCollection(BuildConfiguration... targetConfiguration)
      throws InvalidConfigurationException {
    this(ImmutableList.copyOf(targetConfiguration));
  }

  public BuildConfigurationCollection(List<BuildConfiguration> targetConfigurations)
      throws InvalidConfigurationException {
    this.targetConfigurations = ImmutableList.copyOf(targetConfigurations);

    // Except for the host configuration (which may be idential across target configs), the other
    // configurations must all have different cache keys or we will end up with problems.
    HashMap<String, BuildConfiguration> cacheKeyConflictDetector = new HashMap<>();
    for (BuildConfiguration target : targetConfigurations) {
      for (BuildConfiguration config : getAllReachableConfigurations(target)) {
        if (config.isHostConfiguration()) {
          continue;
        }
        if (cacheKeyConflictDetector.containsKey(config.cacheKey())) {
          throw new InvalidConfigurationException("Conflicting configurations: " + config + " & "
              + cacheKeyConflictDetector.get(config.cacheKey()));
        }
        cacheKeyConflictDetector.put(config.cacheKey(), config);
      }
    }
  }

  /**
   * Creates an empty configuration collection which will return null for everything.
   */
  public BuildConfigurationCollection() {
    this.targetConfigurations = ImmutableList.of();
  }

  /**
   * Calculates the configuration of a direct dependency. If a rule in some BUILD file refers
   * to a target (like another rule or a source file) using a label attribute, that target needs
   * to have a configuration, too. This method figures out the proper configuration for the
   * dependency.
   *
   * @param fromRule the rule that's depending on some target
   * @param fromConfiguration the configuration of the depending rule
   * @param attribute the attribute using which the rule depends on that target (eg. "srcs")
   * @param toTarget the target that's dependeded on
   * @return the configuration that should be associated to {@code toTarget}
   */
  public static BuildConfiguration configureTarget(Rule fromRule,
      BuildConfiguration fromConfiguration, Attribute attribute, Target toTarget) {
    // Fantastic configurations and where to find them:

    // I. Input files and package groups have no configurations.
    // We don't want to duplicate them. Also, if we had partial analysis caching, we wouldn't want
    // to reload them because of configuration changes.
    if (toTarget instanceof InputFile || toTarget instanceof PackageGroup) {
      return null;
    }

    // II. Host configurations never switch to another.
    // All prerequisites of host targets have the same host configuration.
    if (fromConfiguration.isHostConfiguration()) {
      return fromConfiguration;
    }

    // III. Attributes determine configurations.
    // The configuration of a prerequisite is determined by the attribute.
    // TODO(bazel-team): Right now we have two mechanisms for this: see
    // Attribute.ConfigurationTransition and Attribute.Configurator. The plan is to get rid the
    // first one.
    @SuppressWarnings("unchecked")
    Configurator<BuildConfiguration, Rule> configurator =
        (Configurator<BuildConfiguration, Rule>) attribute.getConfigurator();
    // If there's a configurator callback, use that. Otherwise, fall back to the legacy approach.
    BuildConfiguration toConfiguration = (configurator != null)
        ? configurator.apply(fromRule, fromConfiguration, attribute, toTarget)
        : fromConfiguration.getConfiguration(attribute.getConfigurationTransition());

    // IV. Allow the transition object to perform an arbitrary switch.
    // Blaze modules can inject configuration transition logic by extending
    // PerConfigurationTransitions class.
    toConfiguration = fromConfiguration.getTransitions().configurationHook(
        fromRule, attribute, toTarget, toConfiguration);

    // V. Allow rule classes to override their own configurations
    Rule associatedRule = toTarget.getAssociatedRule();
    if (associatedRule != null) {
      @SuppressWarnings("unchecked")
      RuleClass.Configurator<BuildConfiguration, Rule> func =
          (RuleClass.Configurator<BuildConfiguration, Rule>)
          associatedRule.getRuleClassObject().getConfigurator();
      toConfiguration = func.apply(associatedRule, toConfiguration);
    }

    return toConfiguration;
  }

  public static BuildConfiguration configureTopLevelTarget(BuildConfiguration topLevelConfiguration,
      Target toTarget) {
    if (toTarget instanceof InputFile || toTarget instanceof PackageGroup) {
      return null;
    }
    return topLevelConfiguration.getTransitions().toplevelConfigurationHook(toTarget);
  }

  public ImmutableList<BuildConfiguration> getTargetConfigurations() {
    return targetConfigurations;
  }

  /**
   * Returns all configurations that can be reached from the target configuration through any kind
   * of configuration transition.
   */
  public Collection<BuildConfiguration> getAllConfigurations() {
    Set<BuildConfiguration> result = new LinkedHashSet<>();
    for (BuildConfiguration config : targetConfigurations) {
      result.addAll(getAllReachableConfigurations(config));
    }
    return result;
  }

  /**
   * Returns all configurations that can be reached from the given configuration through any kind
   * of configuration transition.
   */
  private Collection<BuildConfiguration> getAllReachableConfigurations(
      BuildConfiguration startConfig) {
    Set<BuildConfiguration> result = new LinkedHashSet<>();
    Queue<BuildConfiguration> queue = new LinkedList<>();
    queue.add(startConfig);
    while (!queue.isEmpty()) {
      BuildConfiguration config = queue.remove();
      if (!result.add(config)) {
        continue;
      }
      config.getTransitions().addDirectlyReachableConfigurations(queue);
    }
    return result;
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (!(obj instanceof BuildConfigurationCollection)) {
      return false;
    }
    BuildConfigurationCollection that = (BuildConfigurationCollection) obj;
    return this.targetConfigurations.equals(that.targetConfigurations);
  }

  @Override
  public int hashCode() {
    return targetConfigurations.hashCode();
  }

  /**
   * The outgoing transitions for a build configuration.
   */
  public static class Transitions {
    protected final BuildConfiguration configuration;

    /**
     * Cross-references back to its parent.
     */
    private final BuildConfiguration parent;

    /**
     * Look up table for the configuration transitions, i.e., HOST, DATA, etc.
     */
    private final Map<ConfigurationTransition, ConfigurationHolder> configurationTransitions;

    public Transitions(BuildConfiguration configuration,
        BuildConfiguration parent,
        Map<ConfigurationTransition, ConfigurationHolder> transitionTable) {
      this.configuration = configuration;
      this.parent = parent;
      this.configurationTransitions = ImmutableMap.copyOf(transitionTable);
    }

    /**
     * Adds all configurations that are directly reachable from this configuration through
     * any kind of configuration transition.
     */
    public void addDirectlyReachableConfigurations(Collection<BuildConfiguration> queue) {
      queue.add(parent);
      for (ConfigurationHolder holder : configurationTransitions.values()) {
        if (holder.configuration != null) {
          queue.add(holder.configuration);
        }
      }
    }

    /**
     * Returns the parent configuration of a related configuration. For example
     * the parent configuration of the target Java configuration is the main
     * target configuration. The parent configuration of a main configuration
     * (target, data, host) is the main configuration itself.
     *
     * <p>If two related configurations of two different main configurations are
     * indistinguishable, an arbitrary potential parent configuration is returned.
     *
     * @return the parent configuration or {@code null} if none can be found
     */
    public BuildConfiguration getParentConfiguration() {
      return parent;
    }

    /**
     * Returns the new configuration after traversing a dependency edge with a
     * given configuration transition.
     *
     * @param configurationTransition the configuration transition
     * @return the new configuration
     */
    public BuildConfiguration getConfiguration(ConfigurationTransition configurationTransition) {
      return configurationTransitions.get(configurationTransition).configuration;
    }

    /**
     * Arbitrary configuration transitions can be implemented by overriding this hook.
     */
    @SuppressWarnings("unused")
    public BuildConfiguration configurationHook(Rule fromTarget,
        Attribute attribute, Target toTarget, BuildConfiguration toConfiguration) {
      return toConfiguration;
    }

    /**
     * Associating configurations to top-level targets can be implemented by overriding this hook.
     */
    @SuppressWarnings("unused")
    public BuildConfiguration toplevelConfigurationHook(Target toTarget) {
      return configuration;
    }
  }

  /**
   * A holder class for {@link BuildConfiguration} instances that allows {@code null} values,
   * because none of the Table implementations allow them.
   */
  public static final class ConfigurationHolder {
    private final BuildConfiguration configuration;

    public ConfigurationHolder(BuildConfiguration configuration) {
      this.configuration = configuration;
    }

    @Override
    public int hashCode() {
      return configuration == null ? 0 : configuration.hashCode();
    }

    @Override
    public boolean equals(Object o) {
      if (o == this) {
        return true;
      }
      if (!(o instanceof ConfigurationHolder)) {
        return false;
      }
      return Objects.equals(configuration, ((ConfigurationHolder) o).configuration);
    }
  }
}
