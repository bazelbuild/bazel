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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableListMultimap;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ListMultimap;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.Attribute.SplitTransition;
import com.google.devtools.build.lib.packages.Attribute.Transition;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.util.Preconditions;

import java.io.PrintStream;
import java.io.Serializable;
import java.util.Collection;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
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
  private final BuildConfiguration hostConfiguration;

  public BuildConfigurationCollection(List<BuildConfiguration> targetConfigurations,
      BuildConfiguration hostConfiguration)
      throws InvalidConfigurationException {
    this.targetConfigurations = ImmutableList.copyOf(targetConfigurations);
    this.hostConfiguration = hostConfiguration;

    // Except for the host configuration (which may be identical across target configs), the other
    // configurations must all have different cache keys or we will end up with problems.
    HashMap<String, BuildConfiguration> cacheKeyConflictDetector = new HashMap<>();
    for (BuildConfiguration config : getAllConfigurations()) {
      String cacheKey = config.checksum();
      if (cacheKeyConflictDetector.containsKey(cacheKey)) {
        throw new InvalidConfigurationException("Conflicting configurations: " + config + " & "
            + cacheKeyConflictDetector.get(cacheKey));
      }
      cacheKeyConflictDetector.put(cacheKey, config);
    }
  }

  public static BuildConfiguration configureTopLevelTarget(BuildConfiguration topLevelConfiguration,
      Target toTarget) {
    if (!toTarget.isConfigurable()) {
      return null;
    }
    return topLevelConfiguration.getTransitions().toplevelConfigurationHook(toTarget);
  }

  public ImmutableList<BuildConfiguration> getTargetConfigurations() {
    return targetConfigurations;
  }

  /**
   * Returns the host configuration for this collection.
   *
   * <p>Don't use this method. It's limited in that it assumes a single host configuration for
   * the entire collection. This may not be true in the future and more flexible interfaces based
   * on dynamic configurations will likely supplant this interface anyway. Its main utility is
   * to keep Bazel working while dynamic configuration progress is under way.
   */
  public BuildConfiguration getHostConfiguration() {
    return hostConfiguration;
  }

  /**
   * Returns all configurations that can be reached from the target configuration through any kind
   * of configuration transition.
   */
  public Collection<BuildConfiguration> getAllConfigurations() {
    Set<BuildConfiguration> result = new LinkedHashSet<>();
    for (BuildConfiguration config : targetConfigurations) {
      result.addAll(config.getAllReachableConfigurations());
    }
    return result;
  }

  /**
   * Returns whether this build uses dynamic configurations.
   */
  public boolean useDynamicConfigurations() {
    return getTargetConfigurations().get(0).useDynamicConfigurations();
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
   * Prints the configuration graph in dot format to the given print stream. This is only intended
   * for debugging.
   */
  public void dumpAsDotGraph(PrintStream out) {
    out.println("digraph g {");
    out.println("  ratio = 0.3;");
    for (BuildConfiguration config : getAllConfigurations()) {
      String from = config.checksum();
      for (Map.Entry<? extends Transition, ConfigurationHolder> entry :
          config.getTransitions().getTransitionTable().entrySet()) {
        BuildConfiguration toConfig = entry.getValue().getConfiguration();
        if (toConfig == config) {
          continue;
        }
        String to = toConfig == null ? "ERROR" : toConfig.checksum();
        out.println("  \"" + from + "\" -> \"" + to + "\" [label=\"" + entry.getKey() + "\"]");
      }
    }
    out.println("}");
  }

  /**
   * The outgoing transitions for a build configuration.
   */
  public abstract static class Transitions implements Serializable {
    protected final BuildConfiguration configuration;

    /**
     * Look up table for the configuration transitions, i.e., HOST, DATA, etc.
     */
    private final Map<? extends Transition, ConfigurationHolder> transitionTable;

    // TODO(bazel-team): Consider merging transitionTable into this.
    private final ListMultimap<? super SplitTransition<?>, BuildConfiguration> splitTransitionTable;

    public Transitions(BuildConfiguration configuration,
        Map<? extends Transition, ConfigurationHolder> transitionTable,
        ListMultimap<? extends SplitTransition<?>, BuildConfiguration> splitTransitionTable) {
      this.configuration = configuration;
      this.transitionTable = ImmutableMap.copyOf(transitionTable);
      // Do not remove <SplitTransition<?>, BuildConfiguration>:
      // workaround for Java 7 type inference.
      this.splitTransitionTable =
          ImmutableListMultimap.<SplitTransition<?>, BuildConfiguration>copyOf(
              splitTransitionTable);
    }

    public Map<? extends Transition, ConfigurationHolder> getTransitionTable() {
      return transitionTable;
    }

    public List<BuildConfiguration> getSplitConfigurationsNoSelf(SplitTransition<?> transition) {
      if (splitTransitionTable.containsKey(transition)) {
        return splitTransitionTable.get(transition);
      } else {
        return ImmutableList.of();
      }
    }

    public List<BuildConfiguration> getSplitConfigurations(SplitTransition<?> transition) {
      if (splitTransitionTable.containsKey(transition)) {
        return splitTransitionTable.get(transition);
      } else {
        Preconditions.checkState(transition.defaultsToSelf());
        return ImmutableList.of(configuration);
      }
    }

    /**
     * Adds all configurations that are directly reachable from this configuration through
     * any kind of configuration transition.
     */
    public void addDirectlyReachableConfigurations(Collection<BuildConfiguration> queue) {
      for (ConfigurationHolder holder : transitionTable.values()) {
        if (holder.configuration != null) {
          queue.add(holder.configuration);
        }
      }
      queue.addAll(splitTransitionTable.values());
    }

    /**
     * Artifacts need an owner in Skyframe. By default it's the same configuration as what
     * the configured target has, but it can be overridden if necessary.
     *
     * @return the artifact owner configuration
     */
    public BuildConfiguration getArtifactOwnerConfiguration() {
      return configuration;
    }

    /**
     * Returns the new configuration after traversing a dependency edge with a
     * given configuration transition.
     *
     * <p>Only used for static configuration builds.
     *
     * @param configurationTransition the configuration transition
     * @return the new configuration
     */
    public BuildConfiguration getStaticConfiguration(Transition configurationTransition) {
      Preconditions.checkState(!configuration.useDynamicConfigurations());
      ConfigurationHolder holder = transitionTable.get(configurationTransition);
      if (holder == null && configurationTransition.defaultsToSelf()) {
        return configuration;
      }
      return holder.configuration;
    }

    /**
     * Translates a static configuration {@link Transition} reference into the corresponding
     * dynamic configuration transition.
     *
     * <p>The difference is that with static configurations, the transition just models a desired
     * type of transition that subsequently gets linked to a pre-built global configuration through
     * custom logic in {@link BuildConfigurationCollection.Transitions} and
     * {@link com.google.devtools.build.lib.analysis.ConfigurationCollectionFactory}.
     *
     * <p>With dynamic configurations, the transition directly embeds the semantics, e.g.
     * it includes not just a name but also the logic of how it should transform its input
     * configuration.
     *
     * <p>This is a connecting method meant to keep the two models in sync for the current time
     * in which they must co-exist. Once dynamic configurations are production-ready, we'll remove
     * the static configuration code entirely.
     */
    protected Transition getDynamicTransition(Transition transition) {
      Preconditions.checkState(configuration.useDynamicConfigurations());
      if (transition == Attribute.ConfigurationTransition.NONE) {
        return transition;
      } else if (transition == Attribute.ConfigurationTransition.NULL) {
        return transition;
      } else if (transition == Attribute.ConfigurationTransition.HOST) {
        return HostTransition.INSTANCE;
      } else {
        throw new UnsupportedOperationException("No dynamic mapping for " + transition.toString());
      }
    }

    /**
     * Arbitrary configuration transitions can be implemented by overriding this hook.
     */
    @SuppressWarnings("unused")
    public void configurationHook(Rule fromTarget, Attribute attribute, Target toTarget,
        BuildConfiguration.TransitionApplier transitionApplier) {
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
  public static final class ConfigurationHolder implements Serializable {
    private final BuildConfiguration configuration;

    public ConfigurationHolder(BuildConfiguration configuration) {
      this.configuration = configuration;
    }

    public BuildConfiguration getConfiguration() {
      return configuration;
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
