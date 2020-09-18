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

package com.google.devtools.build.lib.analysis;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.collect.Multimap;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationCollection;
import com.google.devtools.build.lib.analysis.config.ConfigurationResolver;
import com.google.devtools.build.lib.analysis.config.ConfigurationResolver.TopLevelTargetsAndConfigsResult;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.analysis.config.TransitionResolver;
import com.google.devtools.build.lib.analysis.config.transitions.ConfigurationTransition;
import com.google.devtools.build.lib.analysis.config.transitions.NoTransition;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.Info;
import com.google.devtools.build.lib.packages.NativeProvider;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.TriState;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Collection;
import java.util.LinkedHashSet;
import java.util.List;

/**
 * Utility functions for use during analysis.
 */
public final class AnalysisUtils {

  private AnalysisUtils() {
    throw new IllegalStateException(); // utility class
  }

  /**
   * Returns whether link stamping is enabled for a rule.
   *
   * <p>This returns false for unstampable rule classes and for rules used to build tools. Otherwise
   * it returns the value of the stamp attribute, or of the stamp option if the attribute value is
   * -1.
   */
  public static boolean isStampingEnabled(RuleContext ruleContext, BuildConfiguration config) {
    if (config.isToolConfiguration()
        || !ruleContext.attributes().has("stamp", BuildType.TRISTATE)) {
      return false;
    }
    TriState stamp = ruleContext.attributes().get("stamp", BuildType.TRISTATE);
    return stamp == TriState.YES || (stamp == TriState.AUTO && config.stampBinaries());
  }

  public static boolean isStampingEnabled(RuleContext ruleContext) {
    return isStampingEnabled(ruleContext, ruleContext.getConfiguration());
  }

  // TODO(bazel-team): These need Iterable<? extends TransitiveInfoCollection> because they need to
  // be called with Iterable<ConfiguredTarget>. Once the configured target lockdown is complete, we
  // can eliminate the "extends" clauses.
  /**
   * Returns the list of providers of the specified type from a set of transitive info collections.
   */
  public static <C extends TransitiveInfoProvider> List<C> getProviders(
      Iterable<? extends TransitiveInfoCollection> prerequisites, Class<C> provider) {
    ImmutableList.Builder<C> result = ImmutableList.builder();
    for (TransitiveInfoCollection prerequisite : prerequisites) {
      C prerequisiteProvider =  prerequisite.getProvider(provider);
      if (prerequisiteProvider != null) {
        result.add(prerequisiteProvider);
      }
    }
    return result.build();
  }

  /**
   * Returns the list of declared providers (native and Starlark) of the specified Starlark key from
   * a set of transitive info collections.
   */
  public static <T extends Info> List<T> getProviders(
      Iterable<? extends TransitiveInfoCollection> prerequisites,
      final NativeProvider<T> starlarkKey) {
    ImmutableList.Builder<T> result = ImmutableList.builder();
    for (TransitiveInfoCollection prerequisite : prerequisites) {
      T prerequisiteProvider = prerequisite.get(starlarkKey);
      if (prerequisiteProvider != null) {
        result.add(prerequisiteProvider);
      }
    }
    return result.build();
  }

  /**
   * Returns the list of declared providers (native and Starlark) of the specified Starlark key from
   * a set of transitive info collections.
   */
  public static <T extends Info> List<T> getProviders(
      Iterable<? extends TransitiveInfoCollection> prerequisites,
      final BuiltinProvider<T> starlarkKey) {
    ImmutableList.Builder<T> result = ImmutableList.builder();
    for (TransitiveInfoCollection prerequisite : prerequisites) {
      T prerequisiteProvider = prerequisite.get(starlarkKey);
      if (prerequisiteProvider != null) {
        result.add(prerequisiteProvider);
      }
    }
    return result.build();
  }

  /**
   * Returns the iterable of collections that have the specified provider.
   */
  public static <S extends TransitiveInfoCollection, C extends TransitiveInfoProvider> Iterable<S>
      filterByProvider(Iterable<S> prerequisites, final Class<C> provider) {
    return Iterables.filter(prerequisites, target -> target.getProvider(provider) != null);
  }

  /** Returns the iterable of collections that have the specified provider. */
  public static <S extends TransitiveInfoCollection, C extends Info> Iterable<S> filterByProvider(
      Iterable<S> prerequisites, final NativeProvider<C> provider) {
    return Iterables.filter(prerequisites, target -> target.get(provider) != null);
  }

  /** Returns the iterable of collections that have the specified provider. */
  public static <S extends TransitiveInfoCollection, C extends Info> Iterable<S> filterByProvider(
      Iterable<S> prerequisites, final BuiltinProvider<C> provider) {
    return Iterables.filter(prerequisites, target -> target.get(provider) != null);
  }

  /**
   * Returns the path of the associated manifest file for the path of a Fileset. Works for both
   * exec paths and root relative paths.
   */
  public static PathFragment getManifestPathFromFilesetPath(PathFragment filesetDir) {
    PathFragment manifestDir = filesetDir.replaceName("_" + filesetDir.getBaseName());
    PathFragment outputManifestFrag = manifestDir.getRelative("MANIFEST");
    return outputManifestFrag;
  }

  /**
   * Returns a path fragment qualified by the rule name and unique fragment to
   * disambiguate artifacts produced from the source file appearing in
   * multiple rules.
   *
   * <p>For example "//pkg:target" -> "pkg/&lt;fragment&gt;/target.
   */
  public static PathFragment getUniqueDirectory(Label label, PathFragment fragment) {
    return label
        .getPackageIdentifier()
        .getPackagePath()
        .getRelative(fragment)
        .getRelative(label.getName());
  }

  /**
   * Checks that the given provider class either refers to an interface or to a value class.
   */
  public static <T extends TransitiveInfoProvider> void checkProvider(Class<T> clazz) {
    // Write this check in terms of getName() rather than getSimpleName(); the latter is expensive.
    if (!clazz.isInterface() && clazz.getName().contains(".AutoValue_")) {
      // We must have a superclass due to the generic bound above.
      throw new IllegalArgumentException(
          clazz + " is generated by @AutoValue; use " + clazz.getSuperclass() + " instead");
    }
  }

  /**
   * Given a set of *top-level* targets and a configuration collection, evaluate top level
   * transitions, resolve configurations and return the appropriate <Target, Configuration> pair for
   * each target.
   *
   * <p>Preserves the original input ordering.
   */
  // Keep this in sync with PrepareAnalysisPhaseFunction.
  public static TopLevelTargetsAndConfigsResult getTargetsWithConfigs(
      BuildConfigurationCollection configurations,
      Collection<Target> targets,
      ExtendedEventHandler eventHandler,
      ConfiguredRuleClassProvider ruleClassProvider,
      ConfigurationsCollector configurationsCollector)
      throws InvalidConfigurationException {
    // We use a hash set here to remove duplicate nodes; this can happen for input files and package
    // groups.
    LinkedHashSet<TargetAndConfiguration> nodes = new LinkedHashSet<>(targets.size());
    for (BuildConfiguration config : configurations.getTargetConfigurations()) {
      for (Target target : targets) {
        nodes.add(new TargetAndConfiguration(target, config));
      }
    }

    // We'll get the configs from ConfigurationsCollector#getConfigurations, which gets
    // configurations for deps including transitions.
    Multimap<BuildConfiguration, DependencyKey> asDeps = targetsToDeps(nodes, ruleClassProvider);

    return ConfigurationResolver.getConfigurationsFromExecutor(
        nodes, asDeps, eventHandler, configurationsCollector);
  }

  @VisibleForTesting
  public static Multimap<BuildConfiguration, DependencyKey> targetsToDeps(
      Collection<TargetAndConfiguration> nodes, ConfiguredRuleClassProvider ruleClassProvider) {
    Multimap<BuildConfiguration, DependencyKey> asDeps = ArrayListMultimap.create();
    for (TargetAndConfiguration targetAndConfig : nodes) {
      ConfigurationTransition transition =
          TransitionResolver.evaluateTransition(
              targetAndConfig.getConfiguration(),
              NoTransition.INSTANCE,
              targetAndConfig.getTarget(),
              ruleClassProvider.getTrimmingTransitionFactory());
      if (targetAndConfig.getConfiguration() != null) {
        // TODO(bazel-team): support top-level aspects
        asDeps.put(
            targetAndConfig.getConfiguration(),
            DependencyKey.builder()
                .setLabel(targetAndConfig.getLabel())
                .setTransition(transition)
                .build());
      }
    }
    return asDeps;
  }
}
