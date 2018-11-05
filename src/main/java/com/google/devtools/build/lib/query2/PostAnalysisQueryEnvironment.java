// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.query2;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.analysis.TargetAndConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.cmdline.TargetPattern;
import com.google.devtools.build.lib.collect.compacthashset.CompactHashSet;
import com.google.devtools.build.lib.concurrent.MultisetSemaphore;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.packages.DependencyFilter;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleTransitionFactory;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.pkgcache.FilteringPolicies;
import com.google.devtools.build.lib.pkgcache.PackageManager;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.pkgcache.TargetPatternEvaluator;
import com.google.devtools.build.lib.query2.engine.KeyExtractor;
import com.google.devtools.build.lib.query2.engine.MinDepthUniquifier;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment;
import com.google.devtools.build.lib.query2.engine.QueryEvalResult;
import com.google.devtools.build.lib.query2.engine.QueryException;
import com.google.devtools.build.lib.query2.engine.QueryExpression;
import com.google.devtools.build.lib.query2.engine.QueryExpressionContext;
import com.google.devtools.build.lib.query2.engine.QueryUtil.MinDepthUniquifierImpl;
import com.google.devtools.build.lib.query2.engine.QueryUtil.MutableKeyExtractorBackedMapImpl;
import com.google.devtools.build.lib.query2.engine.QueryUtil.UniquifierImpl;
import com.google.devtools.build.lib.query2.engine.ThreadSafeOutputFormatterCallback;
import com.google.devtools.build.lib.query2.engine.Uniquifier;
import com.google.devtools.build.lib.rules.AliasConfiguredTarget;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetValue;
import com.google.devtools.build.lib.skyframe.GraphBackedRecursivePackageProvider;
import com.google.devtools.build.lib.skyframe.PackageValue;
import com.google.devtools.build.lib.skyframe.RecursivePackageProviderBackedTargetPatternResolver;
import com.google.devtools.build.lib.skyframe.RecursivePkgValueRootPackageExtractor;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import com.google.devtools.build.lib.skyframe.TargetPatternValue;
import com.google.devtools.build.lib.skyframe.TargetPatternValue.TargetPatternKey;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.WalkableGraph;
import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import javax.annotation.Nullable;

/**
 * {@link QueryEnvironment} that runs queries based on results from the analysis phase.
 *
 * <p>This environment can theoretically be used for multiple queries, but currently is only ever
 * used for one over the course of its lifetime. If this ever changed to be used for multiple, the
 * {@link TargetAccessor} field should be initialized on a per-query basis not a per-environment
 * basis.
 *
 * <p>Aspects are also not supported, but probably should be in some fashion.
 */
public abstract class PostAnalysisQueryEnvironment<T> extends AbstractBlazeQueryEnvironment<T> {
  protected final TopLevelConfigurations topLevelConfigurations;
  protected final BuildConfiguration hostConfiguration;
  private final String parserPrefix;
  private final PathPackageLocator pkgPath;
  private final Supplier<WalkableGraph> walkableGraphSupplier;
  protected WalkableGraph graph;

  private static final Function<SkyKey, ConfiguredTargetKey> SKYKEY_TO_CTKEY =
      skyKey -> (ConfiguredTargetKey) skyKey.argument();
  private static final ImmutableList<TargetPatternKey> ALL_PATTERNS;

  static {
    TargetPattern targetPattern;
    try {
      targetPattern = TargetPattern.defaultParser().parse("//...");
    } catch (TargetParsingException e) {
      throw new IllegalStateException(e);
    }
    ALL_PATTERNS =
        ImmutableList.of(
            new TargetPatternKey(
                targetPattern, FilteringPolicies.NO_FILTER, false, "", ImmutableSet.of()));
  }

  protected RecursivePackageProviderBackedTargetPatternResolver resolver;

  public PostAnalysisQueryEnvironment(
      boolean keepGoing,
      ExtendedEventHandler eventHandler,
      Iterable<QueryFunction> extraFunctions,
      TopLevelConfigurations topLevelConfigurations,
      BuildConfiguration hostConfiguration,
      String parserPrefix,
      PathPackageLocator pkgPath,
      Supplier<WalkableGraph> walkableGraphSupplier,
      Set<Setting> settings) {
    super(keepGoing, true, Rule.ALL_LABELS, eventHandler, settings, extraFunctions);
    this.topLevelConfigurations = topLevelConfigurations;
    this.hostConfiguration = hostConfiguration;
    this.parserPrefix = parserPrefix;
    this.pkgPath = pkgPath;
    this.walkableGraphSupplier = walkableGraphSupplier;
  }

  public abstract ImmutableList<NamedThreadSafeOutputFormatterCallback<T>>
      getDefaultOutputFormatters(
          TargetAccessor<T> accessor,
          ExtendedEventHandler eventHandler,
          OutputStream outputStream,
          SkyframeExecutor skyframeExecutor,
          BuildConfiguration hostConfiguration,
          @Nullable RuleTransitionFactory trimmingTransitionFactory,
          PackageManager packageManager);

  public abstract String getOutputFormat();

  protected abstract KeyExtractor<T, ConfiguredTargetKey> getConfiguredTargetKeyExtractor();

  @Override
  public QueryEvalResult evaluateQuery(
      QueryExpression expr, ThreadSafeOutputFormatterCallback<T> callback)
      throws QueryException, InterruptedException, IOException {
    beforeEvaluateQuery();
    return super.evaluateQuery(expr, callback);
  }

  private void beforeEvaluateQuery() throws QueryException {
    graph = walkableGraphSupplier.get();
    GraphBackedRecursivePackageProvider graphBackedRecursivePackageProvider =
        new GraphBackedRecursivePackageProvider(
            graph, ALL_PATTERNS, pkgPath, new RecursivePkgValueRootPackageExtractor());
    resolver =
        new RecursivePackageProviderBackedTargetPatternResolver(
            graphBackedRecursivePackageProvider,
            eventHandler,
            FilteringPolicies.NO_FILTER,
            MultisetSemaphore.unbounded());
    checkSettings(settings);
  }

  // Check to make sure the settings requested are currently supported by this class
  private void checkSettings(Set<Setting> settings) throws QueryException {
    if (settings.contains(Setting.NO_NODEP_DEPS)
        || settings.contains(Setting.TESTS_EXPRESSION_STRICT)) {
      settings =
          Sets.difference(
              settings, ImmutableSet.of(Setting.NO_HOST_DEPS, Setting.NO_IMPLICIT_DEPS));
      throw new QueryException(
          String.format(
              "The following filter(s) are not currently supported by configured query: %s",
              settings.toString()));
    }
  }

  public BuildConfiguration getHostConfiguration() {
    return hostConfiguration;
  }

  // TODO(bazel-team): It's weird that this untemplated function exists. Fix? Or don't implement?
  @Override
  public Target getTarget(Label label) throws TargetNotFoundException, InterruptedException {
    try {
      return ((PackageValue)
              walkableGraphSupplier.get().getValue(PackageValue.key(label.getPackageIdentifier())))
          .getPackage()
          .getTarget(label.getName());
    } catch (NoSuchTargetException e) {
      throw new TargetNotFoundException(e);
    }
  }

  @Override
  public T getOrCreate(T target) {
    return target;
  }

  /**
   * This method has to exist because {@link AliasConfiguredTarget#getLabel()} returns the label of
   * the "actual" target instead of the alias target. Grr.
   */
  public abstract Label getCorrectLabel(T target);

  @Nullable
  protected abstract T getHostConfiguredTarget(Label label) throws InterruptedException;

  @Nullable
  protected abstract T getTargetConfiguredTarget(Label label) throws InterruptedException;

  @Nullable
  protected abstract T getNullConfiguredTarget(Label label) throws InterruptedException;

  @Nullable
  ConfiguredTargetValue getConfiguredTargetValue(SkyKey key) throws InterruptedException {
    return (ConfiguredTargetValue) walkableGraphSupplier.get().getValue(key);
  }

  @Nullable
  protected abstract T getValueFromKey(SkyKey key) throws InterruptedException;

  protected TargetPattern getPattern(String pattern) throws TargetParsingException {
    TargetPatternKey targetPatternKey =
        ((TargetPatternKey)
            TargetPatternValue.key(
                    pattern, TargetPatternEvaluator.DEFAULT_FILTERING_POLICY, parserPrefix)
                .argument());
    return targetPatternKey.getParsedPattern();
  }

  ThreadSafeMutableSet<T> getFwdDeps(Iterable<T> targets) throws InterruptedException {
    Map<SkyKey, T> targetsByKey = Maps.newHashMapWithExpectedSize(Iterables.size(targets));
    for (T target : targets) {
      targetsByKey.put(getSkyKey(target), target);
    }
    Map<SkyKey, Collection<T>> directDeps =
        targetifyValues(graph.getDirectDeps(targetsByKey.keySet()));
    if (targetsByKey.size() != directDeps.size()) {
      Iterable<ConfiguredTargetKey> missingTargets =
          Sets.difference(targetsByKey.keySet(), directDeps.keySet()).stream()
              .map(SKYKEY_TO_CTKEY)
              .collect(Collectors.toList());
      eventHandler.handle(Event.warn("Targets were missing from graph: " + missingTargets));
    }
    ThreadSafeMutableSet<T> result = createThreadSafeMutableSet();
    for (Map.Entry<SkyKey, Collection<T>> entry : directDeps.entrySet()) {
      result.addAll(filterFwdDeps(targetsByKey.get(entry.getKey()), entry.getValue()));
    }
    return result;
  }

  @Override
  public ThreadSafeMutableSet<T> getFwdDeps(Iterable<T> targets, QueryExpressionContext<T> context)
      throws InterruptedException {
    return getFwdDeps(targets);
  }

  private Collection<T> filterFwdDeps(T configTarget, Collection<T> rawFwdDeps) {
    if (settings.isEmpty()) {
      return rawFwdDeps;
    }
    return getAllowedDeps(configTarget, rawFwdDeps);
  }

  @Override
  public Collection<T> getReverseDeps(Iterable<T> targets, QueryExpressionContext<T> context)
      throws InterruptedException {
    Map<SkyKey, T> targetsByKey = Maps.newHashMapWithExpectedSize(Iterables.size(targets));
    for (T target : targets) {
      targetsByKey.put(getSkyKey(target), target);
    }
    Map<SkyKey, Collection<T>> reverseDepsByKey =
        targetifyValues(graph.getReverseDeps(targetsByKey.keySet()));
    if (targetsByKey.size() != reverseDepsByKey.size()) {
      Iterable<ConfiguredTargetKey> missingTargets =
          Sets.difference(targetsByKey.keySet(), reverseDepsByKey.keySet()).stream()
              .map(SKYKEY_TO_CTKEY)
              .collect(Collectors.toList());
      eventHandler.handle(Event.warn("Targets were missing from graph: " + missingTargets));
    }
    Map<T, Collection<T>> reverseDepsByCT = new HashMap<>();
    for (Map.Entry<SkyKey, Collection<T>> entry : reverseDepsByKey.entrySet()) {
      reverseDepsByCT.put(targetsByKey.get(entry.getKey()), entry.getValue());
    }
    return reverseDepsByCT.isEmpty() ? Collections.emptyList() : filterReverseDeps(reverseDepsByCT);
  }

  private Collection<T> filterReverseDeps(Map<T, Collection<T>> rawReverseDeps) {
    Set<T> result = CompactHashSet.create();
    for (Map.Entry<T, Collection<T>> targetAndRdeps : rawReverseDeps.entrySet()) {
      ImmutableSet.Builder<T> ruleDeps = ImmutableSet.builder();
      for (T parent : targetAndRdeps.getValue()) {
        if (parent instanceof RuleConfiguredTarget
            && dependencyFilter != DependencyFilter.ALL_DEPS) {
          ruleDeps.add(parent);
        } else {
          result.add(parent);
        }
      }
      result.addAll(getAllowedDeps((targetAndRdeps.getKey()), ruleDeps.build()));
    }
    return result;
  }

  /**
   * @param target source target
   * @param deps next level of deps to filter
   */
  protected Collection<T> getAllowedDeps(T target, Collection<T> deps) {
    // It's possible to query on a target that's configured in the host configuration. In those
    // cases if --nohost_deps is turned on, we only allow reachable targets that are ALSO in the
    // host config. This is somewhat counterintuitive and subject to change in the future but seems
    // like the best option right now.
    if (settings.contains(Setting.NO_HOST_DEPS)) {
      BuildConfiguration currentConfig = getConfiguration(target);
      if (currentConfig != null && currentConfig.isHostConfiguration()) {
        deps =
            deps.stream()
                .filter(
                    dep ->
                        getConfiguration(dep) != null
                            && getConfiguration(dep).isHostConfiguration())
                .collect(Collectors.toList());
      } else {
        deps =
            deps.stream()
                .filter(
                    dep ->
                        getConfiguration(dep) != null
                            && !getConfiguration(dep).isHostConfiguration())
                .collect(Collectors.toList());
      }
    }
    if (settings.contains(Setting.NO_IMPLICIT_DEPS)) {
      RuleConfiguredTarget ruleConfiguredTarget = getRuleConfiguredTarget(target);
      if (ruleConfiguredTarget != null) {
        Set<ConfiguredTargetKey> implicitDeps = ruleConfiguredTarget.getImplicitDeps();
        deps =
            deps.stream()
                .filter(
                    dep ->
                        !implicitDeps.contains(
                            ConfiguredTargetKey.of(getCorrectLabel(dep), getConfiguration(dep))))
                .collect(Collectors.toList());
      }
    }
    return deps;
  }

  protected abstract RuleConfiguredTarget getRuleConfiguredTarget(T target);

  protected Map<SkyKey, Collection<T>> targetifyValues(
      Map<SkyKey, ? extends Iterable<SkyKey>> input) throws InterruptedException {
    Map<SkyKey, Collection<T>> result = new HashMap<>();
    for (Map.Entry<SkyKey, ? extends Iterable<SkyKey>> entry : input.entrySet()) {
      Collection<T> value = new ArrayList<>();
      for (SkyKey key : entry.getValue()) {
        if (key.functionName().equals(SkyFunctions.CONFIGURED_TARGET)) {
          value.add(getValueFromKey(key));
        }
      }
      result.put(entry.getKey(), value);
    }
    return result;
  }

  @Nullable
  protected abstract BuildConfiguration getConfiguration(T target);

  protected abstract ConfiguredTargetKey getSkyKey(T target);

  @Override
  public ThreadSafeMutableSet<T> getTransitiveClosure(
      ThreadSafeMutableSet<T> targets, QueryExpressionContext<T> context)
      throws InterruptedException {
    return SkyQueryUtils.getTransitiveClosure(
        targets, targets1 -> getFwdDeps(targets1, context), createThreadSafeMutableSet());
  }

  @Override
  public void buildTransitiveClosure(
      QueryExpression caller, ThreadSafeMutableSet<T> targetNodes, int maxDepth) {
    // TODO(bazel-team): implement this. Just needed for error-checking.
  }

  @Override
  public ImmutableList<T> getNodesOnPath(T from, T to, QueryExpressionContext<T> context)
      throws InterruptedException {
    return SkyQueryUtils.getNodesOnPath(
        from,
        to,
        targets -> getFwdDeps(targets, context),
        getConfiguredTargetKeyExtractor()::extractKey);
  }

  @Override
  public <V> MutableMap<T, V> createMutableMap() {
    return new MutableKeyExtractorBackedMapImpl<>(getConfiguredTargetKeyExtractor());
  }

  @Override
  public Uniquifier<T> createUniquifier() {
    return new UniquifierImpl<>(getConfiguredTargetKeyExtractor());
  }

  @Override
  public MinDepthUniquifier<T> createMinDepthUniquifier() {
    return new MinDepthUniquifierImpl<>(
        getConfiguredTargetKeyExtractor(), SkyQueryEnvironment.DEFAULT_THREAD_COUNT);
  }

  /** Target patterns are resolved on the fly so no pre-work to be done here. */
  @Override
  protected void preloadOrThrow(QueryExpression caller, Collection<String> patterns) {}

  @Override
  public ThreadSafeMutableSet<T> getBuildFiles(
      QueryExpression caller,
      ThreadSafeMutableSet<T> nodes,
      boolean buildFiles,
      boolean loads,
      QueryExpressionContext<T> context)
      throws QueryException {
    throw new QueryException("buildfiles() doesn't make sense for the configured target graph");
  }

  @Override
  public Collection<T> getSiblingTargetsInPackage(T target) {
    throw new UnsupportedOperationException("siblings() not supported");
  }

  @Override
  public void close() {}

  /** A wrapper class for the set of top-level configurations in a query. */
  public static class TopLevelConfigurations {

    private static final List<BuildConfiguration> SINGLETON_CONFIGURATION_LIST_OF_NULL =
        Collections.singletonList(null);

    /** A map of non-null configured top-level targets sorted by configuration checksum. */
    private final ImmutableMap<Label, BuildConfiguration> nonNulls;
    /**
     * {@code nonNulls} may often have many duplicate values in its value set so we store a sorted
     * set of all the non-null configurations here.
     */
    private final ImmutableSortedSet<BuildConfiguration> nonNullConfigs;
    /** A list of null configured top-level targets. */
    private final ImmutableList<Label> nulls;

    public TopLevelConfigurations(
        Collection<TargetAndConfiguration> topLevelTargetsAndConfigurations) {
      ImmutableMap.Builder<Label, BuildConfiguration> nonNullsBuilder =
          ImmutableMap.builderWithExpectedSize(topLevelTargetsAndConfigurations.size());
      ImmutableList.Builder<Label> nullsBuilder = new ImmutableList.Builder<>();
      for (TargetAndConfiguration targetAndConfiguration : topLevelTargetsAndConfigurations) {
        if (targetAndConfiguration.getConfiguration() == null) {
          nullsBuilder.add(targetAndConfiguration.getLabel());
        } else {
          nonNullsBuilder.put(
              targetAndConfiguration.getLabel(), targetAndConfiguration.getConfiguration());
        }
      }
      nonNulls = nonNullsBuilder.build();
      nonNullConfigs =
          ImmutableSortedSet.copyOf(
              Comparator.comparing(BuildConfiguration::checksum), nonNulls.values());
      nulls = nullsBuilder.build();
    }

    boolean isTopLevelTarget(Label label) {
      return nonNulls.containsKey(label) || nulls.contains(label);
    }

    // This method returns the configuration of a top-level target if it's not null-configured and
    // otherwise returns null (signifying it is null configured).
    @Nullable
    BuildConfiguration getConfigurationForTopLevelTarget(Label label) {
      Preconditions.checkArgument(
          isTopLevelTarget(label),
          "Attempting to get top-level configuration for non-top-level target %s.",
          label);
      return nonNulls.get(label);
    }

    public Iterable<BuildConfiguration> getConfigurations() {
      if (nulls.isEmpty()) {
        return nonNullConfigs;
      } else {
        return Iterables.concat(nonNullConfigs, SINGLETON_CONFIGURATION_LIST_OF_NULL);
      }
    }
  }
}
