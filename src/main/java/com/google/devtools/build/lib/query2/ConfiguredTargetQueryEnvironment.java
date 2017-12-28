// Copyright 2017 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;
import com.google.common.util.concurrent.AsyncFunction;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.cmdline.TargetPattern;
import com.google.devtools.build.lib.cmdline.TargetPattern.Type;
import com.google.devtools.build.lib.collect.compacthashset.CompactHashSet;
import com.google.devtools.build.lib.concurrent.MultisetSemaphore;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.packages.DependencyFilter;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.pkgcache.FilteringPolicies;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.pkgcache.TargetPatternEvaluator;
import com.google.devtools.build.lib.query2.engine.Callback;
import com.google.devtools.build.lib.query2.engine.KeyExtractor;
import com.google.devtools.build.lib.query2.engine.MinDepthUniquifier;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment;
import com.google.devtools.build.lib.query2.engine.QueryEvalResult;
import com.google.devtools.build.lib.query2.engine.QueryException;
import com.google.devtools.build.lib.query2.engine.QueryExpression;
import com.google.devtools.build.lib.query2.engine.QueryUtil.MinDepthUniquifierImpl;
import com.google.devtools.build.lib.query2.engine.QueryUtil.MutableKeyExtractorBackedMapImpl;
import com.google.devtools.build.lib.query2.engine.QueryUtil.ThreadSafeMutableKeyExtractorBackedSetImpl;
import com.google.devtools.build.lib.query2.engine.QueryUtil.UniquifierImpl;
import com.google.devtools.build.lib.query2.engine.ThreadSafeOutputFormatterCallback;
import com.google.devtools.build.lib.query2.engine.Uniquifier;
import com.google.devtools.build.lib.query2.output.QueryOptions;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetValue;
import com.google.devtools.build.lib.skyframe.GraphBackedRecursivePackageProvider;
import com.google.devtools.build.lib.skyframe.RecursivePackageProviderBackedTargetPatternResolver;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.skyframe.TargetPatternValue;
import com.google.devtools.build.lib.skyframe.TargetPatternValue.TargetPatternKey;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.WalkableGraph;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import javax.annotation.Nullable;

/**
 * {@link QueryEnvironment} that runs queries over the configured target (analysis) graph.
 *
 * <p>This object can theoretically be used for multiple queries, but currently is only ever used
 * for one over the course of its lifetime.
 *
 * <p>There is currently no way to specify a configuration in the query syntax. Instead, the default
 * configuration that will be used for any raw labels is provided in the constructor of this
 * environment. That will probably have to change.
 *
 * <p>On the other end, recursive target patterns are not supported.
 *
 * <p>Aspects are also not supported, but probably should be in some fashion.
 */
public class ConfiguredTargetQueryEnvironment
    extends AbstractBlazeQueryEnvironment<ConfiguredTarget> {
  private final BuildConfiguration defaultTargetConfiguration;
  private final BuildConfiguration hostConfiguration;
  private final String parserPrefix;
  protected final PathPackageLocator pkgPath;
  private final Supplier<WalkableGraph> walkableGraphSupplier;
  private final ConfiguredTargetAccessor accessor = new ConfiguredTargetAccessor();
  protected WalkableGraph graph;

  private static final Function<ConfiguredTarget, SkyKey> CT_TO_SKYKEY =
      target -> ConfiguredTargetValue.key(target.getLabel(), target.getConfiguration());
  private static final Function<SkyKey, ConfiguredTargetKey> SKYKEY_TO_CTKEY =
      skyKey -> (ConfiguredTargetKey) skyKey.argument();
  private static final ImmutableList<TargetPatternKey> ALL_PATTERNS;
  private static final KeyExtractor<ConfiguredTarget, ConfiguredTargetKey>
      CONFIGURED_TARGET_KEY_EXTRACTOR = ConfiguredTargetKey::of;

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

  private RecursivePackageProviderBackedTargetPatternResolver resolver;

  public ConfiguredTargetQueryEnvironment(
      boolean keepGoing,
      ExtendedEventHandler eventHandler,
      Iterable<QueryFunction> extraFunctions,
      BuildConfiguration defaultTargetConfiguration,
      BuildConfiguration hostConfiguration,
      String parserPrefix,
      PathPackageLocator pkgPath,
      Supplier<WalkableGraph> walkableGraphSupplier,
      Set<Setting> settings) {
    super(keepGoing, true, Rule.ALL_LABELS, eventHandler, settings, extraFunctions);
    this.defaultTargetConfiguration = defaultTargetConfiguration;
    this.hostConfiguration = hostConfiguration;
    this.parserPrefix = parserPrefix;
    this.pkgPath = pkgPath;
    this.walkableGraphSupplier = walkableGraphSupplier;
  }

  private void beforeEvaluateQuery() throws InterruptedException, QueryException {
    graph = walkableGraphSupplier.get();
    GraphBackedRecursivePackageProvider graphBackedRecursivePackageProvider =
        new GraphBackedRecursivePackageProvider(graph, ALL_PATTERNS, pkgPath);
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

  @Nullable
  private ConfiguredTarget getConfiguredTarget(SkyKey key) throws InterruptedException {
    ConfiguredTargetValue value =
        ((ConfiguredTargetValue) walkableGraphSupplier.get().getValue(key));
    return value == null ? null : value.getConfiguredTarget();
  }

  @Override
  public void close() {}

  @Override
  public QueryEvalResult evaluateQuery(
      QueryExpression expr, ThreadSafeOutputFormatterCallback<ConfiguredTarget> callback)
      throws QueryException, InterruptedException, IOException {
    beforeEvaluateQuery();
    return super.evaluateQuery(expr, callback);
  }

  private TargetPattern getPattern(String pattern)
      throws TargetParsingException, InterruptedException {
    TargetPatternKey targetPatternKey =
        ((TargetPatternKey)
            TargetPatternValue.key(
                    pattern, TargetPatternEvaluator.DEFAULT_FILTERING_POLICY, parserPrefix)
                .argument());
    return targetPatternKey.getParsedPattern();
  }

  @Override
  public Collection<ConfiguredTarget> getSiblingTargetsInPackage(ConfiguredTarget target) {
    throw new UnsupportedOperationException("siblings() not supported");
  }

  @Override
  public QueryTaskFuture<Void> getTargetsMatchingPattern(
      QueryExpression owner, String pattern, Callback<ConfiguredTarget> callback) {
    TargetPattern patternToEval;
    try {
      patternToEval = getPattern(pattern);
    } catch (TargetParsingException tpe) {
      try {
        reportBuildFileError(owner, tpe.getMessage());
      } catch (QueryException qe) {
        return immediateFailedFuture(qe);
      }
      return immediateSuccessfulFuture(null);
    } catch (InterruptedException ie) {
      return immediateCancelledFuture();
    }
    AsyncFunction<TargetParsingException, Void> reportBuildFileErrorAsyncFunction =
        exn -> {
          reportBuildFileError(owner, exn.getMessage());
          return Futures.immediateFuture(null);
        };
    return QueryTaskFutureImpl.ofDelegate(
        Futures.catchingAsync(
            patternToEval.evalAdaptedForAsync(
                resolver,
                ImmutableSet.of(),
                ImmutableSet.of(),
                (Callback<Target>)
                    partialResult -> {
                      List<ConfiguredTarget> transformedResult = new ArrayList<>();
                      for (Target target : partialResult) {
                        ConfiguredTarget configuredTarget = getConfiguredTarget(target.getLabel());
                        if (configuredTarget != null) {
                          transformedResult.add(configuredTarget);
                        }
                      }
                      callback.process(transformedResult);
                    },
                QueryException.class),
            TargetParsingException.class,
            reportBuildFileErrorAsyncFunction,
            MoreExecutors.directExecutor()));
  }

  @Override
  public ConfiguredTarget getOrCreate(ConfiguredTarget target) {
    return target;
  }

  private Map<SkyKey, Collection<ConfiguredTarget>> targetifyValues(
      Map<SkyKey, ? extends Iterable<SkyKey>> input) throws InterruptedException {
    Map<SkyKey, Collection<ConfiguredTarget>> result = new HashMap<>();
    for (Map.Entry<SkyKey, ? extends Iterable<SkyKey>> entry : input.entrySet()) {
      Collection<ConfiguredTarget> value = new ArrayList<>();
      for (SkyKey key : entry.getValue()) {
        if (key.functionName().equals(SkyFunctions.CONFIGURED_TARGET)) {
          value.add(getConfiguredTarget(key));
        }
      }
      result.put(entry.getKey(), value);
    }
    return result;
  }

  private Collection<ConfiguredTarget> filterFwdDeps(
      ConfiguredTarget configTarget, Collection<ConfiguredTarget> rawFwdDeps)
      throws InterruptedException {
    if (!(configTarget instanceof RuleConfiguredTarget) || settings.isEmpty()) {
      return rawFwdDeps;
    }
    return getAllowedDeps(configTarget, rawFwdDeps);
  }

  /**
   * @param target source target
   * @param deps next level of deps to filter
   */
  private Collection<ConfiguredTarget> getAllowedDeps(
      ConfiguredTarget target, Collection<ConfiguredTarget> deps) {
    // It's possible to query on a target that's configured in the host configuration. In those
    // cases if --nohost_deps is turned on, we only allow reachable targets that are ALSO in the
    // host config. This is somewhat counterintuitive and subject to change in the future but seems
    // like the best option right now.
    if (settings.contains(Setting.NO_HOST_DEPS)) {
      BuildConfiguration currentConfig = target.getConfiguration();
      if (currentConfig != null && currentConfig.isHostConfiguration()) {
        deps =
            deps.stream()
                .filter(
                    dep ->
                        dep.getConfiguration() != null
                            && dep.getConfiguration().isHostConfiguration())
                .collect(Collectors.toList());
      } else {
        deps =
            deps.stream()
                .filter(
                    dep ->
                        dep.getConfiguration() == null
                            || !dep.getConfiguration().isHostConfiguration())
                .collect(Collectors.toList());
      }
    }
    if (settings.contains(Setting.NO_IMPLICIT_DEPS) && target instanceof RuleConfiguredTarget) {
      Set<ConfiguredTargetKey> implicitDeps = ((RuleConfiguredTarget) target).getImplicitDeps();
      deps =
          deps.stream()
              .filter(
                  dep ->
                      !implicitDeps.contains(
                          ConfiguredTargetKey.of(
                              dep.getTarget().getLabel(), dep.getConfiguration())))
              .collect(Collectors.toList());
    }
    return deps;
  }

  @Override
  public ThreadSafeMutableSet<ConfiguredTarget> getFwdDeps(Iterable<ConfiguredTarget> targets)
      throws InterruptedException {
    Map<SkyKey, ConfiguredTarget> targetsByKey = new HashMap<>(Iterables.size(targets));
    for (ConfiguredTarget target : targets) {
      targetsByKey.put(CT_TO_SKYKEY.apply(target), target);
    }
    Map<SkyKey, Collection<ConfiguredTarget>> directDeps =
        targetifyValues(graph.getDirectDeps(targetsByKey.keySet()));
    if (targetsByKey.keySet().size() != directDeps.keySet().size()) {
      Iterable<ConfiguredTargetKey> missingTargets =
          Sets.difference(targetsByKey.keySet(), directDeps.keySet())
              .stream()
              .map(SKYKEY_TO_CTKEY)
              .collect(Collectors.toList());
      eventHandler.handle(Event.warn("Targets were missing from graph: " + missingTargets));
    }
    ThreadSafeMutableSet<ConfiguredTarget> result = createThreadSafeMutableSet();
    for (Entry<SkyKey, Collection<ConfiguredTarget>> entry : directDeps.entrySet()) {
      result.addAll(filterFwdDeps(targetsByKey.get(entry.getKey()), entry.getValue()));
    }
    return result;
  }

  private Collection<ConfiguredTarget> filterReverseDeps(
      Map<ConfiguredTarget, Collection<ConfiguredTarget>> rawReverseDeps) {
    Set<ConfiguredTarget> result = CompactHashSet.create();
    for (Map.Entry<ConfiguredTarget, Collection<ConfiguredTarget>> targetAndRdeps :
        rawReverseDeps.entrySet()) {
      ImmutableSet.Builder<ConfiguredTarget> ruleDeps = ImmutableSet.builder();
      for (ConfiguredTarget parent : targetAndRdeps.getValue()) {
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

  @Override
  public Collection<ConfiguredTarget> getReverseDeps(Iterable<ConfiguredTarget> targets)
      throws InterruptedException {
    Map<SkyKey, ConfiguredTarget> targetsByKey = new HashMap<>(Iterables.size(targets));
    for (ConfiguredTarget target : targets) {
      targetsByKey.put(CT_TO_SKYKEY.apply(target), target);
    }
    Map<SkyKey, Collection<ConfiguredTarget>> reverseDepsByKey =
        targetifyValues(graph.getReverseDeps(targetsByKey.keySet()));
    if (targetsByKey.size() != reverseDepsByKey.size()) {
      Iterable<ConfiguredTargetKey> missingTargets =
          Sets.difference(targetsByKey.keySet(), reverseDepsByKey.keySet())
              .stream()
              .map(SKYKEY_TO_CTKEY)
              .collect(Collectors.toList());
      eventHandler.handle(Event.warn("Targets were missing from graph: " + missingTargets));
    }
    Map<ConfiguredTarget, Collection<ConfiguredTarget>> reverseDepsByCT = new HashMap<>();
    for (Map.Entry<SkyKey, Collection<ConfiguredTarget>> entry : reverseDepsByKey.entrySet()) {
      reverseDepsByCT.put(targetsByKey.get(entry.getKey()), entry.getValue());
    }
    return reverseDepsByCT.isEmpty() ? Collections.emptyList() : filterReverseDeps(reverseDepsByCT);
  }

  @Override
  public ThreadSafeMutableSet<ConfiguredTarget> getTransitiveClosure(
      ThreadSafeMutableSet<ConfiguredTarget> targets) throws InterruptedException {
    return SkyQueryUtils.getTransitiveClosure(
        targets, this::getFwdDeps, createThreadSafeMutableSet());
  }

  @Override
  public void buildTransitiveClosure(
      QueryExpression caller, ThreadSafeMutableSet<ConfiguredTarget> targetNodes, int maxDepth)
      throws QueryException, InterruptedException {
    // TODO(bazel-team): implement this. Just needed for error-checking.
  }

  @Override
  public ImmutableList<ConfiguredTarget> getNodesOnPath(ConfiguredTarget from, ConfiguredTarget to)
      throws InterruptedException {
    return SkyQueryUtils.getNodesOnPath(from, to, this::getFwdDeps, ConfiguredTargetKey::of);
  }

  @Override
  public TargetAccessor<ConfiguredTarget> getAccessor() {
    return accessor;
  }

  // TODO(bazel-team): It's weird that this untemplated function exists. Fix? Or don't implement?
  @Override
  public Target getTarget(Label label)
      throws TargetNotFoundException, QueryException, InterruptedException {
    ConfiguredTarget configuredTarget = getConfiguredTarget(label);
    return configuredTarget == null ? null : configuredTarget.getTarget();
  }

  private ConfiguredTarget getConfiguredTarget(Label label) throws InterruptedException {
    // Try with host configuration.
    ConfiguredTarget configuredTarget =
        getConfiguredTarget(ConfiguredTargetValue.key(label, hostConfiguration));
    if (configuredTarget != null) {
      return configuredTarget;
    }
    configuredTarget =
        getConfiguredTarget(ConfiguredTargetValue.key(label, defaultTargetConfiguration));
    if (configuredTarget != null) {
      return configuredTarget;
    }
    // Last chance: source file.
    return getConfiguredTarget(ConfiguredTargetValue.key(label, null));
  }

  @Override
  public ThreadSafeMutableSet<ConfiguredTarget> createThreadSafeMutableSet() {
    return new ThreadSafeMutableKeyExtractorBackedSetImpl<>(
        CONFIGURED_TARGET_KEY_EXTRACTOR,
        ConfiguredTarget.class,
        SkyQueryEnvironment.DEFAULT_THREAD_COUNT);
  }

  @Override
  public <V> MutableMap<ConfiguredTarget, V> createMutableMap() {
    return new MutableKeyExtractorBackedMapImpl<>(CONFIGURED_TARGET_KEY_EXTRACTOR);
  }

  @Override
  public Uniquifier<ConfiguredTarget> createUniquifier() {
    return new UniquifierImpl<>(
        CONFIGURED_TARGET_KEY_EXTRACTOR, SkyQueryEnvironment.DEFAULT_THREAD_COUNT);
  }

  @Override
  public MinDepthUniquifier<ConfiguredTarget> createMinDepthUniquifier() {
    return new MinDepthUniquifierImpl<>(
        CONFIGURED_TARGET_KEY_EXTRACTOR, SkyQueryEnvironment.DEFAULT_THREAD_COUNT);
  }

  @Override
  public ThreadSafeMutableSet<ConfiguredTarget> getBuildFiles(
      QueryExpression caller,
      ThreadSafeMutableSet<ConfiguredTarget> nodes,
      boolean buildFiles,
      boolean subincludes,
      boolean loads)
      throws QueryException, InterruptedException {
    throw new QueryException("buildfiles() doesn't make sense for the configured target graph");
  }

  @Override
  protected void preloadOrThrow(QueryExpression caller, Collection<String> patterns)
      throws QueryException, TargetParsingException, InterruptedException {
    for (String pattern : patterns) {
      if (TargetPattern.defaultParser()
          .parse(pattern)
          .getType()
          .equals(Type.TARGETS_BELOW_DIRECTORY)) {
        // TODO(bazel-team): allow recursive patterns if the pattern is present in the graph? We
        // could do a mini-eval here to update the graph to contain the necessary nodes for
        // GraphBackedRecursivePackageProvider, since all the package loading and directory
        // traversal should already be done.
        throw new QueryException(
            "Recursive pattern '" + pattern + "' is not supported in configured target query");
      }
    }
  }

  public static QueryOptions parseOptions(String rawOptions) throws QueryException {
    List<String> options = new ArrayList<>(Arrays.asList(rawOptions.split(" ")));
    OptionsParser parser = OptionsParser.newOptionsParser(QueryOptions.class);
    parser.setAllowResidue(false);
    try {
      parser.parse(options);
    } catch (OptionsParsingException e) {
      throw new QueryException(e.getMessage());
    }
    return parser.getOptions(QueryOptions.class);
  }
}
