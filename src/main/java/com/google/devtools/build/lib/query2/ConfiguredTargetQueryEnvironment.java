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
import com.google.devtools.build.lib.collect.compacthashset.CompactHashSet;
import com.google.devtools.build.lib.concurrent.MultisetSemaphore;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.packages.DependencyFilter;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleTransitionFactory;
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
import com.google.devtools.build.lib.query2.output.AspectResolver;
import com.google.devtools.build.lib.query2.output.CqueryOptions;
import com.google.devtools.build.lib.query2.output.QueryOptions;
import com.google.devtools.build.lib.rules.AliasConfiguredTarget;
import com.google.devtools.build.lib.skyframe.BuildConfigurationValue;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetValue;
import com.google.devtools.build.lib.skyframe.GraphBackedRecursivePackageProvider;
import com.google.devtools.build.lib.skyframe.PackageValue;
import com.google.devtools.build.lib.skyframe.RecursivePackageProviderBackedTargetPatternResolver;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import com.google.devtools.build.lib.skyframe.TargetPatternValue;
import com.google.devtools.build.lib.skyframe.TargetPatternValue.TargetPatternKey;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.WalkableGraph;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;
import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import javax.annotation.Nullable;

/**
 * {@link QueryEnvironment} that runs queries over the configured target (analysis) graph.
 *
 * <p>This environment can theoretically be used for multiple queries, but currently is only ever
 * used for one over the course of its lifetime. If this ever changed to be used for multiple, the
 * {@link accessor} field should be initialized on a per-query basis not a per-environment basis.
 *
 * <p>There is currently a limited way to specify a configuration in the query syntax via
 * {@link ConfigFunction}. This currently still limits the user to choosing the 'target', 'host', or
 * null configurations. It shouldn't be terribly difficult to expand this with
 * {@link OptionsDiffForReconstruction} to handle fully customizable configurations if the need
 * arises in the future.
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
  private ConfiguredTargetAccessor accessor;
  protected WalkableGraph graph;

  private static final Function<SkyKey, ConfiguredTargetKey> SKYKEY_TO_CTKEY =
      skyKey -> (ConfiguredTargetKey) skyKey.argument();
  private static final ImmutableList<TargetPatternKey> ALL_PATTERNS;
  private final KeyExtractor<ConfiguredTarget, ConfiguredTargetKey> configuredTargetKeyExtractor;

  /** Common query functions and cquery specific functions. */
  public static final ImmutableList<QueryFunction> FUNCTIONS = populateFunctions();
  /** Cquery specific functions. */
  public static final ImmutableList<QueryFunction> CQUERY_FUNCTIONS = getCqueryFunctions();

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
    this.accessor = new ConfiguredTargetAccessor(walkableGraphSupplier.get());
    this.configuredTargetKeyExtractor =
        element -> {
          try {
            return ConfiguredTargetKey.of(
                element,
                element.getConfigurationKey() == null
                    ? null
                    : ((BuildConfigurationValue) graph.getValue(element.getConfigurationKey()))
                        .getConfiguration());
          } catch (InterruptedException e) {
            throw new IllegalStateException("Interruption unexpected in configured query");
          }
        };
  }

  private static ImmutableList<QueryFunction> populateFunctions() {
    return new ImmutableList.Builder<QueryFunction>()
        .addAll(QueryEnvironment.DEFAULT_QUERY_FUNCTIONS)
        .addAll(getCqueryFunctions())
        .build();
  }

  private static ImmutableList<QueryFunction> getCqueryFunctions() {
    return ImmutableList.of(new ConfigFunction());
  }

  public ImmutableList<CqueryThreadsafeCallback> getDefaultOutputFormatters(
      TargetAccessor<ConfiguredTarget> accessor,
      CqueryOptions options,
      Reporter reporter,
      SkyframeExecutor skyframeExecutor,
      BuildConfiguration hostConfiguration,
      @Nullable RuleTransitionFactory trimmingTransitionFactory,
      AspectResolver resolver) {
    OutputStream out = reporter.getOutErr().getOutputStream();
    return new ImmutableList.Builder<CqueryThreadsafeCallback>()
        .add(
            new LabelAndConfigurationOutputFormatterCallback(
                reporter, options, out, skyframeExecutor, accessor))
        .add(
            new TransitionsOutputFormatterCallback(
                reporter,
                options,
                out,
                skyframeExecutor,
                accessor,
                hostConfiguration,
                trimmingTransitionFactory))
        .add(
            new ProtoOutputFormatterCallback(
                reporter, options, out, skyframeExecutor, accessor, resolver))
        .build();
  }

  @Override
  public QueryEvalResult evaluateQuery(
      QueryExpression expr, ThreadSafeOutputFormatterCallback<ConfiguredTarget> callback)
      throws QueryException, InterruptedException, IOException {
    beforeEvaluateQuery();
    return super.evaluateQuery(expr, callback);
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

  public BuildConfiguration getHostConfiguration() {
    return hostConfiguration;
  }

  @Override
  public TargetAccessor<ConfiguredTarget> getAccessor() {
    return accessor;
  }

  // TODO(bazel-team): It's weird that this untemplated function exists. Fix? Or don't implement?
  @Override
  public Target getTarget(Label label)
      throws TargetNotFoundException, QueryException, InterruptedException {
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
  public ConfiguredTarget getOrCreate(ConfiguredTarget target) {
    return target;
  }

  /**
   * This method has to exist because {@link AliasConfiguredTarget#getLabel()} returns
   * the label of the "actual" target instead of the alias target. Grr.
   */
  public static Label getCorrectLabel(ConfiguredTarget target) {
    if (target instanceof AliasConfiguredTarget) {
      return ((AliasConfiguredTarget) target).getOriginalLabel();
    }
    return target.getLabel();
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

  private ConfiguredTarget getConfiguredTarget(Label label) throws InterruptedException {
    // Try with target configuration.
    ConfiguredTarget configuredTarget = getTargetConfiguredTarget(label);
    if (configuredTarget != null) {
      return configuredTarget;
    }
    // Try with host configuration (even when --nohost_deps is set in the case that top-level
    // targets are configured in the host configuration so we are doing a host-configuration-only
    // query).
    configuredTarget = getHostConfiguredTarget(label);
    if (configuredTarget != null) {
      return configuredTarget;
    }
    // Last chance: source file.
    return getNullConfiguredTarget(label);
  }

  @Nullable
  private ConfiguredTarget getConfiguredTarget(SkyKey key) throws InterruptedException {
    ConfiguredTargetValue value =
        ((ConfiguredTargetValue) walkableGraphSupplier.get().getValue(key));
    return value == null ? null : value.getConfiguredTarget();
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

  /**
   * Processes the targets in {@code targets} with the requested {@code configuration}
   *
   * @param pattern the original pattern that {@code targets} were parsed from. Used for error
   *     message.
   * @param targets the set of {@link ConfiguredTarget}s whose labels represent the targets being
   *     requested.
   * @param configuration the configuration to request {@code targets} in.
   * @param callback the callback to receive the results of this method.
   * @return {@link QueryTaskCallable} that returns the correctly configured targets.
   */
  QueryTaskCallable<Void> getConfiguredTargets(
      String pattern,
      ThreadSafeMutableSet<ConfiguredTarget> targets,
      String configuration,
      Callback<ConfiguredTarget> callback) {
    return new QueryTaskCallable<Void>() {
      @Override
      public Void call() throws QueryException, InterruptedException {
        List<ConfiguredTarget> transformedResult = new ArrayList<>();
        for (ConfiguredTarget target : targets) {
          Label label = getCorrectLabel(target);
          ConfiguredTarget configuredTarget;
          switch (configuration) {
            case "\'host\'":
              configuredTarget = getHostConfiguredTarget(label);
              break;
            case "\'target\'":
              configuredTarget = getTargetConfiguredTarget(label);
              break;
            case "\'null\'":
              configuredTarget = getNullConfiguredTarget(label);
              break;
            default:
              throw new QueryException(
                  "the second argument of the config function must be 'target', 'host', or 'null'");
          }
          if (configuredTarget != null) {
            transformedResult.add(configuredTarget);
          }
        }
        if (transformedResult.isEmpty()) {
          throw new QueryException(
              "No target (in) "
                  + pattern
                  + " could be found in the "
                  + configuration
                  + " configuration");
        }
        callback.process(transformedResult);
        return null;
      }
    };
  }

  @Nullable
  private ConfiguredTarget getHostConfiguredTarget(Label label) throws InterruptedException {
    return getConfiguredTarget(ConfiguredTargetValue.key(label, hostConfiguration));
  }

  @Nullable
  private ConfiguredTarget getTargetConfiguredTarget(Label label) throws InterruptedException {
    return getConfiguredTarget(ConfiguredTargetValue.key(label, defaultTargetConfiguration));
  }

  @Nullable
  private ConfiguredTarget getNullConfiguredTarget(Label label) throws InterruptedException {
    return getConfiguredTarget(ConfiguredTargetValue.key(label, null));
  }

  @Override
  public ThreadSafeMutableSet<ConfiguredTarget> getFwdDeps(Iterable<ConfiguredTarget> targets)
      throws InterruptedException {
    Map<SkyKey, ConfiguredTarget> targetsByKey = new HashMap<>(Iterables.size(targets));
    for (ConfiguredTarget target : targets) {
      targetsByKey.put(getSkyKey(target), target);
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
    for (Map.Entry<SkyKey, Collection<ConfiguredTarget>> entry : directDeps.entrySet()) {
      result.addAll(filterFwdDeps(targetsByKey.get(entry.getKey()), entry.getValue()));
    }
    return result;
  }

  private Collection<ConfiguredTarget> filterFwdDeps(
      ConfiguredTarget configTarget, Collection<ConfiguredTarget> rawFwdDeps)
      throws InterruptedException {
    if (settings.isEmpty()) {
      return rawFwdDeps;
    }
    return getAllowedDeps(configTarget, rawFwdDeps);
  }

  @Override
  public Collection<ConfiguredTarget> getReverseDeps(Iterable<ConfiguredTarget> targets)
      throws InterruptedException {
    Map<SkyKey, ConfiguredTarget> targetsByKey = new HashMap<>(Iterables.size(targets));
    for (ConfiguredTarget target : targets) {
      targetsByKey.put(getSkyKey(target), target);
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
    if (settings.contains(Setting.NO_IMPLICIT_DEPS) && target instanceof RuleConfiguredTarget) {
      Set<ConfiguredTargetKey> implicitDeps = ((RuleConfiguredTarget) target).getImplicitDeps();
      deps =
          deps.stream()
              .filter(
                  dep ->
                      !implicitDeps.contains(
                          ConfiguredTargetKey.of(getCorrectLabel(dep), getConfiguration(dep))))
              .collect(Collectors.toList());
    }
    return deps;
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

  @Nullable
  private BuildConfiguration getConfiguration(ConfiguredTarget target) {
    try {
      return target.getConfigurationKey() == null
          ? null
          : ((BuildConfigurationValue) graph.getValue(target.getConfigurationKey()))
              .getConfiguration();
    } catch (InterruptedException e) {
      throw new IllegalStateException("Unexpected interruption during configured target query");
    }
  }

  private ConfiguredTargetKey getSkyKey(ConfiguredTarget target) {
    return ConfiguredTargetKey.of(target, getConfiguration(target));
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
    return SkyQueryUtils.getNodesOnPath(
        from, to, this::getFwdDeps, configuredTargetKeyExtractor::extractKey);
  }

  @Override
  public ThreadSafeMutableSet<ConfiguredTarget> createThreadSafeMutableSet() {
    return new ThreadSafeMutableKeyExtractorBackedSetImpl<>(
        configuredTargetKeyExtractor,
        ConfiguredTarget.class,
        SkyQueryEnvironment.DEFAULT_THREAD_COUNT);
  }

  @Override
  public <V> MutableMap<ConfiguredTarget, V> createMutableMap() {
    return new MutableKeyExtractorBackedMapImpl<>(configuredTargetKeyExtractor);
  }

  @Override
  public Uniquifier<ConfiguredTarget> createUniquifier() {
    return new UniquifierImpl<>(
        configuredTargetKeyExtractor, SkyQueryEnvironment.DEFAULT_THREAD_COUNT);
  }

  @Override
  public MinDepthUniquifier<ConfiguredTarget> createMinDepthUniquifier() {
    return new MinDepthUniquifierImpl<>(
        configuredTargetKeyExtractor, SkyQueryEnvironment.DEFAULT_THREAD_COUNT);
  }

  @Override
  protected void preloadOrThrow(QueryExpression caller, Collection<String> patterns)
      throws QueryException, TargetParsingException, InterruptedException {
    for (String pattern : patterns) {
      if (TargetPattern.defaultParser()
          .parse(pattern)
          .getType()
          .equals(TargetPattern.Type.TARGETS_BELOW_DIRECTORY)) {
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

  @Override
  public ThreadSafeMutableSet<ConfiguredTarget> getBuildFiles(
      QueryExpression caller,
      ThreadSafeMutableSet<ConfiguredTarget> nodes,
      boolean buildFiles,
      boolean loads)
      throws QueryException, InterruptedException {
    throw new QueryException("buildfiles() doesn't make sense for the configured target graph");
  }

  @Override
  public Collection<ConfiguredTarget> getSiblingTargetsInPackage(ConfiguredTarget target) {
    throw new UnsupportedOperationException("siblings() not supported");
  }


  @Override
  public void close() {}
}

