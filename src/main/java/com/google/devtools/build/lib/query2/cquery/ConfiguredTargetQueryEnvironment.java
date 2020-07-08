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
package com.google.devtools.build.lib.query2.cquery;

import com.google.common.base.Functions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.util.concurrent.AsyncFunction;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.transitions.TransitionFactory;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.cmdline.TargetPattern;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.pkgcache.PackageManager;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.query2.NamedThreadSafeOutputFormatterCallback;
import com.google.devtools.build.lib.query2.PostAnalysisQueryEnvironment;
import com.google.devtools.build.lib.query2.SkyQueryEnvironment;
import com.google.devtools.build.lib.query2.cquery.ProtoOutputFormatterCallback.OutputType;
import com.google.devtools.build.lib.query2.engine.Callback;
import com.google.devtools.build.lib.query2.engine.KeyExtractor;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment;
import com.google.devtools.build.lib.query2.engine.QueryException;
import com.google.devtools.build.lib.query2.engine.QueryExpression;
import com.google.devtools.build.lib.query2.engine.QueryUtil.ThreadSafeMutableKeyExtractorBackedSetImpl;
import com.google.devtools.build.lib.query2.query.aspectresolvers.AspectResolver;
import com.google.devtools.build.lib.rules.AliasConfiguredTarget;
import com.google.devtools.build.lib.skyframe.BuildConfigurationValue;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetValue;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.WalkableGraph;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.List;
import java.util.Set;
import java.util.function.Supplier;
import javax.annotation.Nullable;

/**
 * {@link QueryEnvironment} that runs queries over the configured target (analysis) graph.
 *
 * <p>There is currently a limited way to specify a configuration in the query syntax via {@link
 * ConfigFunction}. This currently still limits the user to choosing the 'target', 'host', or null
 * configurations. It shouldn't be terribly difficult to expand this with {@link
 * OptionsDiffForReconstruction} to handle fully customizable configurations if the need arises in
 * the future.
 *
 * <p>Aspects are also not supported, but probably should be in some fashion.
 */
public class ConfiguredTargetQueryEnvironment
    extends PostAnalysisQueryEnvironment<ConfiguredTarget> {
  /** Common query functions and cquery specific functions. */
  public static final ImmutableList<QueryFunction> FUNCTIONS = populateFunctions();
  /** Cquery specific functions. */
  public static final ImmutableList<QueryFunction> CQUERY_FUNCTIONS = getCqueryFunctions();

  private CqueryOptions cqueryOptions;

  private final KeyExtractor<ConfiguredTarget, ConfiguredTargetKey> configuredTargetKeyExtractor;

  private final ConfiguredTargetAccessor accessor;

  /**
   * Stores every configuration in the transitive closure of the build graph as a map from its
   * user-friendly hash to the configuration itself.
   *
   * <p>This is used to find configured targets in, e.g. {@code somepath} queries. Given {@code
   * somepath(//foo, //bar)}, cquery finds the configured targets for {@code //foo} and {@code
   * //bar} by creating a {@link ConfiguredTargetKey} from their labels and <i>some</i>
   * configuration, then querying the {@link WalkableGraph} to find the matching configured target.
   *
   * <p>Having this map lets cquery choose from all available configurations in the graph,
   * particularly includings configurations that aren't the host or top-level.
   *
   * <p>This can also be used in cquery's {@code config} function to match against explicitly
   * specified configs. This, in particular, is where having user-friendly hashes is invaluable.
   */
  private final ImmutableMap<String, BuildConfiguration> transitiveConfigurations;

  @Override
  protected KeyExtractor<ConfiguredTarget, ConfiguredTargetKey> getConfiguredTargetKeyExtractor() {
    return configuredTargetKeyExtractor;
  }

  public ConfiguredTargetQueryEnvironment(
      boolean keepGoing,
      ExtendedEventHandler eventHandler,
      Iterable<QueryFunction> extraFunctions,
      TopLevelConfigurations topLevelConfigurations,
      BuildConfiguration hostConfiguration,
      Collection<SkyKey> transitiveConfigurationKeys,
      String parserPrefix,
      PathPackageLocator pkgPath,
      Supplier<WalkableGraph> walkableGraphSupplier,
      Set<Setting> settings)
      throws InterruptedException {
    super(
        keepGoing,
        eventHandler,
        extraFunctions,
        topLevelConfigurations,
        hostConfiguration,
        parserPrefix,
        pkgPath,
        walkableGraphSupplier,
        settings);
    this.accessor = new ConfiguredTargetAccessor(walkableGraphSupplier.get(), this);
    this.configuredTargetKeyExtractor =
        element ->
            ConfiguredTargetKey.builder()
                .setConfiguredTarget(element)
                .setConfigurationKey(element.getConfigurationKey())
                .build();
    this.transitiveConfigurations =
        getTransitiveConfigurations(transitiveConfigurationKeys, walkableGraphSupplier.get());
  }

  public ConfiguredTargetQueryEnvironment(
      boolean keepGoing,
      ExtendedEventHandler eventHandler,
      Iterable<QueryFunction> extraFunctions,
      TopLevelConfigurations topLevelConfigurations,
      BuildConfiguration hostConfiguration,
      Collection<SkyKey> transitiveConfigurationKeys,
      String parserPrefix,
      PathPackageLocator pkgPath,
      Supplier<WalkableGraph> walkableGraphSupplier,
      CqueryOptions cqueryOptions)
      throws InterruptedException {
    this(
        keepGoing,
        eventHandler,
        extraFunctions,
        topLevelConfigurations,
        hostConfiguration,
        transitiveConfigurationKeys,
        parserPrefix,
        pkgPath,
        walkableGraphSupplier,
        cqueryOptions.toSettings());
    this.cqueryOptions = cqueryOptions;
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

  private static BuildConfiguration mergeEqualBuildConfiguration(
      BuildConfiguration left, BuildConfiguration right) {
    if (!left.equals(right)) {
      throw new IllegalArgumentException(
          "Non-matching configurations " + left.checksum() + ", " + right.checksum());
    }
    return left;
  }

  private static ImmutableMap<String, BuildConfiguration> getTransitiveConfigurations(
      Collection<SkyKey> transitiveConfigurationKeys, WalkableGraph graph)
      throws InterruptedException {
    return graph.getSuccessfulValues(transitiveConfigurationKeys).values().stream()
        .map(value -> (BuildConfigurationValue) value)
        .map(BuildConfigurationValue::getConfiguration)
        .sorted(Comparator.comparing(BuildConfiguration::checksum))
        .collect(
            ImmutableMap.toImmutableMap(
                BuildConfiguration::checksum,
                Functions.identity(),
                ConfiguredTargetQueryEnvironment::mergeEqualBuildConfiguration));
  }

  @Override
  public ImmutableList<NamedThreadSafeOutputFormatterCallback<ConfiguredTarget>>
      getDefaultOutputFormatters(
          TargetAccessor<ConfiguredTarget> accessor,
          ExtendedEventHandler eventHandler,
          OutputStream out,
          SkyframeExecutor skyframeExecutor,
          BuildConfiguration hostConfiguration,
          @Nullable TransitionFactory<Rule> trimmingTransitionFactory,
          PackageManager packageManager) {
    AspectResolver aspectResolver =
        cqueryOptions.aspectDeps.createResolver(packageManager, eventHandler);
    return ImmutableList.of(
        new LabelAndConfigurationOutputFormatterCallback(
            eventHandler, cqueryOptions, out, skyframeExecutor, accessor, true),
        new LabelAndConfigurationOutputFormatterCallback(
            eventHandler, cqueryOptions, out, skyframeExecutor, accessor, false),
        new TransitionsOutputFormatterCallback(
            eventHandler,
            cqueryOptions,
            out,
            skyframeExecutor,
            accessor,
            hostConfiguration,
            trimmingTransitionFactory),
        new ProtoOutputFormatterCallback(
            eventHandler,
            cqueryOptions,
            out,
            skyframeExecutor,
            accessor,
            aspectResolver,
            OutputType.BINARY),
        new ProtoOutputFormatterCallback(
            eventHandler,
            cqueryOptions,
            out,
            skyframeExecutor,
            accessor,
            aspectResolver,
            OutputType.TEXT),
        new ProtoOutputFormatterCallback(
            eventHandler,
            cqueryOptions,
            out,
            skyframeExecutor,
            accessor,
            aspectResolver,
            OutputType.JSON),
        new BuildOutputFormatterCallback(
            eventHandler, cqueryOptions, out, skyframeExecutor, accessor));
  }

  @Override
  public String getOutputFormat() {
    return cqueryOptions.outputFormat;
  }

  @Override
  public ConfiguredTargetAccessor getAccessor() {
    return accessor;
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
    }
    AsyncFunction<TargetParsingException, Void> reportBuildFileErrorAsyncFunction =
        exn -> {
          reportBuildFileError(owner, exn.getMessage());
          return Futures.immediateFuture(null);
        };

    try {
      return QueryTaskFutureImpl.ofDelegate(
          Futures.catchingAsync(
              patternToEval.evalAdaptedForAsync(
                  resolver,
                  getIgnoredPackagePrefixesPathFragments(),
                  /* excludedSubdirectories= */ ImmutableSet.of(),
                  (Callback<Target>)
                      partialResult -> {
                        List<ConfiguredTarget> transformedResult = new ArrayList<>();
                        for (Target target : partialResult) {
                          transformedResult.addAll(getConfiguredTargets(target.getLabel()));
                        }
                        callback.process(transformedResult);
                      },
                  QueryException.class),
              TargetParsingException.class,
              reportBuildFileErrorAsyncFunction,
              MoreExecutors.directExecutor()));
    } catch (InterruptedException e) {
      return immediateCancelledFuture();
    }
  }

  /**
   * Returns the {@link ConfiguredTarget} for the given label and configuration if it exists, else
   * null.
   */
  @Nullable
  private ConfiguredTarget getConfiguredTarget(Label label, BuildConfiguration configuration)
      throws InterruptedException {
    return getValueFromKey(
        ConfiguredTargetKey.builder().setLabel(label).setConfiguration(configuration).build());
  }

  @Override
  @Nullable
  protected ConfiguredTarget getValueFromKey(SkyKey key) throws InterruptedException {
    ConfiguredTargetValue value = getConfiguredTargetValue(key);
    return value == null ? null : value.getConfiguredTarget();
  }

  /**
   * Returns all configured targets in Skyframe with the given label.
   *
   * <p>If there are no matches, returns an empty list.
   */
  private List<ConfiguredTarget> getConfiguredTargets(Label label) throws InterruptedException {
    ImmutableList.Builder<ConfiguredTarget> ans = ImmutableList.builder();
    for (BuildConfiguration config : transitiveConfigurations.values()) {
      ConfiguredTarget ct = getConfiguredTarget(label, config);
      if (ct != null) {
        ans.add(ct);
      }
    }
    ConfiguredTarget nullConfiguredTarget = getNullConfiguredTarget(label);
    if (nullConfiguredTarget != null) {
      ans.add(nullConfiguredTarget);
    }
    return ans.build();
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
    return () -> {
      List<ConfiguredTarget> transformedResult = new ArrayList<>();
      boolean userFriendlyConfigName = true;
      for (ConfiguredTarget target : targets) {
        Label label = getCorrectLabel(target);
        ConfiguredTarget configuredTarget;
        switch (configuration) {
          case "host":
            configuredTarget = getHostConfiguredTarget(label);
            break;
          case "target":
            configuredTarget = getTargetConfiguredTarget(label);
            break;
          case "null":
            configuredTarget = getNullConfiguredTarget(label);
            break;
          default:
            BuildConfiguration config = transitiveConfigurations.get(configuration);
            if (config != null) {
              configuredTarget = getConfiguredTarget(label, config);
              userFriendlyConfigName = false;
              break;
            }
            throw new QueryException(
                "Unknown value '"
                    + configuration
                    + "'. The second argument of config() must be 'target', 'host', 'null', or a"
                    + " valid configuration hash (i.e. one of the outputs of 'blaze config')");
        }
        if (configuredTarget != null) {
          transformedResult.add(configuredTarget);
        }
      }
      if (transformedResult.isEmpty()) {
        throw new QueryException(
            String.format(
                "No target (in) %s could be found in the %s",
                pattern,
                userFriendlyConfigName
                    ? "'" + configuration + "' configuration"
                    : "configuration with checksum '" + configuration + "'"));
      }
      callback.process(transformedResult);
      return null;
    };
  }

  /**
   * This method has to exist because {@link AliasConfiguredTarget#getLabel()} returns the label of
   * the "actual" target instead of the alias target. Grr.
   */
  @Override
  public Label getCorrectLabel(ConfiguredTarget target) {
    // Dereference any aliases that might be present.
    return target.getOriginalLabel();
  }

  @Nullable
  @Override
  protected ConfiguredTarget getHostConfiguredTarget(Label label) throws InterruptedException {
    return getConfiguredTarget(label, hostConfiguration);
  }

  @Nullable
  @Override
  protected ConfiguredTarget getTargetConfiguredTarget(Label label) throws InterruptedException {
    if (topLevelConfigurations.isTopLevelTarget(label)) {
      return getConfiguredTarget(
          label, topLevelConfigurations.getConfigurationForTopLevelTarget(label));
    } else {
      ConfiguredTarget toReturn;
      for (BuildConfiguration configuration : topLevelConfigurations.getConfigurations()) {
        toReturn = getConfiguredTarget(label, configuration);
        if (toReturn != null) {
          return toReturn;
        }
      }
      return null;
    }
  }

  @Nullable
  @Override
  protected ConfiguredTarget getNullConfiguredTarget(Label label) throws InterruptedException {
    return getConfiguredTarget(label, null);
  }

  @Nullable
  @Override
  protected RuleConfiguredTarget getRuleConfiguredTarget(ConfiguredTarget configuredTarget) {
    if (configuredTarget instanceof RuleConfiguredTarget) {
      return (RuleConfiguredTarget) configuredTarget;
    }
    return null;
  }

  @Nullable
  @Override
  protected BuildConfiguration getConfiguration(ConfiguredTarget target) {
    try {
      return target.getConfigurationKey() == null
          ? null
          : ((BuildConfigurationValue) graph.getValue(target.getConfigurationKey()))
              .getConfiguration();
    } catch (InterruptedException e) {
      throw new IllegalStateException("Unexpected interruption during configured target query", e);
    }
  }

  @Override
  protected ConfiguredTargetKey getSkyKey(ConfiguredTarget target) {
    return ConfiguredTargetKey.builder()
        .setConfiguredTarget(target)
        .setConfiguration(getConfiguration(target))
        .build();
  }

  @Override
  public ThreadSafeMutableSet<ConfiguredTarget> createThreadSafeMutableSet() {
    return new ThreadSafeMutableKeyExtractorBackedSetImpl<>(
        configuredTargetKeyExtractor,
        ConfiguredTarget.class,
        SkyQueryEnvironment.DEFAULT_THREAD_COUNT);
  }
}
