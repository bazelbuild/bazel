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

import static com.google.common.collect.ImmutableMap.toImmutableMap;

import com.google.common.base.Joiner;
import com.google.common.base.Verify;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.util.concurrent.AsyncFunction;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.ConfiguredTargetValue;
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
import com.google.devtools.build.lib.server.FailureDetails.ConfigurableQuery;
import com.google.devtools.build.lib.skyframe.BuildConfigurationValue;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.WalkableGraph;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.List;
import java.util.Set;
import java.util.function.Function;
import java.util.function.Supplier;
import javax.annotation.Nullable;

/**
 * {@link QueryEnvironment} that runs queries over the configured target (analysis) graph.
 *
 * <p>Aspects are partially supported. Their dependencies appear as implicit dependencies on the
 * targets they're connected to, but the aspects themselves aren't visible as query nodes. See
 * comments on {@link PostAnalysisQueryEnvironment#targetifyValues} and b/163052263 for details.
 */
public class ConfiguredTargetQueryEnvironment
    extends PostAnalysisQueryEnvironment<KeyedConfiguredTarget> {
  /** Common query functions and cquery specific functions. */
  public static final ImmutableList<QueryFunction> FUNCTIONS = populateFunctions();
  /** Cquery specific functions. */
  public static final ImmutableList<QueryFunction> CQUERY_FUNCTIONS = getCqueryFunctions();

  private CqueryOptions cqueryOptions;

  private final KeyExtractor<KeyedConfiguredTarget, ConfiguredTargetKey>
      configuredTargetKeyExtractor;

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
  protected KeyExtractor<KeyedConfiguredTarget, ConfiguredTargetKey>
      getConfiguredTargetKeyExtractor() {
    return configuredTargetKeyExtractor;
  }

  public ConfiguredTargetQueryEnvironment(
      boolean keepGoing,
      ExtendedEventHandler eventHandler,
      Iterable<QueryFunction> extraFunctions,
      TopLevelConfigurations topLevelConfigurations,
      BuildConfiguration hostConfiguration,
      Collection<SkyKey> transitiveConfigurationKeys,
      PathFragment parserPrefix,
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
    this.configuredTargetKeyExtractor = KeyedConfiguredTarget::getConfiguredTargetKey;
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
      PathFragment parserPrefix,
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

  /**
   * Return supplied BuildConfiguration if both are equal else throw exception.
   *
   * <p>Noting the background of {@link BuildConfigurationValue.Key::toComparableString}, multiple
   * BuildConfigurationValue.Key can correspond to the same BuildConfiguration, especially when
   * trimming is involved.
   *
   * <p>Note that, in {@link getTransitiveConfigurations}, only interested in the values and
   * throwing away the Keys. Thus, intricacies around Key fragments and options diverging not as
   * relevant anyway.
   */
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
    // mergeEqualBuildConfiguration can only fail if two BuildConfiguration have the same
    // checksum but are not equal. This would be a black swan event.
    return graph.getSuccessfulValues(transitiveConfigurationKeys).values().stream()
        .map(value -> (BuildConfigurationValue) value)
        .map(BuildConfigurationValue::getConfiguration)
        .sorted(Comparator.comparing(BuildConfiguration::checksum))
        .collect(
            toImmutableMap(
                BuildConfiguration::checksum,
                Function.identity(),
                ConfiguredTargetQueryEnvironment::mergeEqualBuildConfiguration));
  }

  @Override
  public ImmutableList<NamedThreadSafeOutputFormatterCallback<KeyedConfiguredTarget>>
      getDefaultOutputFormatters(
          TargetAccessor<KeyedConfiguredTarget> accessor,
          ExtendedEventHandler eventHandler,
          OutputStream out,
          SkyframeExecutor skyframeExecutor,
          BuildConfiguration hostConfiguration,
          @Nullable TransitionFactory<Rule> trimmingTransitionFactory,
          PackageManager packageManager)
          throws QueryException, InterruptedException {
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
            eventHandler, cqueryOptions, out, skyframeExecutor, accessor),
        new GraphOutputFormatterCallback(
            eventHandler,
            cqueryOptions,
            out,
            skyframeExecutor,
            accessor,
            kct -> getFwdDeps(ImmutableList.of(kct))),
        new StarlarkOutputFormatterCallback(
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
      QueryExpression owner, String pattern, Callback<KeyedConfiguredTarget> callback) {
    TargetPattern patternToEval;
    try {
      patternToEval = getPattern(pattern);
    } catch (TargetParsingException tpe) {
      try {
        handleError(owner, tpe.getMessage(), tpe.getDetailedExitCode());
      } catch (QueryException qe) {
        return immediateFailedFuture(qe);
      }
      return immediateSuccessfulFuture(null);
    }
    AsyncFunction<TargetParsingException, Void> reportBuildFileErrorAsyncFunction =
        exn -> {
          handleError(owner, exn.getMessage(), exn.getDetailedExitCode());
          return Futures.immediateFuture(null);
        };

    return QueryTaskFutureImpl.ofDelegate(
        Futures.catchingAsync(
            patternToEval.evalAdaptedForAsync(
                resolver,
                getIgnoredPackagePrefixesPathFragments(),
                /* excludedSubdirectories= */ ImmutableSet.of(),
                (Callback<Target>)
                    partialResult -> {
                      List<KeyedConfiguredTarget> transformedResult = new ArrayList<>();
                      for (Target target : partialResult) {
                        transformedResult.addAll(
                            getConfiguredTargetsForConfigFunction(target.getLabel()));
                      }
                      callback.process(transformedResult);
                    },
                QueryException.class),
            TargetParsingException.class,
            reportBuildFileErrorAsyncFunction,
            MoreExecutors.directExecutor()));
  }

  /**
   * Returns the {@link ConfiguredTarget} for the given label and configuration if it exists, else
   * null.
   */
  @Nullable
  private KeyedConfiguredTarget getConfiguredTarget(Label label, BuildConfiguration configuration)
      throws InterruptedException {
    return getValueFromKey(
        ConfiguredTargetKey.builder().setLabel(label).setConfiguration(configuration).build());
  }

  @Override
  @Nullable
  protected KeyedConfiguredTarget getValueFromKey(SkyKey key) throws InterruptedException {
    ConfiguredTargetValue value = getConfiguredTargetValue(key);
    return value == null
        ? null
        : KeyedConfiguredTarget.create((ConfiguredTargetKey) key, value.getConfiguredTarget());
  }

  /**
   * Returns all configured targets in Skyframe with the given label.
   *
   * <p>If there are no matches, returns an empty list.
   */
  private List<KeyedConfiguredTarget> getConfiguredTargetsForConfigFunction(Label label)
      throws InterruptedException {
    ImmutableList.Builder<KeyedConfiguredTarget> ans = ImmutableList.builder();
    for (BuildConfiguration config : transitiveConfigurations.values()) {
      KeyedConfiguredTarget kct = getConfiguredTarget(label, config);
      if (kct != null) {
        ans.add(kct);
      }
    }
    KeyedConfiguredTarget nullConfiguredTarget = getNullConfiguredTarget(label);
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
   * @param configPrefix the configuration to request {@code targets} in. This can be the
   *     configuration's checksum, any prefix of its checksum, or the special identifiers "host",
   *     "target", or "null".
   * @param callback the callback to receive the results of this method.
   * @return {@link QueryTaskCallable} that returns the correctly configured targets.
   */
  QueryTaskCallable<Void> getConfiguredTargetsForConfigFunction(
      String pattern,
      ThreadSafeMutableSet<KeyedConfiguredTarget> targets,
      String configPrefix,
      Callback<KeyedConfiguredTarget> callback) {
    // There's no technical reason other callers beside ConfigFunction can't call this. But they'd
    // need to adjust the error messaging below to not make it config()-specific. Please don't just
    // remove that line: the counter-priority is making error messages as clear, precise, and
    // actionable as possible.
    return () -> {
      List<KeyedConfiguredTarget> transformedResult = new ArrayList<>();
      boolean userFriendlyConfigName = true;
      for (KeyedConfiguredTarget target : targets) {
        Label label = getCorrectLabel(target);
        KeyedConfiguredTarget keyedConfiguredTarget;
        switch (configPrefix) {
          case "host":
            keyedConfiguredTarget = getHostConfiguredTarget(label);
            break;
          case "target":
            keyedConfiguredTarget = getTargetConfiguredTarget(label);
            break;
          case "null":
            keyedConfiguredTarget = getNullConfiguredTarget(label);
            break;
          default:
            ImmutableList<String> matchingConfigs =
                transitiveConfigurations.keySet().stream()
                    .filter(fullConfig -> fullConfig.startsWith(configPrefix))
                    .collect(ImmutableList.toImmutableList());
            if (matchingConfigs.size() == 1) {
              keyedConfiguredTarget =
                  getConfiguredTarget(
                      label,
                      Verify.verifyNotNull(transitiveConfigurations.get(matchingConfigs.get(0))));
              userFriendlyConfigName = false;
            } else if (matchingConfigs.size() >= 2) {
              throw new QueryException(
                  String.format(
                      "Configuration ID '%s' is ambiguous.\n"
                          + "'%s' is a prefix of multiple configurations:\n "
                          + Joiner.on("\n ").join(matchingConfigs)
                          + "\n\n"
                          + "Use a longer prefix to uniquely identify one configuration.",
                      configPrefix,
                      configPrefix),
                  ConfigurableQuery.Code.INCORRECT_CONFIG_ARGUMENT_ERROR);
            } else {
              throw new QueryException(
                  String.format("Unknown configuration ID '%s'.\n", configPrefix)
                      + "config()'s second argument must identify a unique configuration.\n"
                      + "\n"
                      + "Valid values:\n"
                      + " 'target' for the default configuration\n"
                      + " 'host' for the host configuration\n"
                      + " 'null' for source files (which have no configuration)\n"
                      + " an arbitrary configuration's full or short ID\n"
                      + "\n"
                      + "A short ID is any prefix of a full ID. cquery shows short IDs. 'bazel "
                      + "config' shows full IDs.\n"
                      + "\n"
                      + "For more help, see https://docs.bazel.build/cquery.html.",
                  ConfigurableQuery.Code.INCORRECT_CONFIG_ARGUMENT_ERROR);
            }
        }
        if (keyedConfiguredTarget != null) {
          transformedResult.add(keyedConfiguredTarget);
        }
      }
      if (transformedResult.isEmpty()) {
        throw new QueryException(
            String.format(
                "No target (in) %s could be found in the %s",
                pattern,
                userFriendlyConfigName
                    ? "'" + configPrefix + "' configuration"
                    : "configuration with checksum '" + configPrefix + "'"),
            ConfigurableQuery.Code.TARGET_MISSING);
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
  public Label getCorrectLabel(KeyedConfiguredTarget target) {
    // Dereference any aliases that might be present.
    return target.getConfiguredTarget().getOriginalLabel();
  }

  @Nullable
  @Override
  protected KeyedConfiguredTarget getHostConfiguredTarget(Label label) throws InterruptedException {
    return getConfiguredTarget(label, hostConfiguration);
  }

  @Nullable
  @Override
  protected KeyedConfiguredTarget getTargetConfiguredTarget(Label label)
      throws InterruptedException {
    if (topLevelConfigurations.isTopLevelTarget(label)) {
      return getConfiguredTarget(
          label, topLevelConfigurations.getConfigurationForTopLevelTarget(label));
    } else {
      KeyedConfiguredTarget toReturn;
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
  protected KeyedConfiguredTarget getNullConfiguredTarget(Label label) throws InterruptedException {
    return getConfiguredTarget(label, null);
  }

  @Nullable
  @Override
  protected RuleConfiguredTarget getRuleConfiguredTarget(KeyedConfiguredTarget configuredTarget) {
    if (configuredTarget.getConfiguredTarget() instanceof RuleConfiguredTarget) {
      return (RuleConfiguredTarget) configuredTarget.getConfiguredTarget();
    }
    return null;
  }

  @Nullable
  @Override
  protected BuildConfiguration getConfiguration(KeyedConfiguredTarget target) {
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
  protected ConfiguredTargetKey getSkyKey(KeyedConfiguredTarget target) {
    return target.getConfiguredTargetKey();
  }

  @Override
  public ThreadSafeMutableSet<KeyedConfiguredTarget> createThreadSafeMutableSet() {
    return new ThreadSafeMutableKeyExtractorBackedSetImpl<>(
        configuredTargetKeyExtractor,
        KeyedConfiguredTarget.class,
        SkyQueryEnvironment.DEFAULT_THREAD_COUNT);
  }
}
