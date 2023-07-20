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
import com.google.devtools.build.lib.analysis.TopLevelArtifactContext;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.cmdline.TargetPattern;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.packages.RuleClassProvider;
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
import com.google.devtools.build.lib.skyframe.BuildConfigurationKey;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.WalkableGraph;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;
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
    extends PostAnalysisQueryEnvironment<ConfiguredTarget> {
  /** Common query functions and cquery specific functions. */
  public static final ImmutableList<QueryFunction> FUNCTIONS = populateFunctions();
  /** Cquery specific functions. */
  public static final ImmutableList<QueryFunction> CQUERY_FUNCTIONS = getCqueryFunctions();

  private CqueryOptions cqueryOptions;

  private final TopLevelArtifactContext topLevelArtifactContext;

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
   * particularly including configurations that aren't the top-level.
   *
   * <p>This can also be used in cquery's {@code config} function to match against explicitly
   * specified configs. This, in particular, is where having user-friendly hashes is invaluable.
   */
  private final ImmutableMap<String, BuildConfigurationValue> transitiveConfigurations;

  @Override
  protected KeyExtractor<ConfiguredTarget, ConfiguredTargetKey> getConfiguredTargetKeyExtractor() {
    return configuredTargetKeyExtractor;
  }

  public ConfiguredTargetQueryEnvironment(
      boolean keepGoing,
      ExtendedEventHandler eventHandler,
      Iterable<QueryFunction> extraFunctions,
      TopLevelConfigurations topLevelConfigurations,
      Collection<SkyKey> transitiveConfigurationKeys,
      TargetPattern.Parser mainRepoTargetParser,
      PathPackageLocator pkgPath,
      Supplier<WalkableGraph> walkableGraphSupplier,
      Set<Setting> settings,
      TopLevelArtifactContext topLevelArtifactContext)
      throws InterruptedException {
    super(
        keepGoing,
        eventHandler,
        extraFunctions,
        topLevelConfigurations,
        mainRepoTargetParser,
        pkgPath,
        walkableGraphSupplier,
        settings);
    this.accessor = new ConfiguredTargetAccessor(walkableGraphSupplier.get(), this);
    this.configuredTargetKeyExtractor = ConfiguredTargetKey::fromConfiguredTarget;
    this.transitiveConfigurations =
        getTransitiveConfigurations(transitiveConfigurationKeys, walkableGraphSupplier.get());
    this.topLevelArtifactContext = topLevelArtifactContext;
  }

  public ConfiguredTargetQueryEnvironment(
      boolean keepGoing,
      ExtendedEventHandler eventHandler,
      Iterable<QueryFunction> extraFunctions,
      TopLevelConfigurations topLevelConfigurations,
      Collection<SkyKey> transitiveConfigurationKeys,
      TargetPattern.Parser mainRepoTargetParser,
      PathPackageLocator pkgPath,
      Supplier<WalkableGraph> walkableGraphSupplier,
      CqueryOptions cqueryOptions,
      TopLevelArtifactContext topLevelArtifactContext)
      throws InterruptedException {
    this(
        keepGoing,
        eventHandler,
        extraFunctions,
        topLevelConfigurations,
        transitiveConfigurationKeys,
        mainRepoTargetParser,
        pkgPath,
        walkableGraphSupplier,
        cqueryOptions.toSettings(),
        topLevelArtifactContext);
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

  private static ImmutableMap<String, BuildConfigurationValue> getTransitiveConfigurations(
      Collection<SkyKey> transitiveConfigurationKeys, WalkableGraph graph)
      throws InterruptedException {
    // BuildConfigurationKey and BuildConfigurationValue should be 1:1
    // so merge function intentionally omitted
    return graph.getSuccessfulValues(transitiveConfigurationKeys).values().stream()
        .map(BuildConfigurationValue.class::cast)
        .sorted(Comparator.comparing(BuildConfigurationValue::checksum))
        .collect(toImmutableMap(BuildConfigurationValue::checksum, Function.identity()));
  }

  @Override
  public ImmutableList<NamedThreadSafeOutputFormatterCallback<ConfiguredTarget>>
      getDefaultOutputFormatters(
          TargetAccessor<ConfiguredTarget> accessor,
          ExtendedEventHandler eventHandler,
          OutputStream out,
          SkyframeExecutor skyframeExecutor,
          RuleClassProvider ruleClassProvider,
          PackageManager packageManager)
          throws QueryException, InterruptedException {
    AspectResolver aspectResolver =
        cqueryOptions.aspectDeps.createResolver(packageManager, eventHandler);
    return ImmutableList.of(
        new LabelAndConfigurationOutputFormatterCallback(
            eventHandler,
            cqueryOptions,
            out,
            skyframeExecutor,
            accessor,
            true,
            getMainRepoMapping()),
        new LabelAndConfigurationOutputFormatterCallback(
            eventHandler,
            cqueryOptions,
            out,
            skyframeExecutor,
            accessor,
            false,
            getMainRepoMapping()),
        new TransitionsOutputFormatterCallback(
            eventHandler,
            cqueryOptions,
            out,
            skyframeExecutor,
            accessor,
            ruleClassProvider,
            getMainRepoMapping()),
        new ProtoOutputFormatterCallback(
            eventHandler,
            cqueryOptions,
            out,
            skyframeExecutor,
            accessor,
            aspectResolver,
            OutputType.BINARY,
            ruleClassProvider),
        new ProtoOutputFormatterCallback(
            eventHandler,
            cqueryOptions,
            out,
            skyframeExecutor,
            accessor,
            aspectResolver,
            OutputType.DELIMITED_BINARY,
            ruleClassProvider),
        new ProtoOutputFormatterCallback(
            eventHandler,
            cqueryOptions,
            out,
            skyframeExecutor,
            accessor,
            aspectResolver,
            OutputType.TEXT,
            ruleClassProvider),
        new ProtoOutputFormatterCallback(
            eventHandler,
            cqueryOptions,
            out,
            skyframeExecutor,
            accessor,
            aspectResolver,
            OutputType.JSON,
            ruleClassProvider),
        new BuildOutputFormatterCallback(
            eventHandler, cqueryOptions, out, skyframeExecutor, accessor),
        new GraphOutputFormatterCallback(
            eventHandler,
            cqueryOptions,
            out,
            skyframeExecutor,
            accessor,
            kct -> getFwdDeps(ImmutableList.of(kct)),
            getMainRepoMapping()),
        new StarlarkOutputFormatterCallback(
            eventHandler, cqueryOptions, out, skyframeExecutor, accessor),
        new FilesOutputFormatterCallback(
            eventHandler, cqueryOptions, out, skyframeExecutor, accessor, topLevelArtifactContext));
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
                      List<ConfiguredTarget> transformedResult = new ArrayList<>();
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
  private ConfiguredTarget getConfiguredTarget(
      Label label, @Nullable BuildConfigurationValue configuration) throws InterruptedException {
    BuildConfigurationKey configurationKey = configuration == null ? null : configuration.getKey();
    ConfiguredTarget target =
        getValueFromKey(
            ConfiguredTargetKey.builder()
                .setLabel(label)
                .setConfigurationKey(configurationKey)
                .build());
    // The configurations might not match if the target's configuration changed due to a transition
    // or trimming. Filters such targets.
    if (target == null || !Objects.equals(configurationKey, target.getConfigurationKey())) {
      return null;
    }
    return target;
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
  private ImmutableList<ConfiguredTarget> getConfiguredTargetsForConfigFunction(Label label)
      throws InterruptedException {
    ImmutableList.Builder<ConfiguredTarget> ans = ImmutableList.builder();
    for (BuildConfigurationValue config : transitiveConfigurations.values()) {
      ConfiguredTarget kct = getConfiguredTarget(label, config);
      if (kct != null) {
        ans.add(kct);
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
   * @param targetsFuture the set of {@link ConfiguredTarget}s whose labels represent the targets
   *     being requested.
   * @param configPrefix the configuration to request {@code targets} in. This can be the
   *     configuration's checksum, any prefix of its checksum, or the special identifiers "target"
   *     or "null".
   * @param callback the callback to receive the results of this method.
   * @return {@link QueryTaskCallable} that returns the correctly configured targets.
   */
  @SuppressWarnings("unchecked")
  <T> QueryTaskCallable<Void> getConfiguredTargetsForConfigFunction(
      String pattern,
      QueryTaskFuture<ThreadSafeMutableSet<T>> targetsFuture,
      String configPrefix,
      Callback<ConfiguredTarget> callback) {
    // There's no technical reason other callers beside ConfigFunction can't call this. But they'd
    // need to adjust the error messaging below to not make it config()-specific. Please don't just
    // remove that line: the counter-priority is making error messages as clear, precise, and
    // actionable as possible.
    return () -> {
      ThreadSafeMutableSet<ConfiguredTarget> targets =
          (ThreadSafeMutableSet<ConfiguredTarget>) targetsFuture.getIfSuccessful();
      List<ConfiguredTarget> transformedResult = new ArrayList<>();
      boolean userFriendlyConfigName = true;
      for (ConfiguredTarget target : targets) {
        Label label = getCorrectLabel(target);
        ConfiguredTarget keyedConfiguredTarget;
        switch (configPrefix) {
          case "host":
            throw new QueryException(
                "'host' configuration no longer exists. Use a specific configuration hash instead",
                ConfigurableQuery.Code.INCORRECT_CONFIG_ARGUMENT_ERROR);
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
                      + " 'null' for source files (which have no configuration)\n"
                      + " an arbitrary configuration's full or short ID\n"
                      + "\n"
                      + "A short ID is any prefix of a full ID. cquery shows short IDs. 'bazel "
                      + "config' shows full IDs.\n"
                      + "\n"
                      + "For more help, see https://bazel.build/docs/cquery.",
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
  public Label getCorrectLabel(ConfiguredTarget target) {
    // Dereference any aliases that might be present.
    return target.getOriginalLabel();
  }

  @Nullable
  @Override
  protected ConfiguredTarget getTargetConfiguredTarget(Label label) throws InterruptedException {
    if (topLevelConfigurations.isTopLevelTarget(label)) {
      return getConfiguredTarget(
          label, topLevelConfigurations.getConfigurationForTopLevelTarget(label));
    } else {
      ConfiguredTarget toReturn;
      for (BuildConfigurationValue configuration : topLevelConfigurations.getConfigurations()) {
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
  protected BuildConfigurationValue getConfiguration(ConfiguredTarget target) {
    try {
      return target.getConfigurationKey() == null
          ? null
          : (BuildConfigurationValue) graph.getValue(target.getConfigurationKey());
    } catch (InterruptedException e) {
      throw new IllegalStateException("Unexpected interruption during configured target query", e);
    }
  }

  @Override
  protected ConfiguredTargetKey getConfiguredTargetKey(ConfiguredTarget target) {
    return ConfiguredTargetKey.fromConfiguredTarget(target);
  }

  @Override
  public ThreadSafeMutableSet<ConfiguredTarget> createThreadSafeMutableSet() {
    return new ThreadSafeMutableKeyExtractorBackedSetImpl<>(
        configuredTargetKeyExtractor,
        ConfiguredTarget.class,
        SkyQueryEnvironment.DEFAULT_THREAD_COUNT);
  }
}
