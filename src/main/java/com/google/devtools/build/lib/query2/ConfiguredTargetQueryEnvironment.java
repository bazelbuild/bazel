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
import com.google.common.util.concurrent.AsyncFunction;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.cmdline.TargetPattern;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.packages.RuleTransitionFactory;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.pkgcache.PackageManager;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.query2.ProtoOutputFormatterCallback.OutputType;
import com.google.devtools.build.lib.query2.engine.Callback;
import com.google.devtools.build.lib.query2.engine.KeyExtractor;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment;
import com.google.devtools.build.lib.query2.engine.QueryException;
import com.google.devtools.build.lib.query2.engine.QueryExpression;
import com.google.devtools.build.lib.query2.engine.QueryUtil.ThreadSafeMutableKeyExtractorBackedSetImpl;
import com.google.devtools.build.lib.query2.output.AspectResolver;
import com.google.devtools.build.lib.query2.output.CqueryOptions;
import com.google.devtools.build.lib.rules.AliasConfiguredTarget;
import com.google.devtools.build.lib.skyframe.BuildConfigurationValue;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetValue;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.WalkableGraph;
import java.io.OutputStream;
import java.util.ArrayList;
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
      String parserPrefix,
      PathPackageLocator pkgPath,
      Supplier<WalkableGraph> walkableGraphSupplier,
      Set<Setting> settings) {
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
        element -> {
          try {
            return ConfiguredTargetKey.of(
                element,
                element.getConfigurationKey() == null
                    ? null
                    : ((BuildConfigurationValue) graph.getValue(element.getConfigurationKey()))
                        .getConfiguration());
          } catch (InterruptedException e) {
            throw new IllegalStateException("Interruption unexpected in configured query", e);
          }
        };
  }

  public ConfiguredTargetQueryEnvironment(
      boolean keepGoing,
      ExtendedEventHandler eventHandler,
      Iterable<QueryFunction> extraFunctions,
      TopLevelConfigurations topLevelConfigurations,
      BuildConfiguration hostConfiguration,
      String parserPrefix,
      PathPackageLocator pkgPath,
      Supplier<WalkableGraph> walkableGraphSupplier,
      CqueryOptions cqueryOptions) {
    this(
        keepGoing,
        eventHandler,
        extraFunctions,
        topLevelConfigurations,
        hostConfiguration,
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

  @Override
  public ImmutableList<NamedThreadSafeOutputFormatterCallback<ConfiguredTarget>>
      getDefaultOutputFormatters(
          TargetAccessor<ConfiguredTarget> accessor,
          ExtendedEventHandler eventHandler,
          OutputStream out,
          SkyframeExecutor skyframeExecutor,
          BuildConfiguration hostConfiguration,
          @Nullable RuleTransitionFactory trimmingTransitionFactory,
          PackageManager packageManager) {
    AspectResolver aspectResolver =
        cqueryOptions.aspectDeps.createResolver(packageManager, eventHandler);
    return new ImmutableList.Builder<NamedThreadSafeOutputFormatterCallback<ConfiguredTarget>>()
        .add(
            new LabelAndConfigurationOutputFormatterCallback(
                eventHandler, cqueryOptions, out, skyframeExecutor, accessor))
        .add(
            new TransitionsOutputFormatterCallback(
                eventHandler,
                cqueryOptions,
                out,
                skyframeExecutor,
                accessor,
                hostConfiguration,
                trimmingTransitionFactory))
        .add(
            new ProtoOutputFormatterCallback(
                eventHandler,
                cqueryOptions,
                out,
                skyframeExecutor,
                accessor,
                aspectResolver,
                OutputType.BINARY))
        .add(
            new ProtoOutputFormatterCallback(
                eventHandler,
                cqueryOptions,
                out,
                skyframeExecutor,
                accessor,
                aspectResolver,
                OutputType.TEXT))
        .build();
  }

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

  @Override
  @Nullable
  protected ConfiguredTarget getValueFromKey(SkyKey key) throws InterruptedException {
    ConfiguredTargetValue value = getConfiguredTargetValue(key);
    return value == null ? null : value.getConfiguredTarget();
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

  /**
   * This method has to exist because {@link AliasConfiguredTarget#getLabel()} returns the label of
   * the "actual" target instead of the alias target. Grr.
   */
  @Override
  public Label getCorrectLabel(ConfiguredTarget target) {
    if (target instanceof AliasConfiguredTarget) {
      return ((AliasConfiguredTarget) target).getOriginalLabel();
    }
    return target.getLabel();
  }

  @Nullable
  @Override
  protected ConfiguredTarget getHostConfiguredTarget(Label label) throws InterruptedException {
    return getValueFromKey(ConfiguredTargetValue.key(label, hostConfiguration));
  }

  @Nullable
  @Override
  protected ConfiguredTarget getTargetConfiguredTarget(Label label) throws InterruptedException {
    if (topLevelConfigurations.isTopLevelTarget(label)) {
      return getValueFromKey(
          ConfiguredTargetValue.key(
              label, topLevelConfigurations.getConfigurationForTopLevelTarget(label)));
    } else {
      ConfiguredTarget toReturn;
      for (BuildConfiguration configuration : topLevelConfigurations.getConfigurations()) {
        toReturn = getValueFromKey(ConfiguredTargetValue.key(label, configuration));
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
    return getValueFromKey(ConfiguredTargetValue.key(label, null));
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
    return ConfiguredTargetKey.of(target, getConfiguration(target));
  }

  @Override
  public ThreadSafeMutableSet<ConfiguredTarget> createThreadSafeMutableSet() {
    return new ThreadSafeMutableKeyExtractorBackedSetImpl<>(
        configuredTargetKeyExtractor,
        ConfiguredTarget.class,
        SkyQueryEnvironment.DEFAULT_THREAD_COUNT);
  }
}

