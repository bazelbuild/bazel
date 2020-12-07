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
package com.google.devtools.build.lib.query2.aquery;

import com.google.common.collect.ImmutableList;
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
import com.google.devtools.build.lib.query2.aquery.ActionGraphProtoOutputFormatterCallback.OutputType;
import com.google.devtools.build.lib.query2.engine.Callback;
import com.google.devtools.build.lib.query2.engine.InputsFunction;
import com.google.devtools.build.lib.query2.engine.KeyExtractor;
import com.google.devtools.build.lib.query2.engine.MnemonicFunction;
import com.google.devtools.build.lib.query2.engine.OutputsFunction;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment;
import com.google.devtools.build.lib.query2.engine.QueryException;
import com.google.devtools.build.lib.query2.engine.QueryExpression;
import com.google.devtools.build.lib.query2.engine.QueryUtil.ThreadSafeMutableKeyExtractorBackedSetImpl;
import com.google.devtools.build.lib.skyframe.BuildConfigurationValue;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetValue;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import com.google.devtools.build.lib.skyframe.actiongraph.v2.StreamedOutputHandler;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.WalkableGraph;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.function.Supplier;
import javax.annotation.Nullable;

/**
 * {@link QueryEnvironment} that is specialized for running action graph queries over the configured
 * target graph.
 */
public class ActionGraphQueryEnvironment
    extends PostAnalysisQueryEnvironment<ConfiguredTargetValue> {

  public static final ImmutableList<QueryFunction> AQUERY_FUNCTIONS = populateAqueryFunctions();
  public static final ImmutableList<QueryFunction> FUNCTIONS = populateFunctions();
  AqueryOptions aqueryOptions;

  private AqueryActionFilter actionFilters;
  private final KeyExtractor<ConfiguredTargetValue, ConfiguredTargetKey>
      configuredTargetKeyExtractor;
  private final ConfiguredTargetValueAccessor accessor;

  public ActionGraphQueryEnvironment(
      boolean keepGoing,
      ExtendedEventHandler eventHandler,
      Iterable<QueryFunction> extraFunctions,
      TopLevelConfigurations topLevelConfigurations,
      BuildConfiguration hostConfiguration,
      PathFragment parserPrefix,
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
    this.configuredTargetKeyExtractor =
        configuredTargetValue ->
            ConfiguredTargetKey.builder()
                .setConfiguredTarget(configuredTargetValue.getConfiguredTarget())
                .build();
    this.accessor =
        new ConfiguredTargetValueAccessor(
            walkableGraphSupplier.get(), this::getTarget, this.configuredTargetKeyExtractor);
  }

  public ActionGraphQueryEnvironment(
      boolean keepGoing,
      ExtendedEventHandler eventHandler,
      Iterable<QueryFunction> extraFunctions,
      TopLevelConfigurations topLevelConfigurations,
      BuildConfiguration hostConfiguration,
      PathFragment parserPrefix,
      PathPackageLocator pkgPath,
      Supplier<WalkableGraph> walkableGraphSupplier,
      AqueryOptions aqueryOptions) {
    this(
        keepGoing,
        eventHandler,
        extraFunctions,
        topLevelConfigurations,
        hostConfiguration,
        parserPrefix,
        pkgPath,
        walkableGraphSupplier,
        aqueryOptions.toSettings());
    this.aqueryOptions = aqueryOptions;
  }

  private static ImmutableList<QueryFunction> populateFunctions() {
    return ImmutableList.copyOf(QueryEnvironment.DEFAULT_QUERY_FUNCTIONS);
  }

  private static ImmutableList<QueryFunction> populateAqueryFunctions() {
    return ImmutableList.of(new InputsFunction(), new OutputsFunction(), new MnemonicFunction());
  }

  @Override
  public ConfiguredTargetValueAccessor getAccessor() {
    return accessor;
  }

  @Override
  public ImmutableList<NamedThreadSafeOutputFormatterCallback<ConfiguredTargetValue>>
      getDefaultOutputFormatters(
          TargetAccessor<ConfiguredTargetValue> accessor,
          ExtendedEventHandler eventHandler,
          OutputStream out,
          SkyframeExecutor skyframeExecutor,
          BuildConfiguration hostConfiguration,
          @Nullable TransitionFactory<Rule> trimmingTransitionFactory,
          PackageManager packageManager) {
    return aqueryOptions.protoV2
        ? ImmutableList.of(
            new ActionGraphProtoV2OutputFormatterCallback(
                eventHandler,
                aqueryOptions,
                out,
                skyframeExecutor,
                accessor,
                StreamedOutputHandler.OutputType.BINARY,
                actionFilters),
            new ActionGraphProtoV2OutputFormatterCallback(
                eventHandler,
                aqueryOptions,
                out,
                skyframeExecutor,
                accessor,
                StreamedOutputHandler.OutputType.TEXT,
                actionFilters),
            new ActionGraphProtoV2OutputFormatterCallback(
                eventHandler,
                aqueryOptions,
                out,
                skyframeExecutor,
                accessor,
                StreamedOutputHandler.OutputType.JSON,
                actionFilters),
            new ActionGraphTextOutputFormatterCallback(
                eventHandler, aqueryOptions, out, skyframeExecutor, accessor, actionFilters))
        : ImmutableList.of(
            new ActionGraphProtoOutputFormatterCallback(
                eventHandler,
                aqueryOptions,
                out,
                skyframeExecutor,
                accessor,
                OutputType.BINARY,
                actionFilters),
            new ActionGraphProtoOutputFormatterCallback(
                eventHandler,
                aqueryOptions,
                out,
                skyframeExecutor,
                accessor,
                OutputType.TEXT,
                actionFilters),
            new ActionGraphProtoOutputFormatterCallback(
                eventHandler,
                aqueryOptions,
                out,
                skyframeExecutor,
                accessor,
                OutputType.JSON,
                actionFilters),
            new ActionGraphTextOutputFormatterCallback(
                eventHandler, aqueryOptions, out, skyframeExecutor, accessor, actionFilters));
  }

  @Override
  public String getOutputFormat() {
    return aqueryOptions.outputFormat;
  }

  @Override
  protected KeyExtractor<ConfiguredTargetValue, ConfiguredTargetKey>
      getConfiguredTargetKeyExtractor() {
    return configuredTargetKeyExtractor;
  }

  @Override
  public Label getCorrectLabel(ConfiguredTargetValue configuredTargetValue) {
    ConfiguredTarget target = configuredTargetValue.getConfiguredTarget();
    // Dereference any aliases that might be present.
    return target.getOriginalLabel();
  }

  @Nullable
  @Override
  protected ConfiguredTargetValue getHostConfiguredTarget(Label label) throws InterruptedException {
    return this.getConfiguredTargetValue(
        ConfiguredTargetKey.builder().setLabel(label).setConfiguration(hostConfiguration).build());
  }

  @Nullable
  @Override
  protected ConfiguredTargetValue getTargetConfiguredTarget(Label label)
      throws InterruptedException {
    if (topLevelConfigurations.isTopLevelTarget(label)) {
      return this.getConfiguredTargetValue(
          ConfiguredTargetKey.builder()
              .setLabel(label)
              .setConfiguration(topLevelConfigurations.getConfigurationForTopLevelTarget(label))
              .build());
    } else {
      ConfiguredTargetValue toReturn;
      for (BuildConfiguration configuration : topLevelConfigurations.getConfigurations()) {
        toReturn =
            this.getConfiguredTargetValue(
                ConfiguredTargetKey.builder()
                    .setLabel(label)
                    .setConfiguration(configuration)
                    .build());
        if (toReturn != null) {
          return toReturn;
        }
      }
      return null;
    }
  }

  @Nullable
  @Override
  protected ConfiguredTargetValue getNullConfiguredTarget(Label label) throws InterruptedException {
    return this.getConfiguredTargetValue(ConfiguredTargetKey.builder().setLabel(label).build());
  }

  @Nullable
  @Override
  protected ConfiguredTargetValue getValueFromKey(SkyKey key) throws InterruptedException {
    return getConfiguredTargetValue(key);
  }

  @Nullable
  @Override
  protected RuleConfiguredTarget getRuleConfiguredTarget(
      ConfiguredTargetValue configuredTargetValue) {
    ConfiguredTarget configuredTarget = configuredTargetValue.getConfiguredTarget();
    if (configuredTarget instanceof RuleConfiguredTarget) {
      return (RuleConfiguredTarget) configuredTarget;
    }
    return null;
  }

  @Nullable
  @Override
  protected BuildConfiguration getConfiguration(ConfiguredTargetValue configuredTargetValue) {
    ConfiguredTarget target = configuredTargetValue.getConfiguredTarget();
    try {
      return target.getConfigurationKey() == null
          ? null
          : ((BuildConfigurationValue) graph.getValue(target.getConfigurationKey()))
              .getConfiguration();
    } catch (InterruptedException e) {
      throw new IllegalStateException("Unexpected interruption during aquery", e);
    }
  }

  @Override
  protected ConfiguredTargetKey getSkyKey(ConfiguredTargetValue configuredTargetValue) {
    ConfiguredTarget target = configuredTargetValue.getConfiguredTarget();
    return ConfiguredTargetKey.builder()
        .setConfiguredTarget(target)
        .setConfiguration(getConfiguration(configuredTargetValue))
        .build();
  }

  @Override
  public QueryTaskFuture<Void> getTargetsMatchingPattern(
      QueryExpression owner, String pattern, Callback<ConfiguredTargetValue> callback) {
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
    try {
      return QueryTaskFutureImpl.ofDelegate(
          Futures.catchingAsync(
              patternToEval.evalAdaptedForAsync(
                  resolver,
                  getIgnoredPackagePrefixesPathFragments(),
                  /* excludedSubdirectories= */ ImmutableSet.of(),
                  (Callback<Target>)
                      partialResult -> {
                        List<ConfiguredTargetValue> transformedResult = new ArrayList<>();
                        for (Target target : partialResult) {
                          ConfiguredTargetValue configuredTargetValue =
                              getConfiguredTargetValue(target.getLabel());
                          if (configuredTargetValue != null) {
                            transformedResult.add(configuredTargetValue);
                          }
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

  private ConfiguredTargetValue getConfiguredTargetValue(Label label) throws InterruptedException {
    // Try with target configuration.
    ConfiguredTargetValue configuredTargetValue = getTargetConfiguredTarget(label);
    if (configuredTargetValue != null) {
      return configuredTargetValue;
    }
    // Try with host configuration (even when --notool_deps is set in the case that top-level
    // targets are configured in the host configuration so we are doing a host-configuration-only
    // query).
    configuredTargetValue = getHostConfiguredTarget(label);
    if (configuredTargetValue != null) {
      return configuredTargetValue;
    }
    // Last chance: source file.
    return getNullConfiguredTarget(label);
  }

  @Override
  public ThreadSafeMutableSet<ConfiguredTargetValue> createThreadSafeMutableSet() {
    return new ThreadSafeMutableKeyExtractorBackedSetImpl<>(
        configuredTargetKeyExtractor,
        ConfiguredTargetValue.class,
        SkyQueryEnvironment.DEFAULT_THREAD_COUNT);
  }

  public void setActionFilters(AqueryActionFilter actionFilters) {
    this.actionFilters = actionFilters;
  }
}
