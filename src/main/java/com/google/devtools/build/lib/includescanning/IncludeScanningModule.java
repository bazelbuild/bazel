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
package com.google.devtools.build.lib.includescanning;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Supplier;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionContext;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionMetadata;
import com.google.devtools.build.lib.actions.ActionGraph;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactFactory;
import com.google.devtools.build.lib.actions.ArtifactResolver;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.ExecutionStrategy;
import com.google.devtools.build.lib.actions.ExecutorInitException;
import com.google.devtools.build.lib.analysis.ArtifactsToOwnerLabels;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.buildtool.BuildRequest;
import com.google.devtools.build.lib.concurrent.ExecutorUtil;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadHostile;
import com.google.devtools.build.lib.exec.ActionContextProvider;
import com.google.devtools.build.lib.exec.ExecutorBuilder;
import com.google.devtools.build.lib.includescanning.IncludeParser.GrepIncludesFileType;
import com.google.devtools.build.lib.includescanning.IncludeParser.Inclusion;
import com.google.devtools.build.lib.rules.cpp.CppIncludeExtractionContext;
import com.google.devtools.build.lib.rules.cpp.CppIncludeScanningContext;
import com.google.devtools.build.lib.rules.cpp.IncludeScanner.IncludeScanningHeaderData;
import com.google.devtools.build.lib.rules.cpp.SwigIncludeScanningContext;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.WorkspaceBuilder;
import com.google.devtools.build.lib.skyframe.MutableSupplier;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.common.options.OptionsBase;
import java.io.IOException;
import java.util.Collection;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.FutureTask;
import java.util.logging.Logger;

/**
 * Module that provides implementations of {@link CppIncludeExtractionContext},
 * {@link CppIncludeScanningContext}, and {@link SwigIncludeScanningContext}.
 */
public class IncludeScanningModule extends BlazeModule {
  private static final Logger log = Logger.getLogger(IncludeScanningModule.class.getName());

  private static final PathFragment INCLUDE_HINTS_FILENAME =
      PathFragment.create("tools/cpp/INCLUDE_HINTS");

  private final MutableSupplier<SpawnIncludeScanner> spawnIncludeScannerSupplier =
      new MutableSupplier<>();
  private final MutableSupplier<ArtifactFactory> artifactFactory = new MutableSupplier<>();

  protected PathFragment getIncludeHintsFilename() {
    return INCLUDE_HINTS_FILENAME;
  }

  @Override
  @ThreadHostile
  public void executorInit(CommandEnvironment env, BuildRequest request, ExecutorBuilder builder) {
    builder.addActionContextProvider(
        new IncludeScanningActionContextProvider(env, request, spawnIncludeScannerSupplier));
    builder
        .addStrategyByContext(CppIncludeExtractionContext.class, "")
        .addStrategyByContext(SwigIncludeScanningContext.class, "")
        .addStrategyByContext(CppIncludeScanningContext.class, "");
  }

  @Override
  public Iterable<Class<? extends OptionsBase>> getCommandOptions(Command command) {
    return "build".equals(command.name())
        ? ImmutableList.of(IncludeScanningOptions.class)
        : ImmutableList.<Class<? extends OptionsBase>>of();
  }

  @Override
  public void beforeCommand(CommandEnvironment env) {
    artifactFactory.set(env.getSkyframeBuildView().getArtifactFactory());
  }

  @Override
  public void afterCommand() {
    spawnIncludeScannerSupplier.set(null);
    artifactFactory.set(null);
  }

  @Override
  public void workspaceInit(
      BlazeRuntime runtime, BlazeDirectories directories, WorkspaceBuilder builder) {
    builder.addSkyFunctions(getSkyFunctions(getIncludeHintsFilename()));
  }

  @VisibleForTesting
  public static ImmutableMap<SkyFunctionName, SkyFunction> getSkyFunctions(
      PathFragment includeHintsFile) {
    return ImmutableMap.of(
        IncludeScanningSkyFunctions.INCLUDE_HINTS,
        new IncludeHintsFunction(includeHintsFile));
  }

  /**
   * Implementation of {@link CppIncludeExtractionContext}.
   */
  @ExecutionStrategy(contextType = CppIncludeExtractionContext.class)
  public static final class CppIncludeExtractionContextImpl implements CppIncludeExtractionContext {
    private final CommandEnvironment env;

    CppIncludeExtractionContextImpl(CommandEnvironment env) {
      this.env = env;
    }

    @Override
    public ArtifactResolver getArtifactResolver() {
      return env.getSkyframeBuildView().getArtifactFactory();
    }

    @Override
    public void extractIncludes(
        ActionExecutionContext actionExecutionContext,
        Action resourceOwner,
        Artifact primaryInput,
        Artifact primaryOutput,
        Artifact grepIncludes)
        throws ExecException, InterruptedException {
      SpawnIncludeScanner.spawnGrep(
          primaryInput,
          primaryOutput.getExecPath(),
          // We must actually write the .includes files to disk here because they are Artifacts in
          // the action graph.
          /*inMemoryOutput=*/ false,
          resourceOwner,
          actionExecutionContext,
          grepIncludes,
          GrepIncludesFileType.CPP);
    }
  }

  /**
   * SwigIncludeScanningContextImpl implements SwigIncludeScanningContext.
   */
  @ExecutionStrategy(contextType = SwigIncludeScanningContext.class)
  public static final class SwigIncludeScanningContextImpl implements SwigIncludeScanningContext {
    private final CommandEnvironment env;
    private final Supplier<SpawnIncludeScanner> spawnScannerSupplier;
    private final Supplier<ExecutorService> includePool;
    private final ConcurrentMap<Artifact, FutureTask<Collection<Inclusion>>> cache =
        new ConcurrentHashMap<>();

    SwigIncludeScanningContextImpl(
        CommandEnvironment env,
        Supplier<SpawnIncludeScanner> spawnScannerSupplier,
        Supplier<ExecutorService> includePool) {
      this.env = env;
      this.spawnScannerSupplier = spawnScannerSupplier;
      this.includePool = includePool;
    }

    @Override
    public void extractSwigIncludes(
        Set<Artifact> includes,
        ActionExecutionMetadata actionExecutionMetadata,
        ActionExecutionContext actionExecContext,
        Artifact source,
        ImmutableSet<Artifact> legalOutputPaths,
        ImmutableList<PathFragment> swigIncludePaths,
        Artifact grepIncludes)
        throws IOException, ExecException, InterruptedException {
      SwigIncludeScanner scanner =
          new SwigIncludeScanner(
              includePool.get(),
              spawnScannerSupplier.get(),
              cache,
              swigIncludePaths,
              env.getDirectories(),
              env.getSkyframeBuildView().getArtifactFactory(),
              env.getExecRoot());
      ImmutableMap.Builder<PathFragment, Artifact> pathToLegalOutputArtifact =
          ImmutableMap.builder();
      for (Artifact path : legalOutputPaths) {
        pathToLegalOutputArtifact.put(path.getExecPath(), path);
      }
      scanner.process(
          source,
          ImmutableList.of(source),
          // For Swig include scanning just point to the output file in the map.
          new IncludeScanningHeaderData.Builder(
                  pathToLegalOutputArtifact.build(), /*modularHeaders=*/ ImmutableSet.of())
              .build(),
          ImmutableList.of(),
          includes,
          actionExecutionMetadata,
          actionExecContext,
          grepIncludes);
    }
  }

  /**
   * Factory for execution strategies related to include scanning.
   */
  public static class IncludeScanningActionContextProvider extends ActionContextProvider {
    private final CommandEnvironment env;
    private final ImmutableList<ActionContext> strategies;
    private final BuildRequest buildRequest;

    private final Supplier<SpawnIncludeScanner> spawnScannerSupplier;
    private IncludeScannerSupplierImpl includeScannerSupplier;
    private ExecutorService includePool;

    public IncludeScanningActionContextProvider(
        CommandEnvironment env,
        BuildRequest buildRequest,
        MutableSupplier<SpawnIncludeScanner> spawnScannerSupplier) {
      this.env = env;
      this.buildRequest = buildRequest;

      IncludeScanningOptions options = buildRequest.getOptions(IncludeScanningOptions.class);
      spawnScannerSupplier.set(
          new SpawnIncludeScanner(
              env.getExecRoot(),
              options.experimentalRemoteExtractionThreshold));
      this.spawnScannerSupplier = spawnScannerSupplier;
      this.strategies =
          ImmutableList.of(
              new CppIncludeExtractionContextImpl(env),
              new SwigIncludeScanningContextImpl(env, spawnScannerSupplier, () -> includePool),
              new CppIncludeScanningContextImpl(() -> includeScannerSupplier));

      env.getEventBus().register(this);
    }

    @Override
    public Iterable<ActionContext> getActionContexts() {
      return strategies;
    }

    @Override
    public void executionPhaseStarting(
        ActionGraph actionGraph,
        Supplier<ArtifactsToOwnerLabels> topLevelArtifactsToAccountingGroups)
        throws ExecutorInitException, InterruptedException {
      try {
        includeScannerSupplier.init(
            new IncludeParser(
                new IncludeParser.Hints(
                    (IncludeParser.HintsRules)
                        env.getSkyframeExecutor()
                            .evaluateSkyKeyForExecutionSetup(
                                env.getReporter(), IncludeHintsFunction.INCLUDE_HINTS_KEY),
                    env.getSkyframeBuildView().getArtifactFactory())));
      } catch (ExecException e) {
        throw new ExecutorInitException("could not initialize include hints", e);
      }
    }

    @Override
    public void executorCreated(Iterable<ActionContext> usedContexts) throws ExecutorInitException {
      int threads = buildRequest.getOptions(IncludeScanningOptions.class)
          .includeScanningParallelism;
      if (threads > 0) {
        log.info(
            String.format("Include scanning configured to use a pool with %d threads", threads));
        includePool = ExecutorUtil.newSlackPool(threads, "Include scanner");
      } else {
        log.info("Include scanning configured to use a direct executor");
        includePool = MoreExecutors.newDirectExecutorService();
      }
      includeScannerSupplier =
          new IncludeScannerSupplierImpl(
              env.getDirectories(),
              includePool,
              env.getSkyframeBuildView().getArtifactFactory(),
              spawnScannerSupplier,
              env.getExecRoot());

      spawnScannerSupplier.get().setOutputService(env.getOutputService());
      spawnScannerSupplier.get().setInMemoryOutput(
          buildRequest.getOptions(IncludeScanningOptions.class).inMemoryIncludesFiles);
    }
  }
}
