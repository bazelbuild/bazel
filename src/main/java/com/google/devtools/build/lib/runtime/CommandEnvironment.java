// Copyright 2015 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.runtime;

import static com.google.devtools.build.lib.profiler.AutoProfiler.profiled;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.actions.PackageRootResolver;
import com.google.devtools.build.lib.actions.cache.ActionCache;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.BuildView;
import com.google.devtools.build.lib.analysis.SkyframePackageRootResolver;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationCollection;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.DefaultsPackage;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.exec.OutputService;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.pkgcache.PackageCacheOptions;
import com.google.devtools.build.lib.pkgcache.PackageManager;
import com.google.devtools.build.lib.pkgcache.TargetPatternEvaluator;
import com.google.devtools.build.lib.pkgcache.TransitivePackageLoader;
import com.google.devtools.build.lib.profiler.AutoProfiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.common.options.OptionPriority;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;
import com.google.devtools.common.options.OptionsProvider;

import java.io.IOException;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Encapsulates the state needed for a single command. The environment is dropped when the current
 * command is done and all corresponding objects are garbage collected.
 */
public final class CommandEnvironment {
  private final BlazeRuntime runtime;

  private final UUID commandId;  // Unique identifier for the command being run
  private final Reporter reporter;
  private final EventBus eventBus;
  private final BlazeModule.ModuleEnvironment blazeModuleEnvironment;
  private final Map<String, String> clientEnv = new HashMap<>();

  private final BuildView view;

  private PathFragment relativeWorkingDirectory = PathFragment.EMPTY_FRAGMENT;
  private long commandStartTime;
  private OutputService outputService;
  private String outputFileSystem;
  private Path workingDirectory;

  private AtomicReference<AbruptExitException> pendingException = new AtomicReference<>();

  private class BlazeModuleEnvironment implements BlazeModule.ModuleEnvironment {
    @Override
    public Path getFileFromWorkspace(Label label)
        throws NoSuchThingException, InterruptedException, IOException {
      Target target = getPackageManager().getTarget(reporter, label);
      return (outputService != null)
          ? outputService.stageTool(target)
          : target.getPackage().getPackageDirectory().getRelative(target.getName());
    }

    @Override
    public void exit(AbruptExitException exception) {
      pendingException.compareAndSet(null, exception);
    }
  }

  public CommandEnvironment(BlazeRuntime runtime, UUID commandId, EventBus eventBus) {
    this.runtime = runtime;
    this.commandId = commandId;
    this.reporter = new Reporter();
    this.eventBus = eventBus;
    this.blazeModuleEnvironment = new BlazeModuleEnvironment();

    this.view = new BuildView(runtime.getDirectories(), runtime.getRuleClassProvider(),
        runtime.getSkyframeExecutor(), runtime.getCoverageReportActionFactory());

    // TODO(ulfjack): We don't call beforeCommand() in tests, but rely on workingDirectory being set
    // in setupPackageCache(). This leads to NPE if we don't set it here.
    this.workingDirectory = runtime.getWorkspace();
  }

  public BlazeRuntime getRuntime() {
    return runtime;
  }

  public BlazeDirectories getDirectories() {
    return runtime.getDirectories();
  }

  /**
   * Returns the reporter for events.
   */
  public Reporter getReporter() {
    return reporter;
  }

  public EventBus getEventBus() {
    return eventBus;
  }

  public BlazeModule.ModuleEnvironment getBlazeModuleEnvironment() {
    return blazeModuleEnvironment;
  }

  /**
   * Return an unmodifiable view of the blaze client's environment when it invoked the current
   * command.
   */
  public Map<String, String> getClientEnv() {
    return Collections.unmodifiableMap(clientEnv);
  }

  @VisibleForTesting
  void updateClientEnv(List<Map.Entry<String, String>> clientEnvList, boolean ignoreClientEnv) {
    Preconditions.checkState(clientEnv.isEmpty());

    Collection<Map.Entry<String, String>> env =
        ignoreClientEnv ? System.getenv().entrySet() : clientEnvList;
    for (Map.Entry<String, String> entry : env) {
      clientEnv.put(entry.getKey(), entry.getValue());
    }
  }

  public PackageManager getPackageManager() {
    return runtime.getPackageManager();
  }

  public PathFragment getRelativeWorkingDirectory() {
    return relativeWorkingDirectory;
  }

  /**
   * Creates and returns a new target pattern parser.
   */
  public TargetPatternEvaluator newTargetPatternEvaluator() {
    TargetPatternEvaluator result = getPackageManager().newTargetPatternEvaluator();
    result.updateOffset(relativeWorkingDirectory);
    return result;
  }

  public BuildView getView() {
    return view;
  }

  public PackageRootResolver getPackageRootResolver() {
    return new SkyframePackageRootResolver(getSkyframeExecutor(), reporter);
  }

  /**
   * Returns the UUID that Blaze uses to identify everything logged from the current build command.
   * It's also used to invalidate Skyframe nodes that are specific to a certain invocation, such as
   * the build info.
   */
  public UUID getCommandId() {
    return commandId;
  }

  public SkyframeExecutor getSkyframeExecutor() {
    return runtime.getSkyframeExecutor();
  }

  /**
   * Returns the working directory of the {@code blaze} client process.
   *
   * <p>This may be equal to {@code BlazeRuntime#getWorkspace()}, or beneath it.
   *
   * @see BlazeRuntime#getWorkspace()
   */
  public Path getWorkingDirectory() {
    return workingDirectory;
  }

  /**
   * @return the OutputService in use, or null if none.
   */
  public OutputService getOutputService() {
    return outputService;
  }

  public ActionCache getPersistentActionCache() throws IOException {
    return runtime.getPersistentActionCache(reporter);
  }

  /**
   * This method only exists for the benefit of InfoCommand, which needs to construct a {@link
   * BuildConfigurationCollection} without running a full loading phase. Don't add any more clients;
   * instead, we should change info so that it doesn't need the configuration.
   */
  public BuildConfigurationCollection getConfigurations(OptionsProvider optionsProvider)
      throws InvalidConfigurationException, InterruptedException {
    BuildOptions buildOptions = runtime.createBuildOptions(optionsProvider);
    boolean keepGoing = optionsProvider.getOptions(BuildView.Options.class).keepGoing;
    boolean loadingSuccessful =
        loadForConfigurations(reporter,
            ImmutableSet.copyOf(buildOptions.getAllLabels().values()),
            keepGoing);
    if (!loadingSuccessful) {
      throw new InvalidConfigurationException("Configuration creation failed");
    }
    return getSkyframeExecutor().createConfigurations(reporter, runtime.getConfigurationFactory(),
        buildOptions, runtime.getDirectories(), ImmutableSet.<String>of(), keepGoing);
  }

  // TODO(ulfjack): Do we even need this method? With Skyframe, the config creation should
  // implicitly trigger any necessary loading.
  private boolean loadForConfigurations(EventHandler eventHandler,
      Set<Label> labelsToLoad, boolean keepGoing) throws InterruptedException {
    // Use a new Label Visitor here to avoid erasing the cache on the existing one.
    TransitivePackageLoader transitivePackageLoader =
        runtime.getSkyframeExecutor().getPackageManager().newTransitiveLoader();
    boolean loadingSuccessful = transitivePackageLoader.sync(
        eventHandler, ImmutableSet.<Target>of(),
        labelsToLoad, keepGoing, /*parallelThreads=*/10,
        /*maxDepth=*/Integer.MAX_VALUE);
    return loadingSuccessful;
  }

  /**
   * Hook method called by the BlazeCommandDispatcher right before the dispatch
   * of each command ends (while its outcome can still be modified).
   */
  ExitCode precompleteCommand(ExitCode originalExit) {
    eventBus.post(new CommandPrecompleteEvent(originalExit));
    // If Blaze did not suffer an infrastructure failure, check for errors in modules.
    ExitCode exitCode = originalExit;
    AbruptExitException exception = pendingException.get();
    if (!originalExit.isInfrastructureFailure() && exception != null) {
      exitCode = exception.getExitCode();
    }
    return exitCode;
  }

  /**
   * Throws the exception currently queued by a Blaze module.
   *
   * <p>This should be called as often as is practical so that errors are reported as soon as
   * possible. Ideally, we'd not need this, but the event bus swallows exceptions so we raise
   * the exception this way.
   */
  public void throwPendingException() throws AbruptExitException {
    AbruptExitException exception = pendingException.get();
    if (exception != null) {
      throw exception;
    }
  }

  /**
   * Initializes the package cache using the given options, and syncs the package cache. Also
   * injects a defaults package using the options for the {@link BuildConfiguration}.
   *
   * @see DefaultsPackage
   */
  public void setupPackageCache(PackageCacheOptions packageCacheOptions,
      String defaultsPackageContents) throws InterruptedException, AbruptExitException {
    SkyframeExecutor skyframeExecutor = getSkyframeExecutor();
    if (!skyframeExecutor.hasIncrementalState()) {
      skyframeExecutor.resetEvaluator();
    }
    skyframeExecutor.sync(reporter, packageCacheOptions, runtime.getOutputBase(),
        getWorkingDirectory(), defaultsPackageContents, commandId);
  }

  public void recordLastExecutionTime() {
    runtime.recordLastExecutionTime(getCommandStartTime());
  }

  public void recordCommandStartTime(long commandStartTime) {
    this.commandStartTime = commandStartTime;
  }

  public long getCommandStartTime() {
    return commandStartTime;
  }

  void setWorkingDirectory(Path workingDirectory) {
    this.workingDirectory = workingDirectory;
  }

  public String getOutputFileSystem() {
    return outputFileSystem;
  }

  /**
   * Hook method called by the BlazeCommandDispatcher prior to the dispatch of
   * each command.
   *
   * @param options The CommonCommandOptions used by every command.
   * @throws AbruptExitException if this command is unsuitable to be run as specified
   */
  void beforeCommand(Command command, OptionsParser optionsParser,
      CommonCommandOptions options, long execStartTimeNanos)
      throws AbruptExitException {
    commandStartTime -= options.startupTime;

    eventBus.post(new GotOptionsEvent(runtime.getStartupOptionsProvider(), optionsParser));
    throwPendingException();

    outputService = null;
    BlazeModule outputModule = null;
    for (BlazeModule module : runtime.getBlazeModules()) {
      OutputService moduleService = module.getOutputService();
      if (moduleService != null) {
        if (outputService != null) {
          throw new IllegalStateException(String.format(
              "More than one module (%s and %s) returns an output service",
              module.getClass(), outputModule.getClass()));
        }
        outputService = moduleService;
        outputModule = module;
      }
    }

    SkyframeExecutor skyframeExecutor = getSkyframeExecutor();
    skyframeExecutor.setOutputService(outputService);

    this.outputFileSystem = determineOutputFileSystem();

    // Ensure that the working directory will be under the workspace directory.
    Path workspace = runtime.getWorkspace();
    Path workingDirectory;
    if (runtime.inWorkspace()) {
      workingDirectory = workspace.getRelative(options.clientCwd);
    } else {
      workspace = FileSystemUtils.getWorkingDirectory(runtime.getDirectories().getFileSystem());
      workingDirectory = workspace;
    }
    this.relativeWorkingDirectory = workingDirectory.relativeTo(workspace);
    this.workingDirectory = workingDirectory;

    updateClientEnv(options.clientEnv, options.ignoreClientEnv);

    // Fail fast in the case where a Blaze command forgets to install the package path correctly.
    skyframeExecutor.setActive(false);
    // Let skyframe figure out if it needs to store graph edges for this build.
    skyframeExecutor.decideKeepIncrementalState(
        runtime.getStartupOptionsProvider().getOptions(BlazeServerStartupOptions.class).batch,
        optionsParser.getOptions(BuildView.Options.class));

    // Start the performance and memory profilers.
    runtime.beforeCommand(this, options, execStartTimeNanos);

    if (command.builds()) {
      Map<String, String> testEnv = new TreeMap<>();
      for (Map.Entry<String, String> entry :
          optionsParser.getOptions(BuildConfiguration.Options.class).testEnvironment) {
        testEnv.put(entry.getKey(), entry.getValue());
      }

      try {
        for (Map.Entry<String, String> entry : testEnv.entrySet()) {
          if (entry.getValue() == null) {
            String clientValue = clientEnv.get(entry.getKey());
            if (clientValue != null) {
              optionsParser.parse(OptionPriority.SOFTWARE_REQUIREMENT,
                  "test environment variable from client environment",
                  ImmutableList.of(
                      "--test_env=" + entry.getKey() + "=" + clientEnv.get(entry.getKey())));
            }
          }
        }
      } catch (OptionsParsingException e) {
        throw new IllegalStateException(e);
      }
    }
    for (BlazeModule module : runtime.getBlazeModules()) {
      module.handleOptions(optionsParser);
    }

    eventBus.post(
        new CommandStartEvent(command.name(), commandId, getClientEnv(), workingDirectory));
  }

  /**
   * Figures out what file system we are writing output to. Here we use
   * outputBase instead of outputPath because we need a file system to create the latter.
   */
  private String determineOutputFileSystem() {
    if (getOutputService() != null) {
      return getOutputService().getFilesSystemName();
    }
    try (AutoProfiler p = profiled("Finding output file system", ProfilerTask.INFO)) {
      return FileSystemUtils.getFileSystem(runtime.getOutputBase());
    }
  }
}
