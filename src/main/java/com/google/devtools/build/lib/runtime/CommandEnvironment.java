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
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.exec.OutputService;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.pkgcache.PackageCacheOptions;
import com.google.devtools.build.lib.pkgcache.PackageManager;
import com.google.devtools.build.lib.pkgcache.TargetPatternEvaluator;
import com.google.devtools.build.lib.profiler.AutoProfiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.runtime.proto.InvocationPolicyOuterClass.InvocationPolicy;
import com.google.devtools.build.lib.skyframe.SkyframeBuildView;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import com.google.devtools.build.lib.syntax.SkylarkSemanticsOptions;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.common.options.OptionsClassProvider;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;
import com.google.devtools.common.options.OptionsProvider;
import java.io.IOException;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicReference;
import javax.annotation.Nullable;

/**
 * Encapsulates the state needed for a single command. The environment is dropped when the current
 * command is done and all corresponding objects are garbage collected.
 */
public final class CommandEnvironment {
  private final BlazeRuntime runtime;
  private final BlazeWorkspace workspace;
  private final BlazeDirectories directories;

  private UUID commandId;  // Unique identifier for the command being run
  private final Reporter reporter;
  private final EventBus eventBus;
  private final BlazeModule.ModuleEnvironment blazeModuleEnvironment;
  private final Map<String, String> clientEnv = new TreeMap<>();
  private final Set<String> visibleActionEnv = new TreeSet<>();
  private final Set<String> visibleTestEnv = new TreeSet<>();
  private final Map<String, String> actionClientEnv = new TreeMap<>();
  private final TimestampGranularityMonitor timestampGranularityMonitor;
  private final Thread commandThread;

  private String[] crashData;

  private PathFragment relativeWorkingDirectory = PathFragment.EMPTY_FRAGMENT;
  private long commandStartTime;
  private OutputService outputService;
  private Path workingDirectory;
  private String workspaceName;

  private String commandName;
  private OptionsProvider options;

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
      Preconditions.checkNotNull(exception);
      Preconditions.checkNotNull(exception.getExitCode());
      if (pendingException.compareAndSet(null, exception)) {
        // There was no exception, so we're the first one to ask for an exit. Interrupt the command.
        commandThread.interrupt();
      }
    }
  }

  /**
   * Creates a new command environment which can be used for executing commands for the given
   * runtime in the given workspace, which will publish events on the given eventBus. The
   * commandThread passed is interrupted when a module requests an early exit.
   */
  CommandEnvironment(
      BlazeRuntime runtime, BlazeWorkspace workspace, EventBus eventBus, Thread commandThread) {
    this.runtime = runtime;
    this.workspace = workspace;
    this.directories = workspace.getDirectories();
    this.commandId = null; // Will be set once we get the client environment
    this.reporter = new Reporter(eventBus);
    this.eventBus = eventBus;
    this.commandThread = commandThread;
    this.blazeModuleEnvironment = new BlazeModuleEnvironment();
    this.timestampGranularityMonitor = new TimestampGranularityMonitor(runtime.getClock());
    // Record the command's starting time again, for use by
    // TimestampGranularityMonitor.waitForTimestampGranularity().
    // This should be done as close as possible to the start of
    // the command's execution.
    timestampGranularityMonitor.setCommandStartTime();

    // TODO(ulfjack): We don't call beforeCommand() in tests, but rely on workingDirectory being set
    // in setupPackageCache(). This leads to NPE if we don't set it here.
    this.workingDirectory = directories.getWorkspace();
    this.workspaceName = null;

    workspace.getSkyframeExecutor().setEventBus(eventBus);
  }

  /**
   * Same as CommandEnvironment(BlazeRuntime, BlazeWorkspace, EventBus, Thread) but with an
   * explicit commandName and options.
   *
   * ONLY for testing.
   */
  @VisibleForTesting
  CommandEnvironment(
      BlazeRuntime runtime, BlazeWorkspace workspace, EventBus eventBus, Thread commandThread,
      String commandNameForTesting, OptionsProvider optionsForTesting) {
    this(runtime, workspace, eventBus, commandThread);
    // Both commandName and options are normally set by beforeCommand(); however this method is not
    // called in tests (i.e. tests use BlazeRuntimeWrapper). These fields should only be set for
    // testing.
    this.commandName = commandNameForTesting;
    this.options = optionsForTesting;
  }

  public BlazeRuntime getRuntime() {
    return runtime;
  }

  public BlazeWorkspace getBlazeWorkspace() {
    return workspace;
  }

  public BlazeDirectories getDirectories() {
    return directories;
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

  public String getCommandName() {
    return commandName;
  }

  public OptionsProvider getOptions() {
    return options;
  }

  /**
   * Return an ordered version of the client environment restricted to those variables whitelisted
   * by the command-line options to be inheritable by actions.
   */
  public Map<String, String> getWhitelistedActionEnv() {
    return filterClientEnv(visibleActionEnv);
  }

  /**
   * Return an ordered version of the client environment restricted to those variables whitelisted
   * by the command-line options to be inheritable by actions.
   */
  public Map<String, String> getWhitelistedTestEnv() {
    return filterClientEnv(visibleTestEnv);
  }

  private Map<String, String> filterClientEnv(Set<String> vars) {
    Map<String, String> result = new TreeMap<>();
    for (String var : vars) {
      String value = clientEnv.get(var);
      if (value != null) {
        result.put(var, value);
      }
    }
    return Collections.unmodifiableMap(result);
  }

  @VisibleForTesting
  void updateClientEnv(List<Map.Entry<String, String>> clientEnvList) {
    Preconditions.checkState(clientEnv.isEmpty());

    Collection<Map.Entry<String, String>> env = clientEnvList;
    for (Map.Entry<String, String> entry : env) {
      clientEnv.put(entry.getKey(), entry.getValue());
    }
    // Try to set the clientId from the client environment.
    if (commandId == null) {
      String uuidString = clientEnv.get("BAZEL_INTERNAL_INVOCATION_ID");
      if (uuidString != null) {
        try {
          commandId = UUID.fromString(uuidString);
        } catch (IllegalArgumentException e) {
          // String was malformed, so we will resort to generating a random UUID
        }
      }
    }
    if (commandId == null) {
      // We have been provided with the client environment, but it didn't contain
      // the invocation id; hence generate our own.
      commandId = UUID.randomUUID();
    }
    setCommandIdInCrashData();
  }

  public TimestampGranularityMonitor getTimestampGranularityMonitor() {
    return timestampGranularityMonitor;
  }

  public PackageManager getPackageManager() {
    return getSkyframeExecutor().getPackageManager();
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

  public PackageRootResolver getPackageRootResolver() {
    return new SkyframePackageRootResolver(getSkyframeExecutor(), reporter);
  }

  /**
   * Returns the UUID that Blaze uses to identify everything logged from the current build command.
   * It's also used to invalidate Skyframe nodes that are specific to a certain invocation, such as
   * the build info.
   */
  public UUID getCommandId() {
    if (commandId == null) {
      // The commandId should not be requested before the beforeCommand is executed, as the
      // commandId might be set through the client environment. However, to simplify testing,
      // we set the id value before we throw the exception.
      commandId = UUID.randomUUID();
      throw new IllegalArgumentException("Build Id requested before client environment provided");
    }
    return commandId;
  }

  public SkyframeExecutor getSkyframeExecutor() {
    return workspace.getSkyframeExecutor();
  }

  public SkyframeBuildView getSkyframeBuildView() {
    return getSkyframeExecutor().getSkyframeBuildView();
  }

  /**
   * Returns the working directory of the server.
   *
   * <p>This is often the first entry on the {@code --package_path}, but not always.
   * Callers should certainly not make this assumption. The Path returned may be null.
   */
  public Path getWorkspace() {
    return getDirectories().getWorkspace();
  }

  public String getWorkspaceName() {
    Preconditions.checkNotNull(workspaceName);
    return workspaceName;
  }

  public void setWorkspaceName(String workspaceName) {
    Preconditions.checkState(this.workspaceName == null, "workspace name can only be set once");
    this.workspaceName = workspaceName;
  }
  /**
   * Returns if the client passed a valid workspace to be used for the build.
   */
  public boolean inWorkspace() {
    return getDirectories().inWorkspace();
  }

  /**
   * Returns the output base directory associated with this Blaze server
   * process. This is the base directory for shared Blaze state as well as tool
   * and strategy specific subdirectories.
   */
  public Path getOutputBase() {
    return getDirectories().getOutputBase();
  }

  /**
   * Returns the execution root directory associated with this Blaze server
   * process. This is where all input and output files visible to the actual
   * build reside.
   */
  public Path getExecRoot() {
    Preconditions.checkNotNull(workspaceName);
    return getDirectories().getExecRoot(workspaceName);
  }

  /**
   * Returns the directory where actions' outputs and errors will be written. Is below the directory
   * returned by {@link #getExecRoot}.
   */
  public Path getActionConsoleOutputDirectory() {
    return getDirectories().getActionConsoleOutputDirectory(getExecRoot());
  }

  /**
   * Returns the working directory of the {@code blaze} client process.
   *
   * <p>This may be equal to {@code BlazeRuntime#getWorkspace()}, or beneath it.
   *
   * @see #getWorkspace()
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
    return workspace.getPersistentActionCache(reporter);
  }

  /**
   * An array of String values useful if Blaze crashes. For now, just returns the build id as soon
   * as it is determined.
   */
  String[] getCrashData() {
    if (crashData == null) {
      String buildId;
      if (commandId == null) {
        buildId = " (build id not set yet)";
      } else {
        buildId = commandId + " (build id)";
      }
      crashData = new String[] {buildId};
    }
    return crashData;
  }

  private void setCommandIdInCrashData() {
    // Update the command id in the crash data, if it is already generated
    if (crashData != null && crashData.length >= 2) {
      crashData[1] = getCommandId() + " (build id)";
    }
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
    return getSkyframeExecutor().createConfigurations(reporter, runtime.getConfigurationFactory(),
        buildOptions, ImmutableSet.<String>of(), keepGoing);
  }

  /**
   * Prevents any further interruption of this command by modules, and returns the final exit code
   * from modules, or null if no modules requested an abrupt exit.
   *
   * <p>Always returns the same value on subsequent calls.
   */
  @Nullable
  private ExitCode finalizeExitCode() {
    // Set the pending exception so that further calls to exit(AbruptExitException) don't lead to
    // unwanted thread interrupts.
    if (pendingException.compareAndSet(null, new AbruptExitException(null))) {
      return null;
    }
    if (Thread.currentThread() == commandThread) {
      // We may have interrupted the thread in the process, so clear the interrupted bit.
      // Whether the command was interrupted or not, it's about to be over, so don't interrupt later
      // things happening on this thread.
      Thread.interrupted();
    }
    // Extract the exit code (it can be null if someone has already called finalizeExitCode()).
    return getPendingExitCode();
  }

  /**
   * Hook method called by the BlazeCommandDispatcher right before the dispatch
   * of each command ends (while its outcome can still be modified).
   */
  ExitCode precompleteCommand(ExitCode originalExit) {
    eventBus.post(new CommandPrecompleteEvent(originalExit));
    // If Blaze did not suffer an infrastructure failure, check for errors in modules.
    ExitCode exitCode = originalExit;
    ExitCode newExitCode = finalizeExitCode();
    if (!originalExit.isInfrastructureFailure() && newExitCode != null) {
      exitCode = newExitCode;
    }
    return exitCode;
  }

  /**
   * Returns the current exit code requested by modules, or null if no exit has been requested.
   */
  @Nullable
  public ExitCode getPendingExitCode() {
    AbruptExitException exception = getPendingException();
    return exception == null ? null : exception.getExitCode();
  }

  /**
   * Retrieves the exception currently queued by a Blaze module.
   *
   * <p>Prefer getPendingExitCode or throwPendingException where appropriate.
   */
  public AbruptExitException getPendingException() {
    return pendingException.get();
  }

  /**
   * Throws the exception currently queued by a Blaze module.
   *
   * <p>This should be called as often as is practical so that errors are reported as soon as
   * possible. Ideally, we'd not need this, but the event bus swallows exceptions so we raise
   * the exception this way.
   */
  public void throwPendingException() throws AbruptExitException {
    AbruptExitException exception = getPendingException();
    if (exception != null) {
      if (Thread.currentThread() == commandThread) {
        // Throwing this exception counts as the requested interruption. Clear the interrupted bit.
        Thread.interrupted();
      }
      throw exception;
    }
  }

  /**
   * Initializes the package cache using the given options, and syncs the package cache. Also
   * injects a defaults package and the skylark semantics using the options for the {@link
   * BuildConfiguration}.
   *
   * @see DefaultsPackage
   */
  public void setupPackageCache(OptionsClassProvider options,
      String defaultsPackageContents) throws InterruptedException, AbruptExitException {
    SkyframeExecutor skyframeExecutor = getSkyframeExecutor();
    if (!skyframeExecutor.hasIncrementalState()) {
      skyframeExecutor.resetEvaluator();
    }

    for (BlazeModule module : runtime.getBlazeModules()) {
      skyframeExecutor.injectExtraPrecomputedValues(module.getPrecomputedValues());
    }
    skyframeExecutor.sync(
        reporter,
        options.getOptions(PackageCacheOptions.class),
        options.getOptions(SkylarkSemanticsOptions.class),
        getOutputBase(),
        getWorkingDirectory(),
        defaultsPackageContents,
        getCommandId(),
        clientEnv,
        timestampGranularityMonitor,
        options);
  }

  public void recordLastExecutionTime() {
    workspace.recordLastExecutionTime(getCommandStartTime());
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

  /**
   * Hook method called by the BlazeCommandDispatcher prior to the dispatch of each command.
   *
   * @param options The CommonCommandOptions used by every command.
   * @throws AbruptExitException if this command is unsuitable to be run as specified
   */
  void beforeCommand(
      Command command,
      OptionsParser optionsParser,
      CommonCommandOptions options,
      long execStartTimeNanos,
      long waitTimeInMs,
      InvocationPolicy invocationPolicy)
      throws AbruptExitException {
    commandStartTime -= options.startupTime;
    if (runtime.getStartupOptionsProvider().getOptions(BlazeServerStartupOptions.class).watchFS) {
      try {
        // TODO(ulfjack): Get rid of the startup option and drop this code.
        optionsParser.parse("--watchfs");
      } catch (OptionsParsingException e) {
        // This should never happen.
        throw new IllegalStateException(e);
      }
    }
    this.commandName = command.name();
    this.options = optionsParser;

    eventBus.post(
        new GotOptionsEvent(runtime.getStartupOptionsProvider(), optionsParser, invocationPolicy));
    throwPendingException();

    outputService = null;
    BlazeModule outputModule = null;
    if (command.builds()) {
      for (BlazeModule module : runtime.getBlazeModules()) {
        OutputService moduleService = module.getOutputService();
        if (moduleService != null) {
          if (outputService != null) {
            throw new IllegalStateException(
                String.format(
                    "More than one module (%s and %s) returns an output service",
                    module.getClass(), outputModule.getClass()));
          }
          outputService = moduleService;
          outputModule = module;
        }
      }
    }

    SkyframeExecutor skyframeExecutor = getSkyframeExecutor();
    skyframeExecutor.setOutputService(outputService);

    // Ensure that the working directory will be under the workspace directory.
    Path workspace = getWorkspace();
    Path workingDirectory;
    if (inWorkspace()) {
      workingDirectory = workspace.getRelative(options.clientCwd);
    } else {
      workspace = FileSystemUtils.getWorkingDirectory(getDirectories().getFileSystem());
      workingDirectory = workspace;
    }
    this.relativeWorkingDirectory = workingDirectory.relativeTo(workspace);
    this.workingDirectory = workingDirectory;

    updateClientEnv(options.clientEnv);

    // Fail fast in the case where a Blaze command forgets to install the package path correctly.
    skyframeExecutor.setActive(false);
    // Let skyframe figure out if it needs to store graph edges for this build.
    skyframeExecutor.decideKeepIncrementalState(
        runtime.getStartupOptionsProvider().getOptions(BlazeServerStartupOptions.class).batch,
        optionsParser.getOptions(BuildView.Options.class));

    // Start the performance and memory profilers.
    runtime.beforeCommand(this, options, execStartTimeNanos);

    // actionClientEnv contains the environment where values from actionEnvironment are overridden.
    actionClientEnv.putAll(clientEnv);

    if (command.builds()) {
      // Compute the set of environment variables that are whitelisted on the commandline
      // for inheritance.
      for (Map.Entry<String, String> entry :
          optionsParser.getOptions(BuildConfiguration.Options.class).actionEnvironment) {
        if (entry.getValue() == null) {
          visibleActionEnv.add(entry.getKey());
        } else {
          visibleActionEnv.remove(entry.getKey());
          actionClientEnv.put(entry.getKey(), entry.getValue());
        }
      }
      for (Map.Entry<String, String> entry :
          optionsParser.getOptions(BuildConfiguration.Options.class).testEnvironment) {
        if (entry.getValue() == null) {
          visibleTestEnv.add(entry.getKey());
        }
      }
    }

    eventBus.post(new CommandStartEvent(
        command.name(), getCommandId(), getClientEnv(), workingDirectory, getDirectories(),
        waitTimeInMs + options.waitTime));
  }

  /** Returns the name of the file system we are writing output to. */
  public String determineOutputFileSystem() {
    // If we have a fancy OutputService, this may be different between consecutive Blaze commands
    // and so we need to compute it freshly. Otherwise, we can used the immutable value that's
    // precomputed by our BlazeWorkspace.
    if (getOutputService() != null) {
      try (AutoProfiler p = profiled("Finding output file system", ProfilerTask.INFO)) {
        return getOutputService().getFilesSystemName();
      }
    }
    return workspace.getOutputBaseFilesystemTypeName();
  }

  /**
   * Returns the client environment combined with all fixed env var settings from --action_env.
   */
  public Map<String, String> getActionClientEnv() {
    return Collections.unmodifiableMap(actionClientEnv);
  }
}
