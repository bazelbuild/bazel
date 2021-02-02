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

import static com.google.common.base.Preconditions.checkState;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.actions.MetadataProvider;
import com.google.devtools.build.lib.actions.ResourceManager;
import com.google.devtools.build.lib.actions.cache.ActionCache;
import com.google.devtools.build.lib.analysis.AnalysisOptions;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.clock.Clock;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.exec.SingleBuildFileCache;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.pkgcache.PackageManager;
import com.google.devtools.build.lib.pkgcache.PackageOptions;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.runtime.proto.InvocationPolicyOuterClass.InvocationPolicy;
import com.google.devtools.build.lib.server.FailureDetails;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.skyframe.SkyframeBuildView;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import com.google.devtools.build.lib.skyframe.TopDownActionCache;
import com.google.devtools.build.lib.skyframe.WorkspaceInfoFromDiff;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.io.OutErr;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.OutputService;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.common.options.OptionsParsingResult;
import com.google.devtools.common.options.OptionsProvider;
import com.google.protobuf.Any;
import java.io.IOException;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicReference;
import javax.annotation.Nullable;
import javax.annotation.concurrent.GuardedBy;

/**
 * Encapsulates the state needed for a single command. The environment is dropped when the current
 * command is done and all corresponding objects are garbage collected.
 *
 * <p>This class is non-final for mocking purposes. DO NOT extend it in production code.
 */
public class CommandEnvironment {
  private final BlazeRuntime runtime;
  private final BlazeWorkspace workspace;
  private final BlazeDirectories directories;

  private final UUID commandId;  // Unique identifier for the command being run
  private final String buildRequestId;  // Unique identifier for the build being run
  private final Reporter reporter;
  private final EventBus eventBus;
  private final BlazeModule.ModuleEnvironment blazeModuleEnvironment;
  private final Map<String, String> clientEnv;
  private final Set<String> visibleActionEnv = new TreeSet<>();
  private final Set<String> visibleTestEnv = new TreeSet<>();
  private final Map<String, String> repoEnv = new TreeMap<>();
  private final TimestampGranularityMonitor timestampGranularityMonitor;
  private final Thread commandThread;
  private final Command command;
  private final OptionsParsingResult options;
  private final PathPackageLocator packageLocator;
  private final Path workingDirectory;
  private final PathFragment relativeWorkingDirectory;
  private final Duration waitTime;
  private final long commandStartTime;
  private final ImmutableList<Any> commandExtensions;

  private OutputService outputService;
  private TopDownActionCache topDownActionCache;
  private String workspaceName;
  private boolean hasSyncedPackageLoading = false;
  @Nullable private WorkspaceInfoFromDiff workspaceInfoFromDiff;

  // This AtomicReference is set to:
  //   - null, if neither BlazeModuleEnvironment#exit nor #precompleteCommand have been called
  //   - Optional.of(e), if BlazeModuleEnvironment#exit has been called with value e
  //   - Optional.empty(), if #precompleteCommand was called before any call to
  //     BlazeModuleEnvironment#exit
  private final AtomicReference<Optional<AbruptExitException>> pendingException =
      new AtomicReference<>();

  private final Object fileCacheLock = new Object();

  @GuardedBy("fileCacheLock")
  private MetadataProvider fileCache;

  private class BlazeModuleEnvironment implements BlazeModule.ModuleEnvironment {
    @Override
    public Path getFileFromWorkspace(Label label) {
      Path buildFile = getPackageManager().getBuildFileForPackage(label.getPackageIdentifier());
      if (buildFile == null) {
        return null;
      }
      return buildFile.getParentDirectory().getRelative(label.getName());
    }

    @Override
    public void exit(AbruptExitException exception) {
      Preconditions.checkNotNull(exception);
      Preconditions.checkNotNull(exception.getExitCode());
      if (pendingException.compareAndSet(null, Optional.of(exception))) {
        // There was no exception, so we're the first one to ask for an exit. Interrupt the command.
        commandThread.interrupt();
      }
    }
  }

  /**
   * Creates a new command environment which can be used for executing commands for the given
   * runtime in the given workspace, which will publish events on the given eventBus. The
   * commandThread passed is interrupted when a module requests an early exit.
   *
   * @param warnings will be filled with any warnings from command environment initialization.
   */
  CommandEnvironment(
      BlazeRuntime runtime,
      BlazeWorkspace workspace,
      EventBus eventBus,
      Thread commandThread,
      Command command,
      OptionsParsingResult options,
      List<String> warnings,
      long waitTimeInMs,
      long commandStartTime,
      List<Any> commandExtensions) {
    this.runtime = runtime;
    this.workspace = workspace;
    this.directories = workspace.getDirectories();
    this.reporter = new Reporter(eventBus);
    this.eventBus = eventBus;
    this.commandThread = commandThread;
    this.command = command;
    this.options = options;
    this.blazeModuleEnvironment = new BlazeModuleEnvironment();
    this.timestampGranularityMonitor = new TimestampGranularityMonitor(runtime.getClock());
    // Record the command's starting time again, for use by
    // TimestampGranularityMonitor.waitForTimestampGranularity().
    // This should be done as close as possible to the start of
    // the command's execution.
    timestampGranularityMonitor.setCommandStartTime();

    CommonCommandOptions commandOptions =
        Preconditions.checkNotNull(
            options.getOptions(CommonCommandOptions.class),
            "CommandEnvironment needs its options provider to have CommonCommandOptions loaded.");
    Path workspacePath = directories.getWorkspace();
    Path workingDirectory;
    try {
      workingDirectory = computeWorkingDirectory(commandOptions);
    } catch (AbruptExitException e) {
      // We'll exit very soon, but set the working directory to something reasonable so remainder of
      // setup can finish.
      this.blazeModuleEnvironment.exit(e);
      workingDirectory = workspacePath;
    }
    this.workingDirectory = workingDirectory;
    if (getWorkspace() != null) {
      this.relativeWorkingDirectory = workingDirectory.relativeTo(getWorkspace());
    } else {
      this.relativeWorkingDirectory = PathFragment.EMPTY_FRAGMENT;
    }
    this.workspaceName = null;

    this.waitTime = Duration.ofMillis(waitTimeInMs + commandOptions.waitTime);
    this.commandStartTime = commandStartTime - commandOptions.startupTime;
    this.commandExtensions = ImmutableList.copyOf(commandExtensions);
    // If this command supports --package_path we initialize the package locator scoped
    // to the command environment
    if (commandHasPackageOptions(command) && workspacePath != null) {
      this.packageLocator =
          workspace
              .getSkyframeExecutor()
              .createPackageLocator(
                  reporter, options.getOptions(PackageOptions.class).packagePath, workingDirectory);
    } else {
      this.packageLocator = null;
    }
    workspace.getSkyframeExecutor().setEventBus(eventBus);

    ClientOptions clientOptions =
        Preconditions.checkNotNull(
            options.getOptions(ClientOptions.class),
            "CommandEnvironment needs its options provider to have ClientOptions loaded.");

    this.clientEnv = makeMapFromMapEntries(clientOptions.clientEnv);
    this.commandId = computeCommandId(commandOptions.invocationId, warnings);
    this.buildRequestId = computeBuildRequestId(commandOptions.buildRequestId, warnings);

    this.repoEnv.putAll(clientEnv);
    if (command.builds()) {
      // Compute the set of environment variables that are allowlisted on the commandline
      // for inheritance.
      for (Map.Entry<String, String> entry :
          options.getOptions(CoreOptions.class).actionEnvironment) {
        if (entry.getValue() == null) {
          visibleActionEnv.add(entry.getKey());
        } else {
          visibleActionEnv.remove(entry.getKey());
          repoEnv.put(entry.getKey(), entry.getValue());
        }
      }
      for (Map.Entry<String, String> entry :
          options.getOptions(CoreOptions.class).testEnvironment) {
        if (entry.getValue() == null) {
          visibleTestEnv.add(entry.getKey());
        }
      }
    }

    CoreOptions configOpts = options.getOptions(CoreOptions.class);
    if (configOpts != null) {
      for (Map.Entry<String, String> entry : configOpts.repositoryEnvironment) {
        repoEnv.put(entry.getKey(), entry.getValue());
      }
    }
  }

  private Path computeWorkingDirectory(CommonCommandOptions commandOptions)
      throws AbruptExitException {
    Path workspace = getWorkspace();
    Path workingDirectory;
    if (inWorkspace()) {
      PathFragment clientCwd = commandOptions.clientCwd;
      if (clientCwd.containsUplevelReferences()) {
        throw new AbruptExitException(
            DetailedExitCode.of(
                FailureDetail.newBuilder()
                    .setMessage("Client cwd '" + clientCwd + "' contains uplevel references")
                    .setClientEnvironment(
                        FailureDetails.ClientEnvironment.newBuilder()
                            .setCode(FailureDetails.ClientEnvironment.Code.CLIENT_CWD_MALFORMED)
                            .build())
                    .build()));
      }
      if (clientCwd.isAbsolute() && !clientCwd.startsWith(workspace.asFragment())) {
        throw new AbruptExitException(
            DetailedExitCode.of(
                FailureDetail.newBuilder()
                    .setMessage(
                        String.format(
                            "Client cwd '%s' is not inside workspace '%s'", clientCwd, workspace))
                    .setClientEnvironment(
                        FailureDetails.ClientEnvironment.newBuilder()
                            .setCode(FailureDetails.ClientEnvironment.Code.CLIENT_CWD_MALFORMED)
                            .build())
                    .build()));
      }
      workingDirectory = workspace.getRelative(clientCwd);
    } else {
      workingDirectory = FileSystemUtils.getWorkingDirectory(getRuntime().getFileSystem());
    }
    return workingDirectory;
  }

  // Returns whether the given command supports --package_path
  private static boolean commandHasPackageOptions(Command command) {
    return commandHasPackageOptions(command, new HashSet<>());
  }

  private static boolean commandHasPackageOptions(Command command, Set<Command> seen) {
    if (!seen.add(command)) {
      return false;
    }
    for (int i = 0; i < command.options().length; ++i) {
      if (command.options()[i] == PackageOptions.class) {
        return true;
      }
    }
    for (int i = 0; i < command.inherits().length; ++i) {
      Class<? extends BlazeCommand> blazeCommand = command.inherits()[i];
      Command annotation = blazeCommand.getAnnotation(Command.class);
      if (commandHasPackageOptions(annotation, seen)) {
        return true;
      }
    }
    return false;
  }

  public BlazeRuntime getRuntime() {
    return runtime;
  }

  public Clock getClock() {
    return getRuntime().getClock();
  }

  public OptionsProvider getStartupOptionsProvider() {
    return getRuntime().getStartupOptionsProvider();
  }

  public BlazeWorkspace getBlazeWorkspace() {
    return workspace;
  }

  public BlazeDirectories getDirectories() {
    return directories;
  }

  @Nullable
  public PathPackageLocator getPackageLocator() {
    return packageLocator;
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
    return clientEnv;
  }

  public Command getCommand() {
    return command;
  }

  public String getCommandName() {
    return command.name();
  }

  public OptionsParsingResult getOptions() {
    return options;
  }

  /**
   * Return an ordered version of the client environment restricted to those variables allowlisted
   * by the command-line options to be inheritable by actions.
   */
  public Map<String, String> getAllowlistedActionEnv() {
    return filterClientEnv(visibleActionEnv);
  }

  /**
   * Return an ordered version of the client environment restricted to those variables allowlisted
   * by the command-line options to be inheritable by actions.
   */
  public Map<String, String> getAllowlistedTestEnv() {
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

  private static Map<String, String> makeMapFromMapEntries(
      List<Map.Entry<String, String>> mapEntryList) {
    Map<String, String> result = new TreeMap<>();
    for (Map.Entry<String, String> entry : mapEntryList) {
      result.put(entry.getKey(), entry.getValue());
    }
    return Collections.unmodifiableMap(result);
  }

  private UUID computeCommandId(UUID idFromOptions, List<String> warnings) {
    // TODO(b/67895628): Stop reading ids from the environment after the compatibility window has
    // passed.
    UUID commandId = idFromOptions;
    if (commandId == null) { // Try to set the clientId from the client environment.
      String uuidString = clientEnv.getOrDefault("BAZEL_INTERNAL_INVOCATION_ID", "");
      if (!uuidString.isEmpty()) {
        try {
          commandId = UUID.fromString(uuidString);
          warnings.add(
              "BAZEL_INTERNAL_INVOCATION_ID is set. This will soon be deprecated in favor of "
                  + "--invocation_id. Please switch to using the flag.");
        } catch (IllegalArgumentException e) {
          // String was malformed, so we will resort to generating a random UUID
          commandId = UUID.randomUUID();
        }
      } else {
        commandId = UUID.randomUUID();
      }
    }
    return commandId;
  }

  private String computeBuildRequestId(String idFromOptions, List<String> warnings) {
    String buildRequestId = idFromOptions;
    if (buildRequestId == null) {
      String uuidString = clientEnv.getOrDefault("BAZEL_INTERNAL_BUILD_REQUEST_ID", "");
      if (!uuidString.isEmpty()) {
        buildRequestId = uuidString;
        warnings.add(
            "BAZEL_INTERNAL_BUILD_REQUEST_ID is set. This will soon be deprecated in favor of "
                + "--build_request_id. Please switch to using the flag.");
      } else {
        buildRequestId = UUID.randomUUID().toString();
      }
    }
    return buildRequestId;
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

  public Duration getWaitTime() {
    return waitTime;
  }

  List<OutErr> getOutputListeners() {
    List<OutErr> result = new ArrayList<>();
    for (BlazeModule module : runtime.getBlazeModules()) {
      OutErr listener = module.getOutputListener();
      if (listener != null) {
        result.add(listener);
      }
    }
    return result;
  }

  /**
   * Returns the UUID that Blaze uses to identify everything logged from the current build command.
   * It's also used to invalidate Skyframe nodes that are specific to a certain invocation, such as
   * the build info.
   */
  public UUID getCommandId() {
    return commandId;
  }

  /**
   * Returns the ID that Blaze uses to identify everything logged from the current build request.
   * TODO(olaola): this should be a prefixed UUID, but some existing clients still use arbitrary
   * strings, so we accept these when passed by environment variable for compatibility.
   */
  public String getBuildRequestId() {
    return buildRequestId;
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
    checkState(this.workspaceName == null, "workspace name can only be set once");
    this.workspaceName = workspaceName;
    eventBus.post(new ExecRootEvent(getExecRoot()));
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
   * Returns the directory where actions' temporary files will be written. Is below the directory
   * returned by {@link #getExecRoot}.
   */
  public Path getActionTempsDirectory() {
    return getDirectories().getActionTempsDirectory(getExecRoot());
  }

  public Path getPersistentActionOutsDirectory() {
    return getDirectories().getPersistentActionOutsDirectory(getExecRoot());
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

  /** @return the OutputService in use, or null if none. */
  @Nullable
  public OutputService getOutputService() {
    return outputService;
  }

  @VisibleForTesting
  public void setOutputServiceForTesting(@Nullable OutputService outputService) {
    this.outputService = outputService;
  }

  /**
   * Returns workspace information obtained from the {@linkplain
   * com.google.devtools.build.lib.skyframe.DiffAwareness.View#getWorkspaceInfo() diff} or null.
   *
   * <p>We store workspace info as an optimization to allow sharing of information about the
   * workspace if it was derived from the diff at the time of synchronizing the workspace. This way
   * we can make it available earlier during the build and avoid retrieving it again.
   */
  @Nullable
  public WorkspaceInfoFromDiff getWorkspaceInfoFromDiff() {
    return workspaceInfoFromDiff;
  }

  public ActionCache getPersistentActionCache() throws IOException {
    return workspace.getPersistentActionCache(reporter);
  }

  /** Returns the top-down action cache to use, or null. */
  public TopDownActionCache getTopDownActionCache() {
    return topDownActionCache;
  }

  public ResourceManager getLocalResourceManager() {
    return ResourceManager.instance();
  }

  /**
   * Prevents any further interruption of this command by modules, and returns the final {@link
   * DetailedExitCode} from modules, or null if no modules requested an abrupt exit.
   *
   * <p>Always returns the same value on subsequent calls.
   */
  @Nullable
  private DetailedExitCode finalizeDetailedExitCode() {
    // Set the pending exception so that further calls to exit(AbruptExitException) don't lead to
    // unwanted thread interrupts.
    if (pendingException.compareAndSet(null, Optional.empty())) {
      return null;
    }
    if (Thread.currentThread() == commandThread) {
      // We may have interrupted the thread in the process, so clear the interrupted bit.
      // Whether the command was interrupted or not, it's about to be over, so don't interrupt later
      // things happening on this thread.
      Thread.interrupted();
    }
    // Extract the exit code (it can be null if someone has already called finalizeExitCode()).
    return getPendingDetailedExitCode();
  }

  /**
   * Hook method called by the BlazeCommandDispatcher right before the dispatch of each command ends
   * (while its outcome can still be modified).
   */
  DetailedExitCode precompleteCommand(DetailedExitCode originalExit) {
    // TODO(b/138456686): this event is deprecated but is used in several places. Instead of lifting
    //  the ExitCode to a DetailedExitCode, see if it can be deleted.
    eventBus.post(new CommandPrecompleteEvent(originalExit.getExitCode()));
    return finalizeDetailedExitCode();
  }

  /** Returns the current exit code requested by modules, or null if no exit has been requested. */
  @Nullable
  private DetailedExitCode getPendingDetailedExitCode() {
    AbruptExitException exception = getPendingException();
    return exception == null ? null : exception.getDetailedExitCode();
  }

  /**
   * Retrieves the exception currently queued by a Blaze module.
   *
   * <p>Prefer {@link #getPendingDetailedExitCode} or {@link #throwPendingException} where
   * appropriate.
   */
  @Nullable
  public AbruptExitException getPendingException() {
    Optional<AbruptExitException> abruptExitExceptionMaybe = pendingException.get();
    return abruptExitExceptionMaybe == null ? null : abruptExitExceptionMaybe.orElse(null);
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
   * Initializes and syncs the graph with the given options, readying it for the next evaluation.
   */
  public void syncPackageLoading(OptionsProvider options)
      throws InterruptedException, AbruptExitException {
    // We want to ensure that we're never calling #syncPackageLoading twice in the same build
    // because it does the very expensive work of diffing the cache between incremental builds.
    // {@link SequencedSkyframeExecutor#handleDiffs} is the particular method we don't want to be
    // calling twice. We could feasibly factor it out of this call.
    if (hasSyncedPackageLoading) {
      throw new IllegalStateException(
          "We should never call this method more than once over the course of a single command");
    }
    hasSyncedPackageLoading = true;
    workspaceInfoFromDiff =
        getSkyframeExecutor()
            .sync(
                reporter,
                options.getOptions(PackageOptions.class),
                packageLocator,
                options.getOptions(BuildLanguageOptions.class),
                getCommandId(),
                clientEnv,
                timestampGranularityMonitor,
                options);
  }

  public void recordLastExecutionTime() {
    workspace.recordLastExecutionTime(getCommandStartTime());
  }

  public long getCommandStartTime() {
    return commandStartTime;
  }

  /**
   * Hook method called by the BlazeCommandDispatcher prior to the dispatch of each command.
   *
   * @throws AbruptExitException if this command is unsuitable to be run as specified
   */
  @VisibleForTesting
  public void beforeCommand(InvocationPolicy invocationPolicy) throws AbruptExitException {
    CommonCommandOptions commonOptions = options.getOptions(CommonCommandOptions.class);
    eventBus.post(new BuildMetadataEvent(makeMapFromMapEntries(commonOptions.buildMetadata)));
    eventBus.post(
        new GotOptionsEvent(runtime.getStartupOptionsProvider(), options, invocationPolicy));
    throwPendingException();

    outputService = null;
    BlazeModule outputModule = null;
    topDownActionCache = null;
    BlazeModule topDownCachingModule = null;
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

        TopDownActionCache moduleCache = module.getTopDownActionCache();
        if (moduleCache != null) {
          if (topDownActionCache != null) {
            throw new IllegalStateException(
                String.format(
                    "More than one module (%s and %s) returns a top down action cache",
                    module.getClass(), topDownCachingModule.getClass()));
          }
          topDownActionCache = moduleCache;
          topDownCachingModule = module;
        }
      }
    }

    SkyframeExecutor skyframeExecutor = getSkyframeExecutor();
    skyframeExecutor.setOutputService(outputService);
    skyframeExecutor.noteCommandStart();

    // Fail fast in the case where a Blaze command forgets to install the package path correctly.
    skyframeExecutor.setActive(false);
    // Let skyframe figure out how much incremental state it will be keeping.
    AnalysisOptions viewOptions = options.getOptions(AnalysisOptions.class);
    skyframeExecutor.decideKeepIncrementalState(
        runtime.getStartupOptionsProvider().getOptions(BlazeServerStartupOptions.class).batch,
        commonOptions.keepStateAfterBuild, commonOptions.trackIncrementalState,
        viewOptions != null && viewOptions.discardAnalysisCache,
        reporter);

    // Start the performance and memory profilers.
    runtime.beforeCommand(this, commonOptions);

    eventBus.post(new CommandStartEvent());

    // Modules that are subscribed to CommandStartEvent may create pending exceptions.
    throwPendingException();
  }

  /** Returns the name of the file system we are writing output to. */
  public String determineOutputFileSystem() {
    // If we have a fancy OutputService, this may be different between consecutive Blaze commands
    // and so we need to compute it freshly. Otherwise, we can used the immutable value that's
    // precomputed by our BlazeWorkspace.
    if (getOutputService() != null) {
      try (SilentCloseable c =
          Profiler.instance().profile(ProfilerTask.INFO, "Finding output file system")) {
        return getOutputService().getFilesSystemName();
      }
    }
    return workspace.getOutputBaseFilesystemTypeName();
  }

  /** Returns the client environment with all settings from --action_env and --repo_env. */
  public Map<String, String> getRepoEnv() {
    return Collections.unmodifiableMap(repoEnv);
  }

  /** Returns the file cache to use during this build. */
  public MetadataProvider getFileCache() {
    synchronized (fileCacheLock) {
      if (fileCache == null) {
        fileCache =
            new SingleBuildFileCache(getExecRoot().getPathString(), getRuntime().getFileSystem());
      }
      return fileCache;
    }
  }

  /**
   * Returns the {@linkplain
   * com.google.devtools.build.lib.server.CommandProtos.RunRequest#getCommandExtensions extensions}
   * passed to the server for this command.
   *
   * <p>Extensions are arbitrary messages containing additional per-command information.
   */
  public ImmutableList<Any> getCommandExtensions() {
    return commandExtensions;
  }
}
