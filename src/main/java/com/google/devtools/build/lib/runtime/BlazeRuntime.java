// Copyright 2014 The Bazel Authors. All rights reserved.
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

import static com.google.common.collect.ImmutableList.toImmutableList;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.eventbus.EventBus;
import com.google.common.eventbus.SubscriberExceptionHandler;
import com.google.common.flogger.GoogleLogger;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.Uninterruptibles;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.BlazeVersionInfo;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.ServerDirectories;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.test.CoverageReportActionFactory;
import com.google.devtools.build.lib.bazel.repository.downloader.Downloader;
import com.google.devtools.build.lib.bugreport.BugReport;
import com.google.devtools.build.lib.bugreport.BugReporter;
import com.google.devtools.build.lib.buildeventstream.BuildEvent.LocalFile.LocalFileType;
import com.google.devtools.build.lib.buildeventstream.BuildEventArtifactUploader;
import com.google.devtools.build.lib.buildeventstream.BuildEventArtifactUploader.UploadContext;
import com.google.devtools.build.lib.buildeventstream.BuildEventProtocolOptions;
import com.google.devtools.build.lib.buildtool.buildevent.ProfilerStartedEvent;
import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.clock.Clock;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.events.OutputFilter;
import com.google.devtools.build.lib.exec.BinTools;
import com.google.devtools.build.lib.jni.JniLoader;
import com.google.devtools.build.lib.packages.Package.Builder.DefaultPackageSettings;
import com.google.devtools.build.lib.packages.Package.Builder.PackageSettings;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.packages.PackageLoadingListener;
import com.google.devtools.build.lib.packages.PackageValidator;
import com.google.devtools.build.lib.profiler.AutoProfiler;
import com.google.devtools.build.lib.profiler.MemoryProfiler;
import com.google.devtools.build.lib.profiler.ProfilePhase;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.Profiler.Format;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.query2.QueryEnvironmentFactory;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryFunction;
import com.google.devtools.build.lib.query2.query.output.OutputFormatter;
import com.google.devtools.build.lib.query2.query.output.OutputFormatters;
import com.google.devtools.build.lib.runtime.CommandDispatcher.LockingMode;
import com.google.devtools.build.lib.runtime.proto.InvocationPolicyOuterClass.InvocationPolicy;
import com.google.devtools.build.lib.server.CommandProtos.EnvironmentVariable;
import com.google.devtools.build.lib.server.CommandProtos.ExecRequest;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.Filesystem;
import com.google.devtools.build.lib.server.FailureDetails.Interrupted.Code;
import com.google.devtools.build.lib.server.GrpcServerImpl;
import com.google.devtools.build.lib.server.PidFileWatcher;
import com.google.devtools.build.lib.server.RPCServer;
import com.google.devtools.build.lib.server.ShutdownHooks;
import com.google.devtools.build.lib.server.signal.InterruptSignalHandler;
import com.google.devtools.build.lib.shell.JavaSubprocessFactory;
import com.google.devtools.build.lib.shell.SubprocessBuilder;
import com.google.devtools.build.lib.shell.SubprocessFactory;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.CrashFailureDetails;
import com.google.devtools.build.lib.util.CustomExitCodePublisher;
import com.google.devtools.build.lib.util.CustomFailureDetailPublisher;
import com.google.devtools.build.lib.util.DebugLoggerConfigurator;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.util.InterruptedFailureDetails;
import com.google.devtools.build.lib.util.LoggingUtil;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.ProcessUtils;
import com.google.devtools.build.lib.util.TestType;
import com.google.devtools.build.lib.util.ThreadUtils;
import com.google.devtools.build.lib.util.io.OutErr;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.windows.WindowsSubprocessFactory;
import com.google.devtools.common.options.CommandNameCache;
import com.google.devtools.common.options.InvocationPolicyParser;
import com.google.devtools.common.options.OptionDefinition;
import com.google.devtools.common.options.OptionPriority.PriorityCategory;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;
import com.google.devtools.common.options.OptionsParsingResult;
import com.google.devtools.common.options.OptionsProvider;
import com.google.devtools.common.options.TriState;
import java.io.File;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;
import java.lang.reflect.Type;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.UUID;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.logging.Handler;
import java.util.logging.LogRecord;
import java.util.logging.Logger;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;

/**
 * The BlazeRuntime class encapsulates the immutable configuration of the current instance. These
 * runtime settings and services are available to most parts of any Blaze application for the
 * duration of the batch run or server lifetime.
 *
 * <p>The parts specific to the current command are stored in {@link CommandEnvironment}.
 */
public final class BlazeRuntime implements BugReport.BlazeRuntimeInterface {

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private final FileSystem fileSystem;
  private final ImmutableList<BlazeModule> blazeModules;
  private final Map<String, BlazeCommand> commandMap = new LinkedHashMap<>();
  private final Clock clock;
  private final Runnable abruptShutdownHandler;

  private final PackageFactory packageFactory;
  private final ConfiguredRuleClassProvider ruleClassProvider;
  // For bazel info.
  private final ImmutableMap<String, InfoItem> infoItems;
  // For bazel query.
  private final QueryEnvironmentFactory queryEnvironmentFactory;
  private final ImmutableList<QueryFunction> queryFunctions;
  private final ImmutableList<OutputFormatter> queryOutputFormatters;

  private final AtomicReference<DetailedExitCode> storedExitCode = new AtomicReference<>();

  // We pass this through here to make it available to the MasterLogWriter.
  private final OptionsParsingResult startupOptionsProvider;

  private final ProjectFile.Provider projectFileProvider;
  private final QueryRuntimeHelper.Factory queryRuntimeHelperFactory;
  @Nullable private final InvocationPolicy moduleInvocationPolicy;
  private final SubscriberExceptionHandler eventBusExceptionHandler;
  private final BugReporter bugReporter;
  private final String productName;
  private final BuildEventArtifactUploaderFactoryMap buildEventArtifactUploaderFactoryMap;
  private final ActionKeyContext actionKeyContext;
  private final ImmutableMap<String, AuthHeadersProvider> authHeadersProviderMap;
  private final RetainedHeapLimiter retainedHeapLimiter;
  @Nullable private final RepositoryRemoteExecutorFactory repositoryRemoteExecutorFactory;
  private final Supplier<Downloader> downloaderSupplier;

  // Workspace state (currently exactly one workspace per server)
  private BlazeWorkspace workspace;

  private BlazeRuntime(
      FileSystem fileSystem,
      QueryEnvironmentFactory queryEnvironmentFactory,
      ImmutableList<QueryFunction> queryFunctions,
      ImmutableList<OutputFormatter> queryOutputFormatters,
      PackageFactory pkgFactory,
      ConfiguredRuleClassProvider ruleClassProvider,
      ImmutableMap<String, InfoItem> infoItems,
      ActionKeyContext actionKeyContext,
      Clock clock,
      Runnable abruptShutdownHandler,
      OptionsParsingResult startupOptionsProvider,
      ImmutableList<BlazeModule> blazeModules,
      SubscriberExceptionHandler eventBusExceptionHandler,
      BugReporter bugReporter,
      ProjectFile.Provider projectFileProvider,
      QueryRuntimeHelper.Factory queryRuntimeHelperFactory,
      InvocationPolicy moduleInvocationPolicy,
      Iterable<BlazeCommand> commands,
      String productName,
      BuildEventArtifactUploaderFactoryMap buildEventArtifactUploaderFactoryMap,
      ImmutableMap<String, AuthHeadersProvider> authHeadersProviderMap,
      RepositoryRemoteExecutorFactory repositoryRemoteExecutorFactory,
      Supplier<Downloader> downloaderSupplier) {
    // Server state
    this.fileSystem = fileSystem;
    this.blazeModules = blazeModules;
    overrideCommands(commands);

    this.packageFactory = pkgFactory;
    this.projectFileProvider = projectFileProvider;
    this.queryRuntimeHelperFactory = queryRuntimeHelperFactory;
    this.moduleInvocationPolicy = moduleInvocationPolicy;

    this.ruleClassProvider = ruleClassProvider;
    this.infoItems = infoItems;
    this.actionKeyContext = actionKeyContext;
    this.clock = clock;
    this.abruptShutdownHandler = abruptShutdownHandler;
    this.startupOptionsProvider = startupOptionsProvider;
    this.queryEnvironmentFactory = queryEnvironmentFactory;
    this.queryFunctions = queryFunctions;
    this.queryOutputFormatters = queryOutputFormatters;
    this.eventBusExceptionHandler = eventBusExceptionHandler;
    this.bugReporter = bugReporter;
    retainedHeapLimiter = RetainedHeapLimiter.create(bugReporter);

    CommandNameCache.CommandNameCacheInstance.INSTANCE.setCommandNameCache(
        new CommandNameCacheImpl(getCommandMap()));
    this.productName = productName;
    this.buildEventArtifactUploaderFactoryMap = buildEventArtifactUploaderFactoryMap;
    this.authHeadersProviderMap =
        Preconditions.checkNotNull(authHeadersProviderMap, "authHeadersProviderMap");
    this.repositoryRemoteExecutorFactory = repositoryRemoteExecutorFactory;
    this.downloaderSupplier = downloaderSupplier;
  }

  public BlazeWorkspace initWorkspace(BlazeDirectories directories, BinTools binTools)
      throws AbruptExitException {
    Preconditions.checkState(this.workspace == null);
    WorkspaceBuilder builder = new WorkspaceBuilder(directories, binTools);
    for (BlazeModule module : blazeModules) {
      module.workspaceInit(this, directories, builder);
    }
    this.workspace =
        builder.build(this, packageFactory, ruleClassProvider, eventBusExceptionHandler);
    return workspace;
  }

  @Nullable
  public CoverageReportActionFactory getCoverageReportActionFactory(
      OptionsProvider commandOptions) {
    CoverageReportActionFactory firstFactory = null;
    for (BlazeModule module : blazeModules) {
      CoverageReportActionFactory factory = module.getCoverageReportFactory(commandOptions);
      if (factory != null) {
        Preconditions.checkState(
            firstFactory == null, "only one Bazel Module can have a Coverage Report Factory");
        firstFactory = factory;
      }
    }
    return firstFactory;
  }

  /**
   * Adds the given command under the given name to the map of commands.
   *
   * @throws AssertionError if the name is already used by another command.
   */
  private void addCommand(BlazeCommand command) {
    String name = command.getClass().getAnnotation(Command.class).name();
    if (commandMap.containsKey(name)) {
      throw new IllegalStateException("Command name or alias " + name + " is already used.");
    }
    commandMap.put(name, command);
  }

  @VisibleForTesting
  public final void overrideCommands(Iterable<BlazeCommand> commands) {
    commandMap.clear();
    for (BlazeCommand command : commands) {
      addCommand(command);
    }
  }

  @Nullable
  public InvocationPolicy getModuleInvocationPolicy() {
    return moduleInvocationPolicy;
  }

  private BuildEventArtifactUploader newUploader(
      CommandEnvironment env, String buildEventUploadStrategy) throws IOException {
    return getBuildEventArtifactUploaderFactoryMap().select(buildEventUploadStrategy).create(env);
  }

  /** Configure profiling based on the provided options. */
  ProfilerStartedEvent initProfiler(
      ExtendedEventHandler eventHandler,
      BlazeWorkspace workspace,
      CommonCommandOptions options,
      BuildEventProtocolOptions bepOptions,
      CommandEnvironment env,
      long execStartTimeNanos,
      long waitTimeInMs) {
    OutputStream out = null;
    boolean recordFullProfilerData = options.recordFullProfilerData;
    ImmutableSet.Builder<ProfilerTask> profiledTasksBuilder = ImmutableSet.builder();
    Profiler.Format format = Format.JSON_TRACE_FILE_FORMAT;
    Path profilePath = null;
    String profileName = null;
    UploadContext streamingContext = null;
    try {
      if (options.enableTracer || options.profilePath != null) {
        if (options.enableTracerCompression == TriState.YES
            || (options.enableTracerCompression == TriState.AUTO
                && (options.profilePath == null
                    || options.profilePath.toString().endsWith(".gz")))) {
          format = Format.JSON_TRACE_FILE_COMPRESSED_FORMAT;
        } else {
          format = Profiler.Format.JSON_TRACE_FILE_FORMAT;
        }
        if (options.profilePath != null) {
          profilePath = workspace.getWorkspace().getRelative(options.profilePath);
          out = profilePath.getOutputStream();
        } else {
          profileName = "command.profile";
          if (format == Format.JSON_TRACE_FILE_COMPRESSED_FORMAT) {
            profileName = "command.profile.gz";
          }
          if (bepOptions.streamingLogFileUploads) {
            BuildEventArtifactUploader buildEventArtifactUploader =
                newUploader(env, bepOptions.buildEventUploadStrategy);
            streamingContext = buildEventArtifactUploader.startUpload(LocalFileType.LOG, null);
            out = streamingContext.getOutputStream();
          } else {
            profilePath = workspace.getOutputBase().getRelative(profileName);
            out = profilePath.getOutputStream();
          }
        }
        if (profilePath != null && options.announceProfilePath) {
          eventHandler.handle(Event.info("Writing tracer profile to '" + profilePath + "'"));
        }
        for (ProfilerTask profilerTask : ProfilerTask.values()) {
          if (!profilerTask.isVfs()
              // CRITICAL_PATH corresponds to writing the file.
              && profilerTask != ProfilerTask.CRITICAL_PATH
              && profilerTask != ProfilerTask.SKYFUNCTION) {
            profiledTasksBuilder.add(profilerTask);
          }
        }
        profiledTasksBuilder.addAll(options.additionalProfileTasks);
        if (options.recordFullProfilerData) {
          profiledTasksBuilder.addAll(EnumSet.allOf(ProfilerTask.class));
        }
      } else if (options.alwaysProfileSlowOperations) {
        recordFullProfilerData = false;
        out = null;
        for (ProfilerTask profilerTask : ProfilerTask.values()) {
          if (profilerTask.collectsSlowestInstances()) {
            profiledTasksBuilder.add(profilerTask);
          }
        }
      }
      ImmutableSet<ProfilerTask> profiledTasks = profiledTasksBuilder.build();
      if (!profiledTasks.isEmpty()) {
        if (options.slimProfile && options.includePrimaryOutput) {
          eventHandler.handle(
              Event.warn(
                  "Enabling both --slim_profile and"
                      + " --experimental_profile_include_primary_output: the \"out\" field"
                      + " will be omitted in merged actions."));
        }
        Profiler profiler = Profiler.instance();
        profiler.start(
            profiledTasks,
            out,
            format,
            workspace.getOutputBase().toString(),
            env.getCommandId(),
            recordFullProfilerData,
            clock,
            execStartTimeNanos,
            options.enableCpuUsageProfiling,
            options.slimProfile,
            options.includePrimaryOutput,
            options.profileIncludeTargetLabel);
        // Instead of logEvent() we're calling the low level function to pass the timings we took in
        // the launcher. We're setting the INIT phase marker so that it follows immediately the
        // LAUNCH phase.
        long startupTimeNanos = options.startupTime * 1000000L;
        long waitTimeNanos = waitTimeInMs * 1000000L;
        long clientStartTimeNanos = execStartTimeNanos - startupTimeNanos - waitTimeNanos;
        profiler.logSimpleTaskDuration(
            clientStartTimeNanos,
            Duration.ofNanos(startupTimeNanos),
            ProfilerTask.PHASE,
            ProfilePhase.LAUNCH.description);
        if (options.extractDataTime > 0) {
          profiler.logSimpleTaskDuration(
              clientStartTimeNanos,
              Duration.ofMillis(options.extractDataTime),
              ProfilerTask.PHASE,
              "Extracting Bazel binary");
        }
        if (options.waitTime > 0) {
          profiler.logSimpleTaskDuration(
              clientStartTimeNanos,
              Duration.ofMillis(options.waitTime),
              ProfilerTask.PHASE,
              "Blocking on busy Bazel server (in client)");
        }
        if (waitTimeInMs > 0) {
          profiler.logSimpleTaskDuration(
              clientStartTimeNanos + startupTimeNanos,
              Duration.ofMillis(waitTimeInMs),
              ProfilerTask.PHASE,
              "Blocking on busy Bazel server (in server)");
        }
        profiler.logSimpleTaskDuration(
            execStartTimeNanos, Duration.ZERO, ProfilerTask.PHASE, ProfilePhase.INIT.description);
      }
    } catch (IOException e) {
      eventHandler.handle(Event.error("Error while creating profile file: " + e.getMessage()));
    }
    return new ProfilerStartedEvent(profileName, profilePath, streamingContext);
  }

  public FileSystem getFileSystem() {
    return fileSystem;
  }

  public BlazeWorkspace getWorkspace() {
    return workspace;
  }

  public ActionKeyContext getActionKeyContext() {
    return actionKeyContext;
  }

  /** The directory in which blaze stores the server state - that is, the socket file and a log. */
  private Path getServerDirectory() {
    return getWorkspace().getDirectories().getOutputBase().getChild("server");
  }

  /**
   * Returns the {@link QueryEnvironmentFactory} that should be used to create a {@link
   * com.google.devtools.build.lib.query2.common.AbstractBlazeQueryEnvironment}, whenever one is
   * needed.
   */
  public QueryEnvironmentFactory getQueryEnvironmentFactory() {
    return queryEnvironmentFactory;
  }

  public ImmutableList<QueryFunction> getQueryFunctions() {
    return queryFunctions;
  }

  public ImmutableList<OutputFormatter> getQueryOutputFormatters() {
    return queryOutputFormatters;
  }

  /** Returns the package factory. */
  public PackageFactory getPackageFactory() {
    return packageFactory;
  }

  /** Returns the rule class provider. */
  public ConfiguredRuleClassProvider getRuleClassProvider() {
    return ruleClassProvider;
  }

  public ImmutableMap<String, InfoItem> getInfoItems() {
    return infoItems;
  }

  public Iterable<BlazeModule> getBlazeModules() {
    return blazeModules;
  }

  public BuildOptions getDefaultBuildOptions() {
    BuildOptions options = null;
    for (BlazeModule module : blazeModules) {
      BuildOptions optionsFromModule = module.getDefaultBuildOptions(this);
      if (optionsFromModule != null) {
        if (options == null) {
          options = optionsFromModule;
        } else {
          throw new IllegalArgumentException(
              "Two or more bazel modules contained default build options.");
        }
      }
    }
    if (options == null) {
      throw new IllegalArgumentException("No default build options specified in any Bazel module");
    }
    return options;
  }

  /**
   * Returns the first module that is an instance of a given class or interface.
   *
   * @param moduleClass a class or interface that we want to match to a module
   * @param <T> the type of the module's class
   * @return a module that is an instance of this class or interface
   */
  @SuppressWarnings("unchecked")
  public <T> T getBlazeModule(Class<T> moduleClass) {
    for (BlazeModule module : blazeModules) {
      if (moduleClass.isInstance(module)) {
        return (T) module;
      }
    }
    return null;
  }

  /**
   * Returns a provider for project file objects. Can be null if no such provider was set by any of
   * the modules.
   */
  @Nullable
  public ProjectFile.Provider getProjectFileProvider() {
    return projectFileProvider;
  }

  public QueryRuntimeHelper.Factory getQueryRuntimeHelperFactory() {
    return queryRuntimeHelperFactory;
  }

  RetainedHeapLimiter getRetainedHeapLimiter() {
    return retainedHeapLimiter;
  }

  /**
   * Hook method called by the BlazeCommandDispatcher prior to the dispatch of each command.
   *
   * @param options The CommonCommandOptions used by every command.
   */
  void beforeCommand(CommandEnvironment env, CommonCommandOptions options) {
    if (options.memoryProfilePath != null) {
      Path memoryProfilePath = env.getWorkingDirectory().getRelative(options.memoryProfilePath);
      MemoryProfiler.instance()
          .setStableMemoryParameters(options.memoryProfileStableHeapParameters);
      try {
        MemoryProfiler.instance().start(memoryProfilePath.getOutputStream());
      } catch (IOException e) {
        env.getReporter()
            .handle(Event.error("Error while creating memory profile file: " + e.getMessage()));
      }
    }
  }

  @Override
  public void cleanUpForCrash(DetailedExitCode exitCode) {
    // TODO(b/167592709): remove verbose logging when bug resolved.
    logger.atInfo().log("Cleaning up in crash: %s", exitCode);
    if (declareExitCode(exitCode)) {
      // Only try to publish events if we won the exit code race. Otherwise someone else is already
      // exiting for us.
      EventBus eventBus = workspace.getSkyframeExecutor().getEventBus();
      if (eventBus != null) {
        workspace
            .getSkyframeExecutor()
            .postLoggingStatsWhenCrashing(
                new ExtendedEventHandler() {
                  @Override
                  public void post(Postable obj) {
                    eventBus.post(obj);
                  }

                  @Override
                  public void handle(Event event) {}
                });
        eventBus.post(new CommandCompleteEvent(exitCode));
      }
    }
    // We don't call #shutDown() here because all it does is shut down the modules, and who knows if
    // they can be trusted.  Instead, we call runtime#shutdownOnCrash() which attempts to cleanly
    // shut down those modules that might have something pending to do as a best-effort operation.
    shutDownModulesOnCrash();
  }

  private boolean declareExitCode(DetailedExitCode detailedExitCode) {
    return storedExitCode.compareAndSet(null, detailedExitCode);
  }

  /**
   * Posts the {@link CommandCompleteEvent}, so that listeners can tidy up. Called by {@link
   * #afterCommand}, and by BugReport when crashing from an exception in an async thread.
   *
   * <p>Returns null if {@code exitCode} was registered as the exit code, and the {@link
   * DetailedExitCode} to use if another thread already registered an exit code.
   */
  @Nullable
  private DetailedExitCode notifyCommandComplete(DetailedExitCode exitCode) {
    if (!declareExitCode(exitCode)) {
      // This command has already been called, presumably because there is a race between the main
      // thread and a worker thread that crashed. Don't try to arbitrate the dispute. If the main
      // thread won the race (unlikely, but possible), this may be incorrectly logged as a success.
      return storedExitCode.get();
    }
    workspace.getSkyframeExecutor().getEventBus().post(new CommandCompleteEvent(exitCode));
    return null;
  }

  /**
   * Hook method called by the BlazeCommandDispatcher after the dispatch of each command. Returns a
   * new exit code in case exceptions were encountered during cleanup.
   */
  @VisibleForTesting
  public BlazeCommandResult afterCommand(CommandEnvironment env, BlazeCommandResult commandResult) {
    // Remove any filters that the command might have added to the reporter.
    env.getReporter().setOutputFilter(OutputFilter.OUTPUT_EVERYTHING);

    BlazeCommandResult afterCommandResult = null;
    for (BlazeModule module : blazeModules) {
      try (SilentCloseable c = Profiler.instance().profile(module + ".afterCommand")) {
        module.afterCommand();
      } catch (AbruptExitException e) {
        env.getReporter().handle(Event.error(e.getMessage()));
        // It's not ideal but we can only return one exit code, so we just pick the code of the
        // last exception.
        afterCommandResult = BlazeCommandResult.detailedExitCode(e.getDetailedExitCode());
      }
    }

    env.getEventBus().post(new AfterCommandEvent());

    // Wipe the dependency graph if requested. Note that this method always runs at the end of
    // a commands unless the server crashes, in which case no inmemory state will linger for the
    // next build anyway.
    CommonCommandOptions commonOptions =
        Preconditions.checkNotNull(env.getOptions().getOptions(CommonCommandOptions.class));
    if (!commonOptions.keepStateAfterBuild) {
      workspace.getSkyframeExecutor().resetEvaluator();
    }

    // Build-related commands already call this hook in BuildTool#stopRequest, but non-build
    // commands might also need to notify the SkyframeExecutor. It's called in #stopRequest so that
    // timing metrics for builds can be more accurate (since this call can be slow).
    try {
      workspace.getSkyframeExecutor().notifyCommandComplete(env.getReporter());
    } catch (InterruptedException e) {
      logger.atInfo().withCause(e).log("Interrupted in afterCommand");
      afterCommandResult =
          BlazeCommandResult.detailedExitCode(
              InterruptedFailureDetails.detailedExitCode(
                  "executor completion interrupted", Code.EXECUTOR_COMPLETION));
      Thread.currentThread().interrupt();
    }

    BlazeCommandResult finalCommandResult;
    if (!commandResult.getExitCode().isInfrastructureFailure() && afterCommandResult != null) {
      finalCommandResult = afterCommandResult;
    } else {
      finalCommandResult = commandResult;
    }
    DetailedExitCode otherThreadWonExitCode =
        notifyCommandComplete(finalCommandResult.getDetailedExitCode());
    if (otherThreadWonExitCode != null) {
      finalCommandResult = BlazeCommandResult.detailedExitCode(otherThreadWonExitCode);
    }
    env.getBlazeWorkspace().clearEventBus();

    for (BlazeModule module : blazeModules) {
      try (SilentCloseable closeable = Profiler.instance().profile(module + ".commandComplete")) {
        module.commandComplete();
      }
    }

    try {
      Profiler.instance().stop();
      MemoryProfiler.instance().stop();
    } catch (IOException e) {
      env.getReporter().handle(Event.error("Error while writing profile file: " + e.getMessage()));
    }

    env.getReporter().clearEventBus();
    actionKeyContext.clear();
    DebugLoggerConfigurator.flushServerLog();
    storedExitCode.set(null);
    return finalCommandResult;
  }

  /**
   * Returns the Clock-instance used for the entire build. Before, individual classes (such as
   * Profiler) used to specify the type of clock (e.g. EpochClock) they wanted to use. This made it
   * difficult to get Blaze working on Windows as some of the clocks available for Linux aren't
   * (directly) available on Windows. Setting the Blaze-wide clock upon construction of BlazeRuntime
   * allows injecting whatever Clock instance should be used from BlazeMain.
   *
   * @return The Blaze-wide clock
   */
  public Clock getClock() {
    return clock;
  }

  /**
   * Returns the {@link BugReporter} that should be used when filing bug reports, if possible. Use
   * this in preference to {@link BugReport#sendBugReport} for ease of testing codepaths that file
   * bug reports.
   */
  public BugReporter getBugReporter() {
    return bugReporter;
  }

  public OptionsParsingResult getStartupOptionsProvider() {
    return startupOptionsProvider;
  }

  public Map<String, BlazeCommand> getCommandMap() {
    return commandMap;
  }

  /** Invokes {@link BlazeModule#blazeShutdown()} on all registered modules. */
  public void shutdown() {
    try {
      for (BlazeModule module : blazeModules) {
        module.blazeShutdown();
      }
    } finally {
      DebugLoggerConfigurator.flushServerLog();
    }
  }

  public void prepareForAbruptShutdown() {
    if (abruptShutdownHandler != null) {
      abruptShutdownHandler.run();
    }
  }

  /** Invokes {@link BlazeModule#blazeShutdownOnCrash()} on all registered modules. */
  private void shutDownModulesOnCrash() {
    // TODO(b/167592709): remove verbose logging when bug resolved.
    logger.atInfo().log("Shutting down modules on crash: %s", blazeModules);
    try {
      for (BlazeModule module : blazeModules) {
        logger.atInfo().log("Shutting down %s on crash", module);
        module.blazeShutdownOnCrash();
      }
    } finally {
      DebugLoggerConfigurator.flushServerLog();
    }
  }

  /** Creates a BuildOptions class for the given options taken from an optionsProvider. */
  public BuildOptions createBuildOptions(OptionsProvider optionsProvider) {
    return ruleClassProvider.createBuildOptions(optionsProvider);
  }

  /**
   * Main method for the Blaze server startup. Note: This method logs exceptions to remote servers.
   * Do not add this to a unittest.
   */
  public static void main(Iterable<Class<? extends BlazeModule>> moduleClasses, String[] args) {
    setupUncaughtHandlerAtStartup(args);
    List<BlazeModule> modules = createModules(moduleClasses);
    // blaze.cc will put --batch first if the user set it.
    if (args.length >= 1 && args[0].equals("--batch")) {
      // Run Blaze in batch mode.
      System.exit(batchMain(modules, args));
    }
    logger.atInfo().log(
        "Starting Bazel server with %s, args %s", maybeGetPidString(), Arrays.toString(args));
    try {
      // Run Blaze in server mode.
      System.exit(serverMain(modules, OutErr.SYSTEM_OUT_ERR, args));
    } catch (RuntimeException | Error e) { // A definite bug...
      BugReport.printBug(OutErr.SYSTEM_OUT_ERR, e, /* oomMessage = */ null);
      BugReport.sendBugReport(e, Arrays.asList(args));
      CustomFailureDetailPublisher.maybeWriteFailureDetailFile(CrashFailureDetails.forThrowable(e));
      System.exit(ExitCode.BLAZE_INTERNAL_ERROR.getNumericExitCode());
      throw e; // Shouldn't get here.
    }
  }

  @VisibleForTesting
  public static List<BlazeModule> createModules(
      Iterable<Class<? extends BlazeModule>> moduleClasses) {
    ImmutableList.Builder<BlazeModule> result = ImmutableList.builder();
    for (Class<? extends BlazeModule> moduleClass : moduleClasses) {
      try {
        BlazeModule module = moduleClass.getConstructor().newInstance();
        result.add(module);
      } catch (Throwable e) {
        throw new IllegalStateException("Cannot instantiate module " + moduleClass.getName(), e);
      }
    }

    return result.build();
  }

  /** Command line options split in to two parts: startup options and everything else. */
  @VisibleForTesting
  static class CommandLineOptions {
    private final ImmutableList<String> startupArgs;
    private final ImmutableList<String> otherArgs;

    CommandLineOptions(ImmutableList<String> startupArgs, ImmutableList<String> otherArgs) {
      this.startupArgs = Preconditions.checkNotNull(startupArgs);
      this.otherArgs = Preconditions.checkNotNull(otherArgs);
    }

    public ImmutableList<String> getStartupArgs() {
      return startupArgs;
    }

    public ImmutableList<String> getOtherArgs() {
      return otherArgs;
    }
  }

  /**
   * Splits given options into two lists - arguments matching options defined in this class and
   * everything else, while preserving order in each list.
   *
   * <p>Note that this method relies on the startup options always being in the <code>--flag=ARG
   * </code> form (instead of <code>--flag ARG</code>). This is enforced by <code>GetArgumentArray()
   * </code> in <code>blaze.cc</code> by reconstructing the startup options from their parsed
   * versions instead of using <code>argv</code> verbatim.
   */
  static CommandLineOptions splitStartupOptions(Iterable<BlazeModule> modules, String... args) {
    List<String> prefixes = new ArrayList<>();
    List<OptionDefinition> startupOptions = Lists.newArrayList();
    for (Class<? extends OptionsBase> defaultOptions :
        BlazeCommandUtils.getStartupOptions(modules)) {
      startupOptions.addAll(OptionsParser.getOptionDefinitions(defaultOptions));
    }

    for (OptionDefinition optionDefinition : startupOptions) {
      Type optionType = optionDefinition.getField().getType();
      prefixes.add("--" + optionDefinition.getOptionName());
      if (optionType == boolean.class || optionType == TriState.class) {
        prefixes.add("--no" + optionDefinition.getOptionName());
      }
    }

    List<String> startupArgs = new ArrayList<>();
    List<String> otherArgs = Lists.newArrayList(args);

    for (Iterator<String> argi = otherArgs.iterator(); argi.hasNext(); ) {
      String arg = argi.next();
      if (!arg.startsWith("--")) {
        break; // stop at command - all startup options would be specified before it.
      }
      for (String prefix : prefixes) {
        if (arg.startsWith(prefix)) {
          startupArgs.add(arg);
          argi.remove();
          break;
        }
      }
    }
    return new CommandLineOptions(
        ImmutableList.copyOf(startupArgs), ImmutableList.copyOf(otherArgs));
  }

  private static InterruptSignalHandler captureSigint() {
    final Thread mainThread = Thread.currentThread();
    final AtomicInteger numInterrupts = new AtomicInteger();

    final Runnable interruptWatcher =
        () -> {
          int count = 0;
          // Not an actual infinite loop because it's run in a daemon thread.
          while (true) {
            count++;
            Uninterruptibles.sleepUninterruptibly(10, TimeUnit.SECONDS);
            logger.atWarning().log("Slow interrupt number %d in batch mode", count);
            ThreadUtils.warnAboutSlowInterrupt();
          }
        };

    return new InterruptSignalHandler() {
      @Override
      public void run() {
        logger.atInfo().log("User interrupt");
        OutErr.SYSTEM_OUT_ERR.printErrLn("Bazel received an interrupt");
        mainThread.interrupt();

        int curNumInterrupts = numInterrupts.incrementAndGet();
        if (curNumInterrupts == 1) {
          Thread interruptWatcherThread = new Thread(interruptWatcher, "interrupt-watcher");
          interruptWatcherThread.setDaemon(true);
          interruptWatcherThread.start();
        } else if (curNumInterrupts == 2) {
          logger.atWarning().log("Second --batch interrupt: Reverting to JVM SIGINT handler");
          uninstall();
        }
      }
    };
  }

  /**
   * A main method that runs blaze commands in batch mode. The return value indicates the desired
   * exit status of the program.
   */
  private static int batchMain(Iterable<BlazeModule> modules, String[] args) {
    InterruptSignalHandler signalHandler = captureSigint();
    CommandLineOptions commandLineOptions = splitStartupOptions(modules, args);
    logger.atInfo().log(
        "Running Bazel in batch mode with %s, startup args %s",
        maybeGetPidString(), commandLineOptions.getStartupArgs());

    BlazeRuntime runtime;
    InvocationPolicy policy;
    try {
      runtime = newRuntime(modules, commandLineOptions.getStartupArgs(), null);
      policy =
          InvocationPolicyParser.parsePolicy(
              runtime
                  .getStartupOptionsProvider()
                  .getOptions(BlazeServerStartupOptions.class)
                  .invocationPolicy);
    } catch (OptionsParsingException e) {
      OutErr.SYSTEM_OUT_ERR.printErrLn(e.getMessage());
      return ExitCode.COMMAND_LINE_ERROR.getNumericExitCode();
    } catch (AbruptExitException e) {
      OutErr.SYSTEM_OUT_ERR.printErrLn(e.getMessage());
      return e.getExitCode().getNumericExitCode();
    }

    ImmutableList.Builder<Pair<String, String>> startupOptionsFromCommandLine =
        ImmutableList.builder();
    for (String option : commandLineOptions.getStartupArgs()) {
      startupOptionsFromCommandLine.add(new Pair<>("", option));
    }

    BlazeCommandDispatcher dispatcher =
        new BlazeCommandDispatcher(runtime, BlazeCommandDispatcher.UNKNOWN_SERVER_PID);
    boolean shutdownDone = false;

    try {
      logger.atInfo().log(
          SafeRequestLogging.getRequestLogString(commandLineOptions.getOtherArgs()));
      BlazeCommandResult result =
          dispatcher.exec(
              policy,
              commandLineOptions.getOtherArgs(),
              OutErr.SYSTEM_OUT_ERR,
              LockingMode.ERROR_OUT,
              "batch client",
              runtime.getClock().currentTimeMillis(),
              Optional.of(startupOptionsFromCommandLine.build()));
      if (result.getExecRequest() == null) {
        // Simple case: we are given an exit code
        return result.getExitCode().getNumericExitCode();
      }

      // Not so simple case: we need to execute a binary on shutdown. exec() is not accessible from
      // Java and is impossible on Windows in any case, so we just execute the binary after getting
      // out of the way as completely as possible and forward its exit code.
      // When this code is executed, no locks are held: the client lock is released by the client
      // before it executes any command and the server lock is handled by BlazeCommandDispatcher,
      // whose job is done by the time we get here.
      runtime.shutdown();
      dispatcher.shutdown();
      shutdownDone = true;
      signalHandler.uninstall();
      ExecRequest request = result.getExecRequest();
      String[] argv = new String[request.getArgvCount()];
      for (int i = 0; i < argv.length; i++) {
        argv[i] = request.getArgv(i).toString(StandardCharsets.ISO_8859_1);
      }

      String workingDirectory = request.getWorkingDirectory().toString(StandardCharsets.ISO_8859_1);
      try {
        ProcessBuilder process =
            new ProcessBuilder().command(argv).directory(new File(workingDirectory)).inheritIO();

        for (int i = 0; i < request.getEnvironmentVariableCount(); i++) {
          EnvironmentVariable variable = request.getEnvironmentVariable(i);
          process
              .environment()
              .put(
                  variable.getName().toString(StandardCharsets.ISO_8859_1),
                  variable.getValue().toString(StandardCharsets.ISO_8859_1));
        }

        return process.start().waitFor();
      } catch (IOException e) {
        // We are in batch mode, thus, stdout/stderr are the same as that of the client.
        System.err.println("Cannot execute process for 'run' command: " + e.getMessage());
        logger.atSevere().withCause(e).log("Exception while executing binary from 'run' command");
        return ExitCode.LOCAL_ENVIRONMENTAL_ERROR.getNumericExitCode();
      }
    } catch (InterruptedException e) {
      // This is almost main(), so it's okay to just swallow it. We are exiting soon.
      return ExitCode.INTERRUPTED.getNumericExitCode();
    } finally {
      if (!shutdownDone) {
        runtime.shutdown();
        dispatcher.shutdown();
      }
    }
  }

  /**
   * A main method that does not send email. The return value indicates the desired exit status of
   * the program.
   */
  private static int serverMain(Iterable<BlazeModule> modules, OutErr outErr, String[] args) {
    InterruptSignalHandler sigintHandler = null;
    try {
      AtomicReference<RPCServer> rpcServerRef = new AtomicReference<>();
      Runnable prepareForAbruptShutdown = () -> rpcServerRef.get().prepareForAbruptShutdown();
      BlazeRuntime runtime = newRuntime(modules, Arrays.asList(args), prepareForAbruptShutdown);

      // server.pid was written in the C++ launcher after fork() but before exec(). The client only
      // accesses the pid file after connecting to the socket which ensures that it gets the correct
      // pid value.
      Path pidFile = runtime.getServerDirectory().getRelative("server.pid.txt");
      int serverPid = readPidFile(pidFile);
      PidFileWatcher pidFileWatcher = new PidFileWatcher(pidFile, serverPid);
      pidFileWatcher.start();

      ShutdownHooks shutdownHooks = ShutdownHooks.createAndRegister();
      shutdownHooks.deleteAtExit(pidFile);

      BlazeCommandDispatcher dispatcher = new BlazeCommandDispatcher(runtime, serverPid);
      BlazeServerStartupOptions startupOptions =
          runtime.getStartupOptionsProvider().getOptions(BlazeServerStartupOptions.class);
      RPCServer rpcServer =
          GrpcServerImpl.create(
              dispatcher,
              shutdownHooks,
              pidFileWatcher,
              runtime.getClock(),
              startupOptions.commandPort,
              runtime.getServerDirectory(),
              serverPid,
              startupOptions.maxIdleSeconds,
              startupOptions.shutdownOnLowSysMem,
              startupOptions.idleServerTasks);
      rpcServerRef.set(rpcServer);

      // Register the signal handler.
      sigintHandler =
          new InterruptSignalHandler() {
            @Override
            public void run() {
              logger.atSevere().log("User interrupt");
              rpcServer.interrupt();
            }
          };

      rpcServer.serve();
      runtime.shutdown();
      dispatcher.shutdown();
      return ExitCode.SUCCESS.getNumericExitCode();
    } catch (OptionsParsingException e) {
      outErr.printErrLn(e.getMessage());
      return ExitCode.COMMAND_LINE_ERROR.getNumericExitCode();
    } catch (AbruptExitException e) {
      outErr.printErrLn(e.getMessage());
      e.printStackTrace(new PrintStream(outErr.getErrorStream(), true));
      FailureDetail failureDetail = e.getDetailedExitCode().getFailureDetail();
      if (failureDetail != null) {
        CustomFailureDetailPublisher.maybeWriteFailureDetailFile(failureDetail);
      }
      return e.getExitCode().getNumericExitCode();
    } finally {
      if (sigintHandler != null) {
        sigintHandler.uninstall();
      }
    }
  }

  private static SubprocessFactory subprocessFactoryImplementation() {
    if (JniLoader.isJniAvailable() && OS.getCurrent() == OS.WINDOWS) {
      return WindowsSubprocessFactory.INSTANCE;
    } else {
      return JavaSubprocessFactory.INSTANCE;
    }
  }

  /**
   * Parses the command line arguments into a {@link OptionsParser} object.
   *
   * <p>This function needs to parse the --option_sources option manually so that the real option
   * parser can set the source for every option correctly. If that cannot be parsed or is missing,
   * we just report an unknown source for every startup option.
   */
  private static OptionsParsingResult parseStartupOptions(
      Iterable<BlazeModule> modules, List<String> args) throws OptionsParsingException {
    ImmutableList<Class<? extends OptionsBase>> optionClasses =
        BlazeCommandUtils.getStartupOptions(modules);

    // First parse the command line so that we get the option_sources argument
    OptionsParser parser =
        OptionsParser.builder().optionsClasses(optionClasses).allowResidue(false).build();
    parser.parse(PriorityCategory.COMMAND_LINE, null, args);
    Map<String, String> optionSources =
        parser.getOptions(BlazeServerStartupOptions.class).optionSources;
    Function<OptionDefinition, String> sourceFunction =
        option ->
            !optionSources.containsKey(option.getOptionName())
                ? "default"
                : optionSources.get(option.getOptionName()).isEmpty()
                    ? "command line"
                    : optionSources.get(option.getOptionName());

    // Then parse the command line again, this time with the correct option sources
    parser = OptionsParser.builder().optionsClasses(optionClasses).allowResidue(false).build();
    parser.parseWithSourceFunction(PriorityCategory.COMMAND_LINE, sourceFunction, args);
    return parser;
  }

  /**
   * Creates a new blaze runtime, given the install and output base directories.
   *
   * <p>Note: This method can and should only be called once per startup, as it also creates the
   * filesystem object that will be used for the runtime. So it should only ever be called from the
   * main method of the Blaze program.
   *
   * @param args Blaze startup options.
   * @return a new BlazeRuntime instance initialized with the given filesystem and directories, and
   *     an error string that, if not null, describes a fatal initialization failure that makes this
   *     runtime unsuitable for real commands
   */
  private static BlazeRuntime newRuntime(
      Iterable<BlazeModule> blazeModules, List<String> args, Runnable abruptShutdownHandler)
      throws AbruptExitException, OptionsParsingException {
    OptionsParsingResult options = parseStartupOptions(blazeModules, args);
    BlazeServerStartupOptions startupOptions = options.getOptions(BlazeServerStartupOptions.class);

    // Set up the failure detail path first, so that it can communicate problems with other flags
    // and module initialization.
    PathFragment failureDetailOut = startupOptions.failureDetailOut;
    if (failureDetailOut == null || !failureDetailOut.isAbsolute()) { // (includes "" default case)
      throw new IllegalArgumentException(
          "Bad --failure_detail_out option specified: '" + failureDetailOut + "'");
    }
    CustomFailureDetailPublisher.setFailureDetailFilePath(failureDetailOut.getPathString());

    for (BlazeModule module : blazeModules) {
      module.globalInit(options);
    }

    String productName = startupOptions.productName.toLowerCase(Locale.US);

    PathFragment workspaceDirectory = startupOptions.workspaceDirectory;
    PathFragment defaultSystemJavabase = startupOptions.defaultSystemJavabase;
    PathFragment outputUserRoot = startupOptions.outputUserRoot;
    PathFragment installBase = startupOptions.installBase;
    PathFragment outputBase = startupOptions.outputBase;

    maybeForceJNIByGettingPid(installBase); // Must be before first use of JNI.

    // From the point of view of the Java program --install_base, --output_base, --output_user_root,
    // and --failure_detail_out are mandatory options, despite the comment in their declarations.

    if (installBase == null || !installBase.isAbsolute()) { // (includes "" default case)
      throw new IllegalArgumentException(
          "Bad --install_base option specified: '" + installBase + "'");
    }
    if (outputUserRoot != null && !outputUserRoot.isAbsolute()) { // (includes "" default case)
      throw new IllegalArgumentException(
          "Bad --output_user_root option specified: '" + outputUserRoot + "'");
    }
    if (outputBase != null && !outputBase.isAbsolute()) { // (includes "" default case)
      throw new IllegalArgumentException(
          "Bad --output_base option specified: '" + outputBase + "'");
    }

    FileSystem fs = null;
    Path execRootBasePath = null;
    for (BlazeModule module : blazeModules) {
      BlazeModule.ModuleFileSystem moduleFs =
          module.getFileSystem(options, outputBase.getRelative(ServerDirectories.EXECROOT));
      if (moduleFs != null) {
        execRootBasePath = moduleFs.virtualExecRootBase();
        Preconditions.checkState(fs == null, "more than one module returns a file system");
        fs = moduleFs.fileSystem();
      }
    }

    Preconditions.checkNotNull(fs, "No module set the file system");

    SubscriberExceptionHandler currentHandlerValue = null;
    for (BlazeModule module : blazeModules) {
      SubscriberExceptionHandler newHandler = module.getEventBusAndAsyncExceptionHandler();
      if (newHandler != null) {
        Preconditions.checkState(
            currentHandlerValue == null, "Two handlers given. Last module: %s", module);
        currentHandlerValue = newHandler;
      }
    }
    if (currentHandlerValue == null) {
      if (startupOptions.fatalEventBusExceptions) {
        currentHandlerValue = (exception, context) -> BugReport.handleCrash(exception);
      } else {
        currentHandlerValue =
            (exception, context) -> {
              if (context == null) {
                BugReport.handleCrash(exception);
              } else {
                BugReport.sendBugReport(
                    exception, ImmutableList.of(), "Failure in EventBus subscriber");
              }
            };
      }
    }
    SubscriberExceptionHandler subscriberExceptionHandler = currentHandlerValue;
    Thread.setDefaultUncaughtExceptionHandler(
        (thread, throwable) -> subscriberExceptionHandler.handleException(throwable, null));
    Path.setFileSystemForSerialization(fs);

    // Set the hook used to display Starlark source lines in a stack trace.
    final FileSystem finalFS = fs;
    EvalException.setSourceReaderSupplier(
        () ->
            loc -> {
              try {
                // TODO(adonovan): opt: cache seen files, as the stack often repeats the same files.
                Path path = finalFS.getPath(PathFragment.create(loc.file()));
                return Iterables.get(FileSystemUtils.readLines(path, UTF_8), loc.line() - 1, null);
              } catch (Throwable unused) {
                // ignore any failure (e.g. ENOENT, security manager rejecting I/O)
              }
              return null;
            });

    SubprocessBuilder.setDefaultSubprocessFactory(subprocessFactoryImplementation());

    Path outputUserRootPath = fs.getPath(outputUserRoot);
    Path installBasePath = fs.getPath(installBase);
    Path outputBasePath = fs.getPath(outputBase);
    if (execRootBasePath == null) {
      execRootBasePath = outputBasePath.getRelative(ServerDirectories.EXECROOT);
    }
    Path workspaceDirectoryPath = null;
    if (!workspaceDirectory.equals(PathFragment.EMPTY_FRAGMENT)) {
      workspaceDirectoryPath = fs.getPath(workspaceDirectory);
    }
    Path defaultSystemJavabasePath = null;
    if (!defaultSystemJavabase.equals(PathFragment.EMPTY_FRAGMENT)) {
      defaultSystemJavabasePath = fs.getPath(defaultSystemJavabase);
    }

    ServerDirectories serverDirectories =
        new ServerDirectories(
            installBasePath,
            outputBasePath,
            outputUserRootPath,
            execRootBasePath,
            startupOptions.installMD5);
    Clock clock = BlazeClock.instance();
    BlazeRuntime.Builder runtimeBuilder =
        new BlazeRuntime.Builder()
            .setProductName(productName)
            .setFileSystem(fs)
            .setServerDirectories(serverDirectories)
            .setActionKeyContext(new ActionKeyContext())
            .setStartupOptionsProvider(options)
            .setClock(clock)
            .setAbruptShutdownHandler(abruptShutdownHandler)
            .setEventBusExceptionHandler(subscriberExceptionHandler);

    if (TestType.isInTest() && System.getenv("NO_CRASH_ON_LOGGING_IN_TEST") == null) {
      LoggingUtil.installRemoteLogger(getTestCrashLogger());
    }

    for (BlazeModule blazeModule : blazeModules) {
      runtimeBuilder.addBlazeModule(blazeModule);
    }

    BlazeRuntime runtime = runtimeBuilder.build();

    CustomExitCodePublisher.setAbruptExitStatusFileDir(
        serverDirectories.getOutputBase().getPathString());

    BlazeDirectories directories =
        new BlazeDirectories(
            serverDirectories, workspaceDirectoryPath, defaultSystemJavabasePath, productName);
    BinTools binTools;
    try {
      binTools = BinTools.forProduction(directories);
    } catch (IOException e) {
      throw createFilesystemExitException(
          "Cannot enumerate embedded binaries: " + e.getMessage(),
          ExitCode.LOCAL_ENVIRONMENTAL_ERROR,
          Filesystem.Code.EMBEDDED_BINARIES_ENUMERATION_FAILURE,
          e);
    }
    // Keep this line last in this method, so that all other initialization is available to it.
    runtime.initWorkspace(directories, binTools);
    return runtime;
  }

  private static AbruptExitException createFilesystemExitException(
      String message, ExitCode exitCode, Filesystem.Code detailedCode, Exception e) {
    return new AbruptExitException(
        DetailedExitCode.of(
            exitCode,
            FailureDetail.newBuilder()
                .setMessage(message)
                .setFilesystem(Filesystem.newBuilder().setCode(detailedCode))
                .build()),
        e);
  }

  private static String maybeGetPidString() {
    Integer pid = maybeForceJNIByGettingPid(null);
    return pid == null ? "" : "pid " + pid + " and ";
  }

  /** Loads JNI libraries, if necessary under the current platform. */
  @Nullable
  private static Integer maybeForceJNIByGettingPid(@Nullable PathFragment installBase) {
    return JniLoader.isJniAvailable() ? getPidUsingJNI(installBase) : null;
  }

  // Force JNI linking at a moment when we have 'installBase' handy, and print
  // an informative error if it fails.
  private static int getPidUsingJNI(@Nullable PathFragment installBase) {
    try {
      return ProcessUtils.getpid(); // force JNI initialization
    } catch (UnsatisfiedLinkError t) {
      System.err.println(
          "JNI initialization failed: "
              + t.getMessage()
              + ".  "
              + "Possibly your installation has been corrupted"
              + (installBase == null
                  ? ""
                  : "; if this problem persists, try 'rm -fr " + installBase + "'")
              + ".");
      throw t;
    }
  }

  /**
   * Returns a logger that crashes as soon as it's written to, since tests should not cause events
   * that would be logged.
   */
  @VisibleForTesting
  public static Future<Logger> getTestCrashLogger() {
    Logger crashLogger = Logger.getAnonymousLogger();
    crashLogger.addHandler(
        new Handler() {
          @Override
          public void publish(LogRecord record) {
            System.err.println("Remote logging disabled for testing, forcing abrupt shutdown.");
            System.err.printf(
                "%s#%s: %s\n",
                record.getSourceClassName(), record.getSourceMethodName(), record.getMessage());

            Throwable e = record.getThrown();
            if (e != null) {
              e.printStackTrace();
            }

            Runtime.getRuntime().halt(ExitCode.BLAZE_INTERNAL_ERROR.getNumericExitCode());
          }

          @Override
          public void flush() {
            throw new IllegalStateException();
          }

          @Override
          public void close() {
            throw new IllegalStateException();
          }
        });
    return Futures.immediateFuture(crashLogger);
  }

  /**
   * Make sure async threads cannot be orphaned at startup. This method makes sure bugs are reported
   * to telemetry and the proper exit code is reported. Will be overwritten with better handler.
   */
  private static void setupUncaughtHandlerAtStartup(final String[] args) {
    Thread.setDefaultUncaughtExceptionHandler(
        (thread, throwable) -> BugReport.handleCrash(throwable, args));
  }

  @Override
  public String getProductName() {
    return productName;
  }

  public BuildEventArtifactUploaderFactoryMap getBuildEventArtifactUploaderFactoryMap() {
    return buildEventArtifactUploaderFactoryMap;
  }

  /** Returns a map of all registered {@link AuthHeadersProvider}s. */
  public ImmutableMap<String, AuthHeadersProvider> getAuthHeadersProvidersMap() {
    return authHeadersProviderMap;
  }

  public RepositoryRemoteExecutorFactory getRepositoryRemoteExecutorFactory() {
    return repositoryRemoteExecutorFactory;
  }

  public Supplier<Downloader> getDownloaderSupplier() {
    return downloaderSupplier;
  }

  /**
   * A builder for {@link BlazeRuntime} objects. The only required fields are the {@link
   * BlazeDirectories}, and the {@link com.google.devtools.build.lib.packages.RuleClassProvider}
   * (except for testing). All other fields have safe default values.
   *
   * <p>The default behavior of the BlazeRuntime's EventBus is to exit the JVM when a subscriber
   * throws an exception. Please plan appropriately.
   */
  public static class Builder {
    private FileSystem fileSystem;
    private ServerDirectories serverDirectories;
    private Clock clock;
    private Runnable abruptShutdownHandler;
    private OptionsParsingResult startupOptionsProvider;
    private final List<BlazeModule> blazeModules = new ArrayList<>();
    private SubscriberExceptionHandler eventBusExceptionHandler =
        (throwable, context) -> BugReport.handleCrash(throwable);
    private UUID instanceId;
    private String productName;
    private ActionKeyContext actionKeyContext;
    private BugReporter bugReporter = BugReporter.defaultInstance();

    @VisibleForTesting
    public BlazeRuntime build() throws AbruptExitException {
      Preconditions.checkNotNull(productName);
      Preconditions.checkNotNull(serverDirectories);
      Preconditions.checkNotNull(startupOptionsProvider);
      ActionKeyContext actionKeyContext =
          this.actionKeyContext != null ? this.actionKeyContext : new ActionKeyContext();
      Clock clock = (this.clock == null) ? BlazeClock.instance() : this.clock;
      UUID instanceId = (this.instanceId == null) ? UUID.randomUUID() : this.instanceId;

      Preconditions.checkNotNull(clock);

      int metricsModules = 0;
      for (BlazeModule module : blazeModules) {
        if (module.postsBuildMetricsEvent()) {
          metricsModules++;
        }
      }
      Preconditions.checkArgument(
          metricsModules < 2, "At most one module may post a BuildMetricsEvent");
      if (metricsModules == 0) {
        blazeModules.add(new DummyMetricsModule());
      }
      for (BlazeModule module : blazeModules) {
        module.blazeStartup(
            startupOptionsProvider,
            BlazeVersionInfo.instance(),
            instanceId,
            fileSystem,
            serverDirectories,
            clock);
      }
      ServerBuilder serverBuilder = new ServerBuilder();
      serverBuilder.addQueryOutputFormatters(OutputFormatters.getDefaultFormatters());
      for (BlazeModule module : blazeModules) {
        module.serverInit(startupOptionsProvider, serverBuilder);
      }

      ConfiguredRuleClassProvider.Builder ruleClassBuilder =
          new ConfiguredRuleClassProvider.Builder();
      BlazeServerStartupOptions blazeServerStartupOptions =
          startupOptionsProvider.getOptions(BlazeServerStartupOptions.class);
      if (blazeServerStartupOptions != null) {
        ruleClassBuilder.enableExecutionTransition(
            blazeServerStartupOptions.enableExecutionTransition);
      }
      for (BlazeModule module : blazeModules) {
        module.initializeRuleClasses(ruleClassBuilder);
      }

      ConfiguredRuleClassProvider ruleClassProvider = ruleClassBuilder.build();

      PackageSettings packageSettings = getPackageSettings(blazeModules);
      PackageFactory packageFactory =
          new PackageFactory(
              ruleClassProvider,
              PackageFactory.makeDefaultSizedForkJoinPoolForGlobbing(),
              serverBuilder.getEnvironmentExtensions(),
              BlazeVersionInfo.instance().getVersion(),
              packageSettings,
              getPackageValidator(blazeModules),
              getPackageLoadingListener(
                  blazeModules, packageSettings, ruleClassProvider, fileSystem));

      ProjectFile.Provider projectFileProvider = null;
      for (BlazeModule module : blazeModules) {
        ProjectFile.Provider candidate = module.createProjectFileProvider();
        if (candidate != null) {
          Preconditions.checkState(
              projectFileProvider == null, "more than one module defines a project file provider");
          projectFileProvider = candidate;
        }
      }

      QueryRuntimeHelper.Factory queryRuntimeHelperFactory = null;
      for (BlazeModule module : blazeModules) {
        QueryRuntimeHelper.Factory candidateFactory = module.getQueryRuntimeHelperFactory();
        if (candidateFactory != null) {
          Preconditions.checkState(
              queryRuntimeHelperFactory == null,
              "more than one module defines a query helper factory");
          queryRuntimeHelperFactory = candidateFactory;
        }
      }
      if (queryRuntimeHelperFactory == null) {
        queryRuntimeHelperFactory = QueryRuntimeHelper.StdoutQueryRuntimeHelperFactory.INSTANCE;
      }

      BlazeRuntime runtime =
          new BlazeRuntime(
              fileSystem,
              serverBuilder.getQueryEnvironmentFactory(),
              serverBuilder.getQueryFunctions(),
              serverBuilder.getQueryOutputFormatters(),
              packageFactory,
              ruleClassProvider,
              serverBuilder.getInfoItems(),
              actionKeyContext,
              clock,
              abruptShutdownHandler,
              startupOptionsProvider,
              ImmutableList.copyOf(blazeModules),
              eventBusExceptionHandler,
              bugReporter,
              projectFileProvider,
              queryRuntimeHelperFactory,
              serverBuilder.getInvocationPolicy(),
              serverBuilder.getCommands(),
              productName,
              serverBuilder.getBuildEventArtifactUploaderMap(),
              serverBuilder.getAuthHeadersProvidersMap(),
              serverBuilder.getRepositoryRemoteExecutorFactory(),
              serverBuilder.getDownloaderSupplier());
      AutoProfiler.setClock(runtime.getClock());
      BugReport.setRuntime(runtime);
      return runtime;
    }

    public Builder setProductName(String productName) {
      this.productName = productName;
      return this;
    }

    public Builder setFileSystem(FileSystem fileSystem) {
      this.fileSystem = fileSystem;
      return this;
    }

    public Builder setServerDirectories(ServerDirectories serverDirectories) {
      this.serverDirectories = serverDirectories;
      return this;
    }

    public Builder setClock(Clock clock) {
      this.clock = clock;
      return this;
    }

    public Builder setAbruptShutdownHandler(Runnable handler) {
      this.abruptShutdownHandler = handler;
      return this;
    }

    public Builder setStartupOptionsProvider(OptionsParsingResult startupOptionsProvider) {
      this.startupOptionsProvider = startupOptionsProvider;
      return this;
    }

    public Builder addBlazeModule(BlazeModule blazeModule) {
      blazeModules.add(blazeModule);
      return this;
    }

    public Builder setInstanceId(UUID id) {
      instanceId = id;
      return this;
    }

    @VisibleForTesting
    public Builder setEventBusExceptionHandler(
        SubscriberExceptionHandler eventBusExceptionHandler) {
      this.eventBusExceptionHandler = eventBusExceptionHandler;
      return this;
    }

    @VisibleForTesting
    public Builder setBugReporter(BugReporter bugReporter) {
      this.bugReporter = bugReporter;
      return this;
    }

    public Builder setActionKeyContext(ActionKeyContext actionKeyContext) {
      this.actionKeyContext = actionKeyContext;
      return this;
    }

    private static PackageSettings getPackageSettings(List<BlazeModule> blazeModules) {
      List<PackageSettings> packageSettingss =
          blazeModules.stream()
              .map(module -> module.getPackageSettings())
              .filter(settings -> settings != null)
              .collect(toImmutableList());
      Preconditions.checkState(
          packageSettingss.size() <= 1, "more than one module defines a PackageSettings");
      return Iterables.getFirst(packageSettingss, DefaultPackageSettings.INSTANCE);
    }

    private static PackageValidator getPackageValidator(List<BlazeModule> blazeModules) {
      List<PackageValidator> packageValidators =
          blazeModules.stream()
              .map(module -> module.getPackageValidator())
              .filter(validator -> validator != null)
              .collect(toImmutableList());
      Preconditions.checkState(
          packageValidators.size() <= 1, "more than one module defined a PackageValidator");
      return Iterables.getFirst(packageValidators, PackageValidator.NOOP_VALIDATOR);
    }
  }

  private static PackageLoadingListener getPackageLoadingListener(
      List<BlazeModule> blazeModules,
      PackageSettings packageBuilderHelper,
      ConfiguredRuleClassProvider ruleClassProvider,
      FileSystem fs) {
    ImmutableList<PackageLoadingListener> listeners =
        blazeModules.stream()
            .map(
                module ->
                    module.getPackageLoadingListener(packageBuilderHelper, ruleClassProvider, fs))
            .filter(validator -> validator != null)
            .collect(toImmutableList());
    return PackageLoadingListener.create(listeners);
  }

  private static int readPidFile(Path pidFile) throws AbruptExitException {
    try {
      return Integer.parseInt(new String(FileSystemUtils.readContentAsLatin1(pidFile)));
    } catch (IOException e) {
      throw createFilesystemExitException(
          "Server pid file read failed: " + e.getMessage(),
          ExitCode.BUILD_FAILURE,
          Filesystem.Code.SERVER_PID_TXT_FILE_READ_FAILURE,
          e);
    } catch (NumberFormatException e) {
      // Invalid contents (not a number) is more likely than not a filesystem issue.
      throw createFilesystemExitException(
          "Server pid file corrupted: " + e.getMessage(),
          ExitCode.BUILD_FAILURE,
          Filesystem.Code.SERVER_PID_TXT_FILE_READ_FAILURE,
          new IOException(e));
    }
  }
}
