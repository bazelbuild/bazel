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

import static com.google.devtools.build.lib.profiler.AutoProfiler.profiledAndLogged;
import static java.nio.charset.StandardCharsets.ISO_8859_1;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Function;
import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Range;
import com.google.common.collect.Sets;
import com.google.common.eventbus.EventBus;
import com.google.common.eventbus.SubscriberExceptionContext;
import com.google.common.eventbus.SubscriberExceptionHandler;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.Uninterruptibles;
import com.google.devtools.build.lib.Constants;
import com.google.devtools.build.lib.actions.cache.ActionCache;
import com.google.devtools.build.lib.actions.cache.CompactPersistentActionCache;
import com.google.devtools.build.lib.actions.cache.NullActionCache;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.BlazeVersionInfo;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.WorkspaceStatusAction;
import com.google.devtools.build.lib.analysis.config.BinTools;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.ConfigurationFactory;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.OutputFilter;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.packages.Preprocessor;
import com.google.devtools.build.lib.packages.RuleClassProvider;
import com.google.devtools.build.lib.pkgcache.PackageManager;
import com.google.devtools.build.lib.profiler.AutoProfiler;
import com.google.devtools.build.lib.profiler.MemoryProfiler;
import com.google.devtools.build.lib.profiler.ProfilePhase;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.Profiler.ProfiledTaskKinds;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.query2.output.OutputFormatter;
import com.google.devtools.build.lib.rules.test.CoverageReportActionFactory;
import com.google.devtools.build.lib.runtime.commands.BuildCommand;
import com.google.devtools.build.lib.runtime.commands.CanonicalizeCommand;
import com.google.devtools.build.lib.runtime.commands.CleanCommand;
import com.google.devtools.build.lib.runtime.commands.DumpCommand;
import com.google.devtools.build.lib.runtime.commands.HelpCommand;
import com.google.devtools.build.lib.runtime.commands.InfoCommand;
import com.google.devtools.build.lib.runtime.commands.MobileInstallCommand;
import com.google.devtools.build.lib.runtime.commands.ProfileCommand;
import com.google.devtools.build.lib.runtime.commands.QueryCommand;
import com.google.devtools.build.lib.runtime.commands.RunCommand;
import com.google.devtools.build.lib.runtime.commands.ShutdownCommand;
import com.google.devtools.build.lib.runtime.commands.TestCommand;
import com.google.devtools.build.lib.runtime.commands.VersionCommand;
import com.google.devtools.build.lib.server.RPCServer;
import com.google.devtools.build.lib.server.ServerCommand;
import com.google.devtools.build.lib.server.signal.InterruptSignalHandler;
import com.google.devtools.build.lib.skyframe.DiffAwareness;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.build.lib.skyframe.SequencedSkyframeExecutorFactory;
import com.google.devtools.build.lib.skyframe.SkyValueDirtinessChecker;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import com.google.devtools.build.lib.skyframe.SkyframeExecutorFactory;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.BlazeClock;
import com.google.devtools.build.lib.util.Clock;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.util.LoggingUtil;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.util.OsUtils;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.util.ThreadUtils;
import com.google.devtools.build.lib.util.io.OutErr;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.JavaIoFileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.UnixFileSystem;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionPriority;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsClassProvider;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;
import com.google.devtools.common.options.OptionsProvider;
import com.google.devtools.common.options.TriState;

import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.logging.Handler;
import java.util.logging.Level;
import java.util.logging.LogRecord;
import java.util.logging.Logger;
import java.util.regex.Pattern;

import javax.annotation.Nullable;

/**
 * The BlazeRuntime class encapsulates the immutable configuration of the current instance. These
 * runtime settings and services are available to most parts of any Blaze application for the
 * duration of the batch run or server lifetime.
 *
 * <p>The parts specific to the current command are stored in {@link CommandEnvironment}.
 */
public final class BlazeRuntime {
  public static final String DO_NOT_BUILD_FILE_NAME = "DO_NOT_BUILD_HERE";

  private static final Pattern suppressFromLog = Pattern.compile(".*(auth|pass|cookie).*",
      Pattern.CASE_INSENSITIVE);

  private static final Logger LOG = Logger.getLogger(BlazeRuntime.class.getName());

  private final Iterable<BlazeModule> blazeModules;
  private final Map<String, BlazeCommand> commandMap = new LinkedHashMap<>();
  private final Clock clock;

  private final PackageFactory packageFactory;
  private final ConfigurationFactory configurationFactory;
  private final ConfiguredRuleClassProvider ruleClassProvider;
  private final TimestampGranularityMonitor timestampGranularityMonitor;

  private final AtomicInteger storedExitCode = new AtomicInteger();

  // We pass this through here to make it available to the MasterLogWriter.
  private final OptionsProvider startupOptionsProvider;

  private final SubscriberExceptionHandler eventBusExceptionHandler;
  private final BinTools binTools;
  private final WorkspaceStatusAction.Factory workspaceStatusActionFactory;
  private final ProjectFile.Provider projectFileProvider;

  // Workspace state (currently exactly one workspace per server)
  private final BlazeDirectories directories;
  private final SkyframeExecutor skyframeExecutor;
  /** The action cache is loaded lazily on the first build command. */
  private ActionCache actionCache;
  /** The execution time range of the previous build command in this server, if any. */
  @Nullable
  private Range<Long> lastExecutionRange = null;

  private BlazeRuntime(BlazeDirectories directories,
      WorkspaceStatusAction.Factory workspaceStatusActionFactory,
      final SkyframeExecutor skyframeExecutor,
      PackageFactory pkgFactory, ConfiguredRuleClassProvider ruleClassProvider,
      ConfigurationFactory configurationFactory, Clock clock,
      OptionsProvider startupOptionsProvider, Iterable<BlazeModule> blazeModules,
      TimestampGranularityMonitor timestampGranularityMonitor,
      SubscriberExceptionHandler eventBusExceptionHandler,
      BinTools binTools, ProjectFile.Provider projectFileProvider,
      Iterable<BlazeCommand> commands) {
    // Server state
    this.blazeModules = blazeModules;
    overrideCommands(commands);

    this.workspaceStatusActionFactory = workspaceStatusActionFactory;
    this.packageFactory = pkgFactory;
    this.binTools = binTools;
    this.projectFileProvider = projectFileProvider;

    this.ruleClassProvider = ruleClassProvider;
    this.configurationFactory = configurationFactory;
    this.clock = clock;
    this.timestampGranularityMonitor = Preconditions.checkNotNull(timestampGranularityMonitor);
    this.startupOptionsProvider = startupOptionsProvider;
    this.eventBusExceptionHandler = eventBusExceptionHandler;

    // Workspace state
    this.directories = directories;
    this.skyframeExecutor = skyframeExecutor;

    if (inWorkspace()) {
      writeOutputBaseReadmeFile();
      writeDoNotBuildHereFile();
    }
    setupExecRoot();
  }

  @Nullable CoverageReportActionFactory getCoverageReportActionFactory() {
    CoverageReportActionFactory firstFactory = null;
    for (BlazeModule module : blazeModules) {
      CoverageReportActionFactory factory = module.getCoverageReportFactory();
      if (factory != null) {
        Preconditions.checkState(firstFactory == null,
            "only one Blaze Module can have a Coverage Report Factory");
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

  final void overrideCommands(Iterable<BlazeCommand> commands) {
    commandMap.clear();
    for (BlazeCommand command : commands) {
      addCommand(command);
    }
    for (BlazeModule module : blazeModules) {
      for (BlazeCommand command : module.getCommands()) {
        addCommand(command);
      }
    }
  }

  public CommandEnvironment initCommand() {
    EventBus eventBus = new EventBus(eventBusExceptionHandler);
    skyframeExecutor.setEventBus(eventBus);
    UUID commandId = UUID.randomUUID();
    return new CommandEnvironment(this, commandId, eventBus);
  }

  private void clearEventBus() {
    // EventBus does not have an unregister() method, so this is how we release memory associated
    // with handlers.
    skyframeExecutor.setEventBus(null);
  }

  /**
   * Conditionally enable profiling.
   */
  private final boolean initProfiler(CommandEnvironment env, CommonCommandOptions options,
      UUID buildID, long execStartTimeNanos) {
    OutputStream out = null;
    boolean recordFullProfilerData = false;
    ProfiledTaskKinds profiledTasks = ProfiledTaskKinds.NONE;

    try {
      if (options.profilePath != null) {
        Path profilePath = getWorkspace().getRelative(options.profilePath);

        recordFullProfilerData = options.recordFullProfilerData;
        out = new BufferedOutputStream(profilePath.getOutputStream(), 1024 * 1024);
        env.getReporter().handle(Event.info("Writing profile data to '" + profilePath + "'"));
        profiledTasks = ProfiledTaskKinds.ALL;
      } else if (options.alwaysProfileSlowOperations) {
        recordFullProfilerData = false;
        out = null;
        profiledTasks = ProfiledTaskKinds.SLOWEST;
      }
      if (profiledTasks != ProfiledTaskKinds.NONE) {
        Profiler.instance().start(profiledTasks, out,
            "Blaze profile for " + getOutputBase() + " at " + new Date()
            + ", build ID: " + buildID,
            recordFullProfilerData, clock, execStartTimeNanos);
        return true;
      }
    } catch (IOException e) {
      env.getReporter().handle(Event.error("Error while creating profile file: " + e.getMessage()));
    }
    return false;
  }

  /**
   * Generates a README file in the output base directory. This README file
   * contains the name of the workspace directory, so that users can figure out
   * which output base directory corresponds to which workspace.
   */
  private void writeOutputBaseReadmeFile() {
    Preconditions.checkNotNull(getWorkspace());
    Path outputBaseReadmeFile = getOutputBase().getRelative("README");
    try {
      FileSystemUtils.writeIsoLatin1(outputBaseReadmeFile, "WORKSPACE: " + getWorkspace(), "",
          "The first line of this file is intentionally easy to parse for various",
          "interactive scripting and debugging purposes.  But please DO NOT write programs",
          "that exploit it, as they will be broken by design: it is not possible to",
          "reverse engineer the set of source trees or the --package_path from the output",
          "tree, and if you attempt it, you will fail, creating subtle and",
          "hard-to-diagnose bugs, that will no doubt get blamed on changes made by the",
          "Blaze team.", "", "This directory was generated by Blaze.",
          "Do not attempt to modify or delete any files in this directory.",
          "Among other issues, Blaze's file system caching assumes that",
          "only Blaze will modify this directory and the files in it,",
          "so if you change anything here you may mess up Blaze's cache.");
    } catch (IOException e) {
      LOG.warning("Couldn't write to '" + outputBaseReadmeFile + "': " + e.getMessage());
    }
  }

  private void writeDoNotBuildHereFile(Path filePath) {
    try {
      FileSystemUtils.createDirectoryAndParents(filePath.getParentDirectory());
      FileSystemUtils.writeContent(filePath, ISO_8859_1, getWorkspace().toString());
    } catch (IOException e) {
      LOG.warning("Couldn't write to '" + filePath + "': " + e.getMessage());
    }
  }

  private void writeDoNotBuildHereFile() {
    Preconditions.checkNotNull(getWorkspace());
    writeDoNotBuildHereFile(getOutputBase().getRelative(DO_NOT_BUILD_FILE_NAME));
    if (startupOptionsProvider.getOptions(BlazeServerStartupOptions.class).deepExecRoot) {
      writeDoNotBuildHereFile(getOutputBase().getRelative("execroot").getRelative(
          DO_NOT_BUILD_FILE_NAME));
    }
  }

  /**
   * Creates the execRoot dir under outputBase.
   */
  private void setupExecRoot() {
    try {
      FileSystemUtils.createDirectoryAndParents(directories.getExecRoot());
    } catch (IOException e) {
      LOG.warning("failed to create execution root '" + directories.getExecRoot() + "': "
          + e.getMessage());
    }
  }

  void recordLastExecutionTime(long commandStartTime) {
    long currentTimeMillis = clock.currentTimeMillis();
    lastExecutionRange = currentTimeMillis >= commandStartTime
        ? Range.closed(commandStartTime, currentTimeMillis)
        : null;
  }

  /**
   * Range that represents the last execution time of a build in millis since epoch.
   */
  @Nullable
  public Range<Long> getLastExecutionTimeRange() {
    return lastExecutionRange;
  }

  public String getWorkspaceName() {
    Path workspace = directories.getWorkspace();
    if (workspace == null) {
      return "";
    }
    return workspace.getBaseName();
  }

  /**
   * Returns the Blaze directories object for this runtime.
   */
  public BlazeDirectories getDirectories() {
    return directories;
  }

  /**
   * Returns the working directory of the server.
   *
   * <p>This is often the first entry on the {@code --package_path}, but not always.
   * Callers should certainly not make this assumption. The Path returned may be null.
   */
  public Path getWorkspace() {
    return directories.getWorkspace();
  }

  /**
   * Returns if the client passed a valid workspace to be used for the build.
   */
  public boolean inWorkspace() {
    return directories.inWorkspace();
  }

  /**
   * Returns the output base directory associated with this Blaze server
   * process. This is the base directory for shared Blaze state as well as tool
   * and strategy specific subdirectories.
   */
  public Path getOutputBase() {
    return directories.getOutputBase();
  }

  /**
   * Returns the output path associated with this Blaze server process..
   */
  public Path getOutputPath() {
    return directories.getOutputPath();
  }

  /**
   * The directory in which blaze stores the server state - that is, the socket
   * file and a log.
   */
  public Path getServerDirectory() {
    return getOutputBase().getChild("server");
  }

  /**
   * Returns the execution root directory associated with this Blaze server
   * process. This is where all input and output files visible to the actual
   * build reside.
   */
  public Path getExecRoot() {
    return directories.getExecRoot();
  }

  public BinTools getBinTools() {
    return binTools;
  }

  /**
   * Returns the skyframe executor.
   */
  public SkyframeExecutor getSkyframeExecutor() {
    return skyframeExecutor;
  }

  /**
   * Returns the package factory.
   */
  public PackageFactory getPackageFactory() {
    return packageFactory;
  }

  public ImmutableList<OutputFormatter> getQueryOutputFormatters() {
    ImmutableList.Builder<OutputFormatter> result = ImmutableList.builder();
    result.addAll(OutputFormatter.getDefaultFormatters());
    for (BlazeModule module : blazeModules) {
      result.addAll(module.getQueryOutputFormatters());
    }

    return result.build();
  }

  /**
   * Returns the package manager.
   */
  public PackageManager getPackageManager() {
    return skyframeExecutor.getPackageManager();
  }

  public WorkspaceStatusAction.Factory getworkspaceStatusActionFactory() {
    return workspaceStatusActionFactory;
  }

  /**
   * Returns the rule class provider.
   */
  public ConfiguredRuleClassProvider getRuleClassProvider() {
    return ruleClassProvider;
  }

  public Iterable<BlazeModule> getBlazeModules() {
    return blazeModules;
  }

  @SuppressWarnings("unchecked")
  public <T extends BlazeModule> T getBlazeModule(Class<T> moduleClass) {
    for (BlazeModule module : blazeModules) {
      if (module.getClass() == moduleClass) {
        return (T) module;
      }
    }

    return null;
  }

  public ConfigurationFactory getConfigurationFactory() {
    return configurationFactory;
  }

  /**
   * Returns reference to the lazily instantiated persistent action cache
   * instance. Note, that method may recreate instance between different build
   * requests, so return value should not be cached.
   */
  public ActionCache getPersistentActionCache(Reporter reporter) throws IOException {
    if (actionCache == null) {
      if (OS.getCurrent() == OS.WINDOWS) {
        // TODO(bazel-team): Add support for a persistent action cache on Windows.
        actionCache = new NullActionCache();
        return actionCache;
      }
      try (AutoProfiler p = profiledAndLogged("Loading action cache", ProfilerTask.INFO, LOG)) {
        try {
          actionCache = new CompactPersistentActionCache(getCacheDirectory(), clock);
        } catch (IOException e) {
          LOG.log(Level.WARNING, "Failed to load action cache: " + e.getMessage(), e);
          LoggingUtil.logToRemote(Level.WARNING, "Failed to load action cache: "
              + e.getMessage(), e);
          reporter.handle(
              Event.error("Error during action cache initialization: " + e.getMessage()
              + ". Corrupted files were renamed to '" + getCacheDirectory() + "/*.bad'. "
              + "Blaze will now reset action cache data, causing a full rebuild"));
          actionCache = new CompactPersistentActionCache(getCacheDirectory(), clock);
        }
      }
    }
    return actionCache;
  }

  /**
   * Removes in-memory caches.
   */
  public void clearCaches() throws IOException {
    skyframeExecutor.resetEvaluator();
    actionCache = null;
    FileSystemUtils.deleteTree(getCacheDirectory());
  }

  /**
   * Returns the TimestampGranularityMonitor. The same monitor object is used
   * across multiple Blaze commands, but it doesn't hold any persistent state
   * across different commands.
   */
  public TimestampGranularityMonitor getTimestampGranularityMonitor() {
    return timestampGranularityMonitor;
  }

  /**
   * Returns path to the cache directory. Path must be inside output base to
   * ensure that users can run concurrent instances of blaze in different
   * clients without attempting to concurrently write to the same action cache
   * on disk, which might not be safe.
   */
  private Path getCacheDirectory() {
    return getOutputBase().getChild("action_cache");
  }

  /**
   * Returns a provider for project file objects. Can be null if no such provider was set by any of
   * the modules.
   */
  @Nullable
  public ProjectFile.Provider getProjectFileProvider() {
    return projectFileProvider;
  }

  /**
   * Hook method called by the BlazeCommandDispatcher prior to the dispatch of
   * each command.
   *
   * @param options The CommonCommandOptions used by every command.
   * @throws AbruptExitException if this command is unsuitable to be run as specified
   */
  void beforeCommand(CommandEnvironment env, CommonCommandOptions options, long execStartTimeNanos)
      throws AbruptExitException {
    // Conditionally enable profiling
    // We need to compensate for launchTimeNanos (measurements taken outside of the jvm).
    long startupTimeNanos = options.startupTime * 1000000L;
    if (initProfiler(env, options, env.getCommandId(), execStartTimeNanos - startupTimeNanos)) {
      Profiler profiler = Profiler.instance();

      // Instead of logEvent() we're calling the low level function to pass the timings we took in
      // the launcher. We're setting the INIT phase marker so that it follows immediately the LAUNCH
      // phase.
      profiler.logSimpleTaskDuration(execStartTimeNanos - startupTimeNanos, 0, ProfilerTask.PHASE,
          ProfilePhase.LAUNCH.description);
      profiler.logSimpleTaskDuration(execStartTimeNanos, 0, ProfilerTask.PHASE,
          ProfilePhase.INIT.description);
    }

    if (options.memoryProfilePath != null) {
      Path memoryProfilePath = env.getWorkingDirectory().getRelative(options.memoryProfilePath);
      try {
        MemoryProfiler.instance().start(memoryProfilePath.getOutputStream());
      } catch (IOException e) {
        env.getReporter().handle(
            Event.error("Error while creating memory profile file: " + e.getMessage()));
      }
    }

    // Initialize exit code to dummy value for afterCommand.
    storedExitCode.set(ExitCode.RESERVED.getNumericExitCode());
  }

  /**
   * Posts the {@link CommandCompleteEvent}, so that listeners can tidy up. Called by {@link
   * #afterCommand}, and by BugReport when crashing from an exception in an async thread.
   */
  public void notifyCommandComplete(int exitCode) {
    if (!storedExitCode.compareAndSet(ExitCode.RESERVED.getNumericExitCode(), exitCode)) {
      // This command has already been called, presumably because there is a race between the main
      // thread and a worker thread that crashed. Don't try to arbitrate the dispute. If the main
      // thread won the race (unlikely, but possible), this may be incorrectly logged as a success.
      return;
    }
    skyframeExecutor.getEventBus().post(new CommandCompleteEvent(exitCode));
  }

  /**
   * Hook method called by the BlazeCommandDispatcher after the dispatch of each
   * command.
   */
  @VisibleForTesting
  public void afterCommand(CommandEnvironment env, int exitCode) {
    // Remove any filters that the command might have added to the reporter.
    env.getReporter().setOutputFilter(OutputFilter.OUTPUT_EVERYTHING);

    notifyCommandComplete(exitCode);

    for (BlazeModule module : blazeModules) {
      module.afterCommand();
    }

    clearEventBus();

    try {
      Profiler.instance().stop();
      MemoryProfiler.instance().stop();
    } catch (IOException e) {
      env.getReporter().handle(Event.error("Error while writing profile file: " + e.getMessage()));
    }
  }

  // Make sure we keep a strong reference to this logger, so that the
  // configuration isn't lost when the gc kicks in.
  private static Logger templateLogger = Logger.getLogger("com.google.devtools.build");

  /**
   * Configures "com.google.devtools.build.*" loggers to the given
   *  {@code level}. Note: This code relies on static state.
   */
  public static void setupLogging(Level level) {
    templateLogger.setLevel(level);
    templateLogger.info("Log level: " + templateLogger.getLevel());
  }

  /**
   * Returns the Clock-instance used for the entire build. Before,
   * individual classes (such as Profiler) used to specify the type
   * of clock (e.g. EpochClock) they wanted to use. This made it
   * difficult to get Blaze working on Windows as some of the clocks
   * available for Linux aren't (directly) available on Windows.
   * Setting the Blaze-wide clock upon construction of BlazeRuntime
   * allows injecting whatever Clock instance should be used from
   * BlazeMain.
   *
   * @return The Blaze-wide clock
   */
  public Clock getClock() {
    return clock;
  }

  public OptionsProvider getStartupOptionsProvider() {
    return startupOptionsProvider;
  }

  /**
   * An array of String values useful if Blaze crashes.
   * For now, just returns the size of the action cache and the build id.
   */
  public String[] getCrashData(CommandEnvironment env) {
    return new String[]{
        getFileSizeString(CompactPersistentActionCache.cacheFile(getCacheDirectory()),
                          "action cache"),
        env.getCommandId() + " (build id)",
    };
  }

  private String getFileSizeString(Path path, String type) {
    try {
      return String.format("%d bytes (%s)", path.getFileSize(), type);
    } catch (IOException e) {
      return String.format("unknown file size (%s)", type);
    }
  }

  public Map<String, BlazeCommand> getCommandMap() {
    return commandMap;
  }

  public void shutdown() {
    for (BlazeModule module : blazeModules) {
      module.blazeShutdown();
    }
  }

  /**
   * Returns the defaults package for the default settings. Should only be called by commands that
   * do <i>not</i> process {@link BuildOptions}, since build options can alter the contents of the
   * defaults package, which will not be reflected here.
   */
  public String getDefaultsPackageContent() {
    return ruleClassProvider.getDefaultsPackageContent();
  }

  /**
   * Returns the defaults package for the given options taken from an optionsProvider.
   */
  public String getDefaultsPackageContent(OptionsClassProvider optionsProvider) {
    return ruleClassProvider.getDefaultsPackageContent(optionsProvider);
  }

  /**
   * Creates a BuildOptions class for the given options taken from an optionsProvider.
   */
  public BuildOptions createBuildOptions(OptionsClassProvider optionsProvider) {
    return ruleClassProvider.createBuildOptions(optionsProvider);
  }

  /**
   * An EventBus exception handler that will report the exception to a remote server, if a
   * handler is registered.
   */
  public static final class RemoteExceptionHandler implements SubscriberExceptionHandler {
    @Override
    public void handleException(Throwable exception, SubscriberExceptionContext context) {
      LOG.log(Level.SEVERE, "Failure in EventBus subscriber", exception);
      LoggingUtil.logToRemote(Level.SEVERE, "Failure in EventBus subscriber.", exception);
    }
  }

  /**
   * An EventBus exception handler that will call BugReport.handleCrash exiting
   * the current thread.
   */
  public static final class BugReportingExceptionHandler implements SubscriberExceptionHandler {
    @Override
    public void handleException(Throwable exception, SubscriberExceptionContext context) {
      BugReport.handleCrash(exception);
    }
  }

  /**
   * Main method for the Blaze server startup. Note: This method logs
   * exceptions to remote servers. Do not add this to a unittest.
   */
  public static void main(Iterable<Class<? extends BlazeModule>> moduleClasses, String[] args) {
    setupUncaughtHandler(args);
    List<BlazeModule> modules = createModules(moduleClasses);
    // blaze.cc will put --batch first if the user set it.
    if (args.length >= 1 && args[0].equals("--batch")) {
      // Run Blaze in batch mode.
      System.exit(batchMain(modules, args));
    }
    LOG.info("Starting Blaze server with args " + Arrays.toString(args));
    try {
      // Run Blaze in server mode.
      System.exit(serverMain(modules, OutErr.SYSTEM_OUT_ERR, args));
    } catch (RuntimeException | Error e) { // A definite bug...
      BugReport.printBug(OutErr.SYSTEM_OUT_ERR, e);
      BugReport.sendBugReport(e, Arrays.asList(args));
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
        BlazeModule module = moduleClass.newInstance();
        result.add(module);
      } catch (Throwable e) {
        throw new IllegalStateException("Cannot instantiate module " + moduleClass.getName(), e);
      }
    }

    return result.build();
  }

  /**
   * Generates a string form of a request to be written to the logs,
   * filtering the user environment to remove anything that looks private.
   * The current filter criteria removes any variable whose name includes
   * "auth", "pass", or "cookie".
   *
   * @param requestStrings
   * @return the filtered request to write to the log.
   */
  @VisibleForTesting
  public static String getRequestLogString(List<String> requestStrings) {
    StringBuilder buf = new StringBuilder();
    buf.append('[');
    String sep = "";
    for (String s : requestStrings) {
      buf.append(sep);
      if (s.startsWith("--client_env")) {
        int varStart = "--client_env=".length();
        int varEnd = s.indexOf('=', varStart);
        String varName = s.substring(varStart, varEnd);
        if (suppressFromLog.matcher(varName).matches()) {
          buf.append("--client_env=");
          buf.append(varName);
          buf.append("=__private_value_removed__");
        } else {
          buf.append(s);
        }
      } else {
        buf.append(s);
      }
      sep = ", ";
    }
    buf.append(']');
    return buf.toString();
  }

  /**
   * Command line options split in to two parts: startup options and everything else.
   */
  @VisibleForTesting
  static class CommandLineOptions {
    private final List<String> startupArgs;
    private final List<String> otherArgs;

    CommandLineOptions(List<String> startupArgs, List<String> otherArgs) {
      this.startupArgs = ImmutableList.copyOf(startupArgs);
      this.otherArgs = ImmutableList.copyOf(otherArgs);
    }

    public List<String> getStartupArgs() {
      return startupArgs;
    }

    public List<String> getOtherArgs() {
      return otherArgs;
    }
  }

  /**
   * Splits given arguments into two lists - arguments matching options defined in this class
   * and everything else, while preserving order in each list.
   */
  static CommandLineOptions splitStartupOptions(
      Iterable<BlazeModule> modules, String... args) {
    List<String> prefixes = new ArrayList<>();
    List<Field> startupFields = Lists.newArrayList();
    for (Class<? extends OptionsBase> defaultOptions
      : BlazeCommandUtils.getStartupOptions(modules)) {
      startupFields.addAll(ImmutableList.copyOf(defaultOptions.getFields()));
    }

    for (Field field : startupFields) {
      if (field.isAnnotationPresent(Option.class)) {
        prefixes.add("--" + field.getAnnotation(Option.class).name());
        if (field.getType() == boolean.class || field.getType() == TriState.class) {
          prefixes.add("--no" + field.getAnnotation(Option.class).name());
        }
      }
    }

    List<String> startupArgs = new ArrayList<>();
    List<String> otherArgs = Lists.newArrayList(args);

    for (Iterator<String> argi = otherArgs.iterator(); argi.hasNext(); ) {
      String arg = argi.next();
      if (!arg.startsWith("--")) {
        break;  // stop at command - all startup options would be specified before it.
      }
      for (String prefix : prefixes) {
        if (arg.startsWith(prefix)) {
          startupArgs.add(arg);
          argi.remove();
          break;
        }
      }
    }
    return new CommandLineOptions(startupArgs, otherArgs);
  }

  private static void captureSigint() {
    final Thread mainThread = Thread.currentThread();
    final AtomicInteger numInterrupts = new AtomicInteger();

    final Runnable interruptWatcher = new Runnable() {
      @Override
      public void run() {
        int count = 0;
        // Not an actual infinite loop because it's run in a daemon thread.
        while (true) {
          count++;
          Uninterruptibles.sleepUninterruptibly(10, TimeUnit.SECONDS);
          LOG.warning("Slow interrupt number " + count + " in batch mode");
          ThreadUtils.warnAboutSlowInterrupt();
        }
      }
    };

    new InterruptSignalHandler() {
      @Override
      public void run() {
        LOG.info("User interrupt");
        OutErr.SYSTEM_OUT_ERR.printErrLn("Blaze received an interrupt");
        mainThread.interrupt();

        int curNumInterrupts = numInterrupts.incrementAndGet();
        if (curNumInterrupts == 1) {
          Thread interruptWatcherThread = new Thread(interruptWatcher, "interrupt-watcher");
          interruptWatcherThread.setDaemon(true);
          interruptWatcherThread.start();
        } else if (curNumInterrupts == 2) {
          LOG.warning("Second --batch interrupt: Reverting to JVM SIGINT handler");
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
    captureSigint();
    CommandLineOptions commandLineOptions = splitStartupOptions(modules, args);
    LOG.info("Running Blaze in batch mode with startup args "
        + commandLineOptions.getStartupArgs());

    BlazeRuntime runtime;
    try {
      runtime = newRuntime(modules, parseOptions(modules, commandLineOptions.getStartupArgs()));
    } catch (OptionsParsingException e) {
      OutErr.SYSTEM_OUT_ERR.printErr(e.getMessage());
      return ExitCode.COMMAND_LINE_ERROR.getNumericExitCode();
    } catch (AbruptExitException e) {
      OutErr.SYSTEM_OUT_ERR.printErr(e.getMessage());
      return e.getExitCode().getNumericExitCode();
    }

    BlazeCommandDispatcher dispatcher = new BlazeCommandDispatcher(runtime);

    try {
      LOG.info(getRequestLogString(commandLineOptions.getOtherArgs()));
      return dispatcher.exec(commandLineOptions.getOtherArgs(), OutErr.SYSTEM_OUT_ERR,
          runtime.getClock().currentTimeMillis());
    } catch (BlazeCommandDispatcher.ShutdownBlazeServerException e) {
      return e.getExitStatus();
    } finally {
      runtime.shutdown();
      dispatcher.shutdown();
    }
  }

  /**
   * A main method that does not send email. The return value indicates the desired exit status of
   * the program.
   */
  private static int serverMain(Iterable<BlazeModule> modules, OutErr outErr, String[] args) {
    try {
      createBlazeRPCServer(modules, Arrays.asList(args)).serve();
      return ExitCode.SUCCESS.getNumericExitCode();
    } catch (OptionsParsingException e) {
      outErr.printErr(e.getMessage());
      return ExitCode.COMMAND_LINE_ERROR.getNumericExitCode();
    } catch (IOException e) {
      outErr.printErr("I/O Error: " + e.getMessage());
      return ExitCode.BUILD_FAILURE.getNumericExitCode();
    } catch (AbruptExitException e) {
      outErr.printErr(e.getMessage());
      return e.getExitCode().getNumericExitCode();
    }
  }

  private static FileSystem fileSystemImplementation() {
    if ("0".equals(System.getProperty("io.bazel.UnixFileSystem"))) {
      // Ignore UnixFileSystem, to be used for bootstrapping.
      return new JavaIoFileSystem();
    }
    // The JNI-based UnixFileSystem is faster, but on Windows it is not available.
    return OS.getCurrent() == OS.WINDOWS ? new JavaIoFileSystem() : new UnixFileSystem();
  }

  /**
   * Creates and returns a new Blaze RPCServer. Call {@link RPCServer#serve()} to start the server.
   */
  private static RPCServer createBlazeRPCServer(Iterable<BlazeModule> modules, List<String> args)
      throws IOException, OptionsParsingException, AbruptExitException {
    OptionsProvider options = parseOptions(modules, args);
    BlazeServerStartupOptions startupOptions = options.getOptions(BlazeServerStartupOptions.class);

    final BlazeRuntime runtime = newRuntime(modules, options);
    final BlazeCommandDispatcher dispatcher = new BlazeCommandDispatcher(runtime);

    final ServerCommand blazeCommand;

    // Adaptor from RPC mechanism to BlazeCommandDispatcher:
    blazeCommand = new ServerCommand() {
      private boolean shutdown = false;

      @Override
      public int exec(List<String> args, OutErr outErr, long firstContactTime) {
        LOG.info(getRequestLogString(args));

        try {
          return dispatcher.exec(args, outErr, firstContactTime);
        } catch (BlazeCommandDispatcher.ShutdownBlazeServerException e) {
          if (e.getCause() != null) {
            StringWriter message = new StringWriter();
            message.write("Shutting down due to exception:\n");
            PrintWriter writer = new PrintWriter(message, true);
            e.printStackTrace(writer);
            writer.flush();
            LOG.severe(message.toString());
          }
          shutdown = true;
          runtime.shutdown();
          dispatcher.shutdown();
          return e.getExitStatus();
        }
      }

      @Override
      public boolean shutdown() {
        return shutdown;
      }
    };

    RPCServer server = RPCServer.newServerWith(runtime.getClock(), blazeCommand,
        runtime.getServerDirectory(), runtime.getWorkspace(), startupOptions.maxIdleSeconds);
    return server;
  }

  private static Function<String, String> sourceFunctionForMap(final Map<String, String> map) {
    return new Function<String, String>() {
      @Override
      public String apply(String input) {
        if (!map.containsKey(input)) {
          return "default";
        }

        if (map.get(input).isEmpty()) {
          return "command line";
        }

        return map.get(input);
      }
    };
  }

  /**
   * Parses the command line arguments into a {@link OptionsParser} object.
   *
   *  <p>This function needs to parse the --option_sources option manually so that the real option
   * parser can set the source for every option correctly. If that cannot be parsed or is missing,
   * we just report an unknown source for every startup option.
   */
  private static OptionsProvider parseOptions(
      Iterable<BlazeModule> modules, List<String> args) throws OptionsParsingException {
    Set<Class<? extends OptionsBase>> optionClasses = Sets.newHashSet();
    optionClasses.addAll(BlazeCommandUtils.getStartupOptions(modules));
    // First parse the command line so that we get the option_sources argument
    OptionsParser parser = OptionsParser.newOptionsParser(optionClasses);
    parser.setAllowResidue(false);
    parser.parse(OptionPriority.COMMAND_LINE, null, args);
    Function<? super String, String> sourceFunction =
        sourceFunctionForMap(parser.getOptions(BlazeServerStartupOptions.class).optionSources);

    // Then parse the command line again, this time with the correct option sources
    parser = OptionsParser.newOptionsParser(optionClasses);
    parser.setAllowResidue(false);
    parser.parseWithSourceFunction(OptionPriority.COMMAND_LINE, sourceFunction, args);
    return parser;
  }

  /**
   * Creates a new blaze runtime, given the install and output base directories.
   *
   * <p>Note: This method can and should only be called once per startup, as it also creates the
   * filesystem object that will be used for the runtime. So it should only ever be called from the
   * main method of the Blaze program.
   *
   * @param options Blaze startup options.
   *
   * @return a new BlazeRuntime instance initialized with the given filesystem and directories, and
   *         an error string that, if not null, describes a fatal initialization failure that makes
   *         this runtime unsuitable for real commands
   */
  private static BlazeRuntime newRuntime(
      Iterable<BlazeModule> blazeModules, OptionsProvider options) throws AbruptExitException {
    for (BlazeModule module : blazeModules) {
      module.globalInit(options);
    }

    BlazeServerStartupOptions startupOptions = options.getOptions(BlazeServerStartupOptions.class);
    PathFragment workspaceDirectory = startupOptions.workspaceDirectory;
    PathFragment installBase = startupOptions.installBase;
    PathFragment outputBase = startupOptions.outputBase;

    OsUtils.maybeForceJNI(installBase);  // Must be before first use of JNI.

    // From the point of view of the Java program --install_base and --output_base
    // are mandatory options, despite the comment in their declarations.
    if (installBase == null || !installBase.isAbsolute()) { // (includes "" default case)
      throw new IllegalArgumentException(
          "Bad --install_base option specified: '" + installBase + "'");
    }
    if (outputBase != null && !outputBase.isAbsolute()) { // (includes "" default case)
      throw new IllegalArgumentException(
          "Bad --output_base option specified: '" + outputBase + "'");
    }

    PathFragment outputPathFragment = BlazeDirectories.outputPathFromOutputBase(
        outputBase, workspaceDirectory, startupOptions.deepExecRoot);
    FileSystem fs = null;
    for (BlazeModule module : blazeModules) {
      FileSystem moduleFs = module.getFileSystem(options, outputPathFragment);
      if (moduleFs != null) {
        Preconditions.checkState(fs == null, "more than one module returns a file system");
        fs = moduleFs;
      }
    }

    if (fs == null) {
      fs = fileSystemImplementation();
    }
    Path.setFileSystemForSerialization(fs);

    Path installBasePath = fs.getPath(installBase);
    Path outputBasePath = fs.getPath(outputBase);
    Path workspaceDirectoryPath = null;
    if (!workspaceDirectory.equals(PathFragment.EMPTY_FRAGMENT)) {
      workspaceDirectoryPath = fs.getPath(workspaceDirectory);
    }

    BlazeDirectories directories =
        new BlazeDirectories(installBasePath, outputBasePath, workspaceDirectoryPath,
                             startupOptions.deepExecRoot, startupOptions.installMD5);

    Clock clock = BlazeClock.instance();

    BinTools binTools;
    try {
      binTools = BinTools.forProduction(directories);
    } catch (IOException e) {
      throw new AbruptExitException(
          "Cannot enumerate embedded binaries: " + e.getMessage(),
          ExitCode.LOCAL_ENVIRONMENTAL_ERROR);
    }

    BlazeRuntime.Builder runtimeBuilder = new BlazeRuntime.Builder().setDirectories(directories)
        .setStartupOptionsProvider(options)
        .setBinTools(binTools)
        .setClock(clock)
        // TODO(bazel-team): Make BugReportingExceptionHandler the default.
        // See bug "Make exceptions in EventBus subscribers fatal"
        .setEventBusExceptionHandler(
            startupOptions.fatalEventBusExceptions || !BlazeVersionInfo.instance().isReleasedBlaze()
                ? new BlazeRuntime.BugReportingExceptionHandler()
                : new BlazeRuntime.RemoteExceptionHandler());

    if (System.getenv("TEST_TMPDIR") != null
        && System.getenv("NO_CRASH_ON_LOGGING_IN_TEST") == null) {
      LoggingUtil.installRemoteLogger(getTestCrashLogger());
    }

    for (BlazeModule blazeModule : blazeModules) {
      runtimeBuilder.addBlazeModule(blazeModule);
    }
    runtimeBuilder.addCommands(getBuiltinCommandList());

    BlazeRuntime runtime = runtimeBuilder.build();
    AutoProfiler.setClock(runtime.getClock());
    BugReport.setRuntime(runtime);
    return runtime;
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
            Throwable e = record.getThrown();
            String message =
                record.getSourceClassName()
                    + "#"
                    + record.getSourceMethodName()
                    + ": "
                    + record.getMessage();
            if (e == null) {
              throw new IllegalStateException(message);
            } else {
              throw new IllegalStateException(message, e);
            }
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
   * Make sure async threads cannot be orphaned. This method makes sure bugs are reported to
   * telemetry and the proper exit code is reported.
   */
  private static void setupUncaughtHandler(final String[] args) {
    Thread.setDefaultUncaughtExceptionHandler(new Thread.UncaughtExceptionHandler() {
      @Override
      public void uncaughtException(Thread thread, Throwable throwable) {
        try {
          BugReport.handleCrash(throwable, args);
        } catch (Throwable t) {
          System.err.println("An exception was caught in " + Constants.PRODUCT_NAME + "'s "
              + "UncaughtExceptionHandler, a bug report may not have been filed.");

          System.err.println("Original uncaught exception:");
          throwable.printStackTrace(System.err);

          System.err.println("Exception encountered during UncaughtExceptionHandler:");
          t.printStackTrace(System.err);

          Runtime.getRuntime().halt(BugReport.getExitCodeForThrowable(throwable));
        }
      }
    });
  }


  /**
   * Returns an immutable list containing new instances of each Blaze command.
   */
  @VisibleForTesting
  public static List<BlazeCommand> getBuiltinCommandList() {
    return ImmutableList.of(
        new BuildCommand(),
        new CanonicalizeCommand(),
        new CleanCommand(),
        new DumpCommand(),
        new HelpCommand(),
        new InfoCommand(),
        new MobileInstallCommand(),
        new ProfileCommand(),
        new QueryCommand(),
        new RunCommand(),
        new ShutdownCommand(),
        new TestCommand(),
        new VersionCommand());
  }

  /**
   * A builder for {@link BlazeRuntime} objects. The only required fields are the {@link
   * BlazeDirectories}, and the {@link RuleClassProvider} (except for testing). All other fields
   * have safe default values.
   *
   * <p>If a {@link ConfigurationFactory} is set, then the builder ignores the host system flag.
   * <p>The default behavior of the BlazeRuntime's EventBus is to exit when a subscriber throws
   * an exception. Please plan appropriately.
   */
  public static class Builder {

    private BlazeDirectories directories;
    private ConfigurationFactory configurationFactory;
    private Clock clock;
    private OptionsProvider startupOptionsProvider;
    private final List<BlazeModule> blazeModules = new ArrayList<>();
    private SubscriberExceptionHandler eventBusExceptionHandler =
        new RemoteExceptionHandler();
    private BinTools binTools;
    private UUID instanceId;
    private final List<BlazeCommand> commands = new ArrayList<>();

    public BlazeRuntime build() throws AbruptExitException {
      Preconditions.checkNotNull(directories);
      Preconditions.checkNotNull(startupOptionsProvider);

      Clock clock = (this.clock == null) ? BlazeClock.instance() : this.clock;
      UUID instanceId =  (this.instanceId == null) ? UUID.randomUUID() : this.instanceId;

      Preconditions.checkNotNull(clock);
      TimestampGranularityMonitor timestampMonitor = new TimestampGranularityMonitor(clock);

      Preprocessor.Factory.Supplier preprocessorFactorySupplier = null;
      SkyframeExecutorFactory skyframeExecutorFactory = null;
      for (BlazeModule module : blazeModules) {
        module.blazeStartup(startupOptionsProvider,
            BlazeVersionInfo.instance(), instanceId, directories, clock);
        Preprocessor.Factory.Supplier modulePreprocessorFactorySupplier =
            module.getPreprocessorFactorySupplier();
        if (modulePreprocessorFactorySupplier != null) {
          Preconditions.checkState(preprocessorFactorySupplier == null,
              "more than one module defines a preprocessor factory supplier");
          preprocessorFactorySupplier = modulePreprocessorFactorySupplier;
        }
        SkyframeExecutorFactory skyFactory = module.getSkyframeExecutorFactory();
        if (skyFactory != null) {
          Preconditions.checkState(skyframeExecutorFactory == null,
              "At most one skyframe factory supported. But found two: %s and %s", skyFactory,
              skyframeExecutorFactory);
          skyframeExecutorFactory = skyFactory;
        }
      }
      if (skyframeExecutorFactory == null) {
        skyframeExecutorFactory = new SequencedSkyframeExecutorFactory();
      }
      if (preprocessorFactorySupplier == null) {
        preprocessorFactorySupplier = Preprocessor.Factory.Supplier.NullSupplier.INSTANCE;
      }

      ConfiguredRuleClassProvider.Builder ruleClassBuilder =
          new ConfiguredRuleClassProvider.Builder();
      for (BlazeModule module : blazeModules) {
        module.initializeRuleClasses(ruleClassBuilder);
      }

      Map<String, String> platformRegexps = null;
      {
        ImmutableMap.Builder<String, String> builder = new ImmutableMap.Builder<>();
        for (BlazeModule module : blazeModules) {
          builder.putAll(module.getPlatformSetRegexps());
        }
        platformRegexps = builder.build();
        if (platformRegexps.isEmpty()) {
          platformRegexps = null; // Use the default.
        }
      }

      Iterable<DiffAwareness.Factory> diffAwarenessFactories;
      {
        ImmutableList.Builder<DiffAwareness.Factory> builder = new ImmutableList.Builder<>();
        boolean watchFS = startupOptionsProvider != null
            && startupOptionsProvider.getOptions(BlazeServerStartupOptions.class).watchFS;
        for (BlazeModule module : blazeModules) {
          builder.addAll(module.getDiffAwarenessFactories(watchFS));
        }
        diffAwarenessFactories = builder.build();
      }

      // Merge filters from Blaze modules that allow some action inputs to be missing.
      Predicate<PathFragment> allowedMissingInputs = null;
      for (BlazeModule module : blazeModules) {
        Predicate<PathFragment> modulePredicate = module.getAllowedMissingInputs();
        if (modulePredicate != null) {
          Preconditions.checkArgument(allowedMissingInputs == null,
              "More than one Blaze module allows missing inputs.");
          allowedMissingInputs = modulePredicate;
        }
      }
      if (allowedMissingInputs == null) {
        allowedMissingInputs = Predicates.alwaysFalse();
      }

      ConfiguredRuleClassProvider ruleClassProvider = ruleClassBuilder.build();
      WorkspaceStatusAction.Factory workspaceStatusActionFactory = null;
      for (BlazeModule module : blazeModules) {
        WorkspaceStatusAction.Factory candidate = module.getWorkspaceStatusActionFactory();
        if (candidate != null) {
          Preconditions.checkState(workspaceStatusActionFactory == null,
              "more than one module defines a workspace status action factory");
          workspaceStatusActionFactory = candidate;
        }
      }

      List<PackageFactory.EnvironmentExtension> extensions = new ArrayList<>();
      for (BlazeModule module : blazeModules) {
        extensions.add(module.getPackageEnvironmentExtension());
      }

      // We use an immutable map builder for the nice side effect that it throws if a duplicate key
      // is inserted.
      ImmutableMap.Builder<SkyFunctionName, SkyFunction> skyFunctions = ImmutableMap.builder();
      for (BlazeModule module : blazeModules) {
        skyFunctions.putAll(module.getSkyFunctions(directories));
      }

      ImmutableList.Builder<PrecomputedValue.Injected> precomputedValues = ImmutableList.builder();
      for (BlazeModule module : blazeModules) {
        precomputedValues.addAll(module.getPrecomputedSkyframeValues());
      }

      ImmutableList.Builder<SkyValueDirtinessChecker> customDirtinessCheckers =
          ImmutableList.builder();
      for (BlazeModule module : blazeModules) {
        customDirtinessCheckers.addAll(module.getCustomDirtinessCheckers());
      }

      final PackageFactory pkgFactory =
          new PackageFactory(ruleClassProvider, platformRegexps, extensions);
      SkyframeExecutor skyframeExecutor =
          skyframeExecutorFactory.create(
              pkgFactory,
              timestampMonitor,
              directories,
              binTools,
              workspaceStatusActionFactory,
              ruleClassProvider.getBuildInfoFactories(),
              diffAwarenessFactories,
              allowedMissingInputs,
              preprocessorFactorySupplier,
              skyFunctions.build(),
              precomputedValues.build(),
              customDirtinessCheckers.build());

      if (configurationFactory == null) {
        configurationFactory = new ConfigurationFactory(
            ruleClassProvider.getConfigurationCollectionFactory(),
            ruleClassProvider.getConfigurationFragments());
      }

      ProjectFile.Provider projectFileProvider = null;
      for (BlazeModule module : blazeModules) {
        ProjectFile.Provider candidate = module.createProjectFileProvider();
        if (candidate != null) {
          Preconditions.checkState(projectFileProvider == null,
              "more than one module defines a project file provider");
          projectFileProvider = candidate;
        }
      }

      return new BlazeRuntime(directories, workspaceStatusActionFactory, skyframeExecutor,
          pkgFactory, ruleClassProvider, configurationFactory,
          clock, startupOptionsProvider, ImmutableList.copyOf(blazeModules),
          timestampMonitor, eventBusExceptionHandler, binTools, projectFileProvider, commands);
    }

    public Builder setBinTools(BinTools binTools) {
      this.binTools = binTools;
      return this;
    }

    public Builder setDirectories(BlazeDirectories directories) {
      this.directories = directories;
      return this;
    }

    /**
     * Creates and sets a new {@link BlazeDirectories} instance with the given
     * parameters.
     */
    public Builder setDirectories(Path installBase, Path outputBase,
        Path workspace) {
      this.directories = new BlazeDirectories(installBase, outputBase, workspace);
      return this;
    }

    public Builder setConfigurationFactory(ConfigurationFactory configurationFactory) {
      this.configurationFactory = configurationFactory;
      return this;
    }

    public Builder setClock(Clock clock) {
      this.clock = clock;
      return this;
    }

    public Builder setStartupOptionsProvider(OptionsProvider startupOptionsProvider) {
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

    public Builder addCommands(BlazeCommand... commands) {
      this.commands.addAll(Arrays.asList(commands));
      return this;
    }

    public Builder addCommands(Iterable<BlazeCommand> commands) {
      Iterables.addAll(this.commands, commands);
      return this;
    }
  }
}
