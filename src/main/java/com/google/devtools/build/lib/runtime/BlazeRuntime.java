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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Function;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Lists;
import com.google.common.eventbus.SubscriberExceptionContext;
import com.google.common.eventbus.SubscriberExceptionHandler;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.Uninterruptibles;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.BlazeVersionInfo;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.ServerDirectories;
import com.google.devtools.build.lib.analysis.config.BinTools;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.ConfigurationFactory;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.OutputFilter;
import com.google.devtools.build.lib.flags.CommandNameCache;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.packages.RuleClassProvider;
import com.google.devtools.build.lib.profiler.AutoProfiler;
import com.google.devtools.build.lib.profiler.MemoryProfiler;
import com.google.devtools.build.lib.profiler.ProfilePhase;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.Profiler.ProfiledTaskKinds;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.query2.AbstractBlazeQueryEnvironment;
import com.google.devtools.build.lib.query2.QueryEnvironmentFactory;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryFunction;
import com.google.devtools.build.lib.query2.output.OutputFormatter;
import com.google.devtools.build.lib.rules.test.CoverageReportActionFactory;
import com.google.devtools.build.lib.runtime.BlazeCommandDispatcher.LockingMode;
import com.google.devtools.build.lib.runtime.commands.InfoItem;
import com.google.devtools.build.lib.runtime.proto.InvocationPolicyOuterClass.InvocationPolicy;
import com.google.devtools.build.lib.server.RPCServer;
import com.google.devtools.build.lib.server.signal.InterruptSignalHandler;
import com.google.devtools.build.lib.shell.JavaSubprocessFactory;
import com.google.devtools.build.lib.shell.Subprocess;
import com.google.devtools.build.lib.shell.SubprocessBuilder;
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
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.JavaIoFileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.UnixFileSystem;
import com.google.devtools.build.lib.vfs.WindowsFileSystem;
import com.google.devtools.build.lib.windows.WindowsSubprocessFactory;
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
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
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
  private static final Pattern suppressFromLog = Pattern.compile(".*(auth|pass|cookie).*",
      Pattern.CASE_INSENSITIVE);

  private static final Logger LOG = Logger.getLogger(BlazeRuntime.class.getName());

  private final Iterable<BlazeModule> blazeModules;
  private final Map<String, BlazeCommand> commandMap = new LinkedHashMap<>();
  private final Clock clock;

  private final PackageFactory packageFactory;
  private final ConfigurationFactory configurationFactory;
  private final ConfiguredRuleClassProvider ruleClassProvider;
  // For bazel info.
  private final ImmutableMap<String, InfoItem> infoItems;
  // For bazel query.
  private final QueryEnvironmentFactory queryEnvironmentFactory;
  private final ImmutableList<QueryFunction> queryFunctions;
  private final ImmutableList<OutputFormatter> queryOutputFormatters;

  private final AtomicInteger storedExitCode = new AtomicInteger();

  // We pass this through here to make it available to the MasterLogWriter.
  private final OptionsProvider startupOptionsProvider;

  private final ProjectFile.Provider projectFileProvider;
  @Nullable
  private final InvocationPolicy invocationPolicy;
  private final String defaultsPackageContent;
  private final SubscriberExceptionHandler eventBusExceptionHandler;
  private final String productName;

  // Workspace state (currently exactly one workspace per server)
  private BlazeWorkspace workspace;

  private BlazeRuntime(
      QueryEnvironmentFactory queryEnvironmentFactory,
      ImmutableList<QueryFunction> queryFunctions,
      ImmutableList<OutputFormatter> queryOutputFormatters,
      PackageFactory pkgFactory,
      ConfiguredRuleClassProvider ruleClassProvider,
      ConfigurationFactory configurationFactory,
      ImmutableMap<String, InfoItem> infoItems,
      Clock clock,
      OptionsProvider startupOptionsProvider,
      Iterable<BlazeModule> blazeModules,
      SubscriberExceptionHandler eventBusExceptionHandler,
      ProjectFile.Provider projectFileProvider,
      InvocationPolicy invocationPolicy,
      Iterable<BlazeCommand> commands,
      String productName) {
    // Server state
    this.blazeModules = blazeModules;
    overrideCommands(commands);

    this.packageFactory = pkgFactory;
    this.projectFileProvider = projectFileProvider;
    this.invocationPolicy = invocationPolicy;

    this.ruleClassProvider = ruleClassProvider;
    this.configurationFactory = configurationFactory;
    this.infoItems = infoItems;
    this.clock = clock;
    this.startupOptionsProvider = startupOptionsProvider;
    this.queryEnvironmentFactory = queryEnvironmentFactory;
    this.queryFunctions = queryFunctions;
    this.queryOutputFormatters = queryOutputFormatters;
    this.eventBusExceptionHandler = eventBusExceptionHandler;

    this.defaultsPackageContent =
        ruleClassProvider.getDefaultsPackageContent(getInvocationPolicy());
    CommandNameCache.CommandNameCacheInstance.INSTANCE.setCommandNameCache(
        new CommandNameCacheImpl(getCommandMap()));
    this.productName = productName;
  }

  public void initWorkspace(BlazeDirectories directories, BinTools binTools)
      throws AbruptExitException {
    Preconditions.checkState(this.workspace == null);
    WorkspaceBuilder builder = new WorkspaceBuilder(directories, binTools);
    for (BlazeModule module : blazeModules) {
      module.workspaceInit(directories, builder);
    }
    this.workspace = builder.build(
        this, packageFactory, ruleClassProvider, getProductName(), eventBusExceptionHandler);
  }

  @Nullable public CoverageReportActionFactory getCoverageReportActionFactory(
      OptionsClassProvider commandOptions) {
    CoverageReportActionFactory firstFactory = null;
    for (BlazeModule module : blazeModules) {
      CoverageReportActionFactory factory = module.getCoverageReportFactory(commandOptions);
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
  }

  /**
   * Initializes a CommandEnvironment to execute a command in this server.
   *
   * <p>This method should be called from the "main" thread on which the command will execute;
   * that thread will receive interruptions if a module requests an early exit.
   */
  public CommandEnvironment initCommand() {
    return workspace.initCommand();
  }

  @Nullable
  public InvocationPolicy getInvocationPolicy() {
    return invocationPolicy;
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
        Path profilePath = env.getWorkspace().getRelative(options.profilePath);

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
            getProductName() + " profile for " + env.getOutputBase() + " at " + new Date()
            + ", build ID: " + buildID,
            recordFullProfilerData, clock, execStartTimeNanos);
        return true;
      }
    } catch (IOException e) {
      env.getReporter().handle(Event.error("Error while creating profile file: " + e.getMessage()));
    }
    return false;
  }

  public BlazeWorkspace getWorkspace() {
    return workspace;
  }

  /**
   * The directory in which blaze stores the server state - that is, the socket
   * file and a log.
   */
  private Path getServerDirectory() {
    return getWorkspace().getDirectories().getOutputBase().getChild("server");
  }

  public boolean writeCommandLog() {
    return startupOptionsProvider.getOptions(BlazeServerStartupOptions.class).writeCommandLog;
  }

  /**
   * Returns the {@link QueryEnvironmentFactory} that should be used to create a
   * {@link AbstractBlazeQueryEnvironment}, whenever one is needed.
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

  /**
   * Returns the package factory.
   */
  public PackageFactory getPackageFactory() {
    return packageFactory;
  }

  /**
   * Returns the rule class provider.
   */
  public ConfiguredRuleClassProvider getRuleClassProvider() {
    return ruleClassProvider;
  }

  public ImmutableMap<String, InfoItem> getInfoItems() {
    return infoItems;
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
  void notifyCommandComplete(int exitCode) {
    if (!storedExitCode.compareAndSet(ExitCode.RESERVED.getNumericExitCode(), exitCode)) {
      // This command has already been called, presumably because there is a race between the main
      // thread and a worker thread that crashed. Don't try to arbitrate the dispute. If the main
      // thread won the race (unlikely, but possible), this may be incorrectly logged as a success.
      return;
    }
    workspace.getSkyframeExecutor().getEventBus().post(new CommandCompleteEvent(exitCode));
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

    env.getBlazeWorkspace().clearEventBus();

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

  public Map<String, BlazeCommand> getCommandMap() {
    return commandMap;
  }

  /** Invokes {@link BlazeModule#blazeShutdown()} on all registered modules. */
  public void shutdown() {
    for (BlazeModule module : blazeModules) {
      module.blazeShutdown();
    }
  }

  /** Invokes {@link BlazeModule#blazeShutdownOnCrash()} on all registered modules. */
  public void shutdownOnCrash() {
    for (BlazeModule module : blazeModules) {
      module.blazeShutdownOnCrash();
    }
  }

  /**
   * Returns the defaults package for the default settings. Should only be called by commands that
   * do <i>not</i> process {@link BuildOptions}, since build options can alter the contents of the
   * defaults package, which will not be reflected here.
   */
  public String getDefaultsPackageContent() {
    return defaultsPackageContent;
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
        BlazeModule module = moduleClass.getConstructor().newInstance();
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
      protected void onSignal() {
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
      runtime = newRuntime(modules, commandLineOptions.getStartupArgs());
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
          LockingMode.ERROR_OUT, "batch client", runtime.getClock().currentTimeMillis());
    } catch (BlazeCommandDispatcher.ShutdownBlazeServerException e) {
      return e.getExitStatus();
    } catch (InterruptedException e) {
      // This is almost main(), so it's okay to just swallow it. We are exiting soon.
      return ExitCode.INTERRUPTED.getNumericExitCode();
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
    InterruptSignalHandler sigintHandler = null;
    try {
      final RPCServer blazeServer = createBlazeRPCServer(modules, Arrays.asList(args));

      // Register the signal handler.
       sigintHandler = new InterruptSignalHandler() {
        @Override
        protected void onSignal() {
          LOG.severe("User interrupt");
          blazeServer.interrupt();
        }
      };

      blazeServer.serve();
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
    } finally {
      if (sigintHandler != null) {
        sigintHandler.uninstall();
      }
    }
  }

  private static FileSystem fileSystemImplementation() {
    if ("0".equals(System.getProperty("io.bazel.EnableJni"))) {
      // Ignore UnixFileSystem, to be used for bootstrapping.
      return OS.getCurrent() == OS.WINDOWS ? new WindowsFileSystem() : new JavaIoFileSystem();
    }
    // The JNI-based UnixFileSystem is faster, but on Windows it is not available.
    return OS.getCurrent() == OS.WINDOWS ? new WindowsFileSystem() : new UnixFileSystem();
  }

  private static Subprocess.Factory subprocessFactoryImplementation() {
    if (!"0".equals(System.getProperty("io.bazel.EnableJni")) && OS.getCurrent() == OS.WINDOWS) {
      return WindowsSubprocessFactory.INSTANCE;
    } else {
      return JavaSubprocessFactory.INSTANCE;
    }
  }

  /**
   * Creates and returns a new Blaze RPCServer. Call {@link RPCServer#serve()} to start the server.
   */
  private static RPCServer createBlazeRPCServer(
      Iterable<BlazeModule> modules, List<String> args)
      throws IOException, OptionsParsingException, AbruptExitException {
    BlazeRuntime runtime = newRuntime(modules, args);
    BlazeCommandDispatcher dispatcher = new BlazeCommandDispatcher(runtime);
    CommandExecutor commandExecutor = new CommandExecutor(runtime, dispatcher);

    BlazeServerStartupOptions startupOptions =
        runtime.getStartupOptionsProvider().getOptions(BlazeServerStartupOptions.class);
    try {
      // This is necessary so that Bazel kind of works during bootstrapping, at which time the
      // gRPC server is not compiled in so that we don't need gRPC for bootstrapping.
      Class<?> factoryClass = Class.forName(
          "com.google.devtools.build.lib.server.GrpcServerImpl$Factory");
    RPCServer.Factory factory = (RPCServer.Factory) factoryClass.getConstructor().newInstance();
    return factory.create(commandExecutor, runtime.getClock(),
        startupOptions.commandPort, runtime.getServerDirectory(),
        startupOptions.maxIdleSeconds);
    } catch (ReflectiveOperationException | IllegalArgumentException e) {
      throw new AbruptExitException("gRPC server not compiled in", ExitCode.BLAZE_INTERNAL_ERROR);
    }
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
    ImmutableList<Class<? extends OptionsBase>> optionClasses =
        BlazeCommandUtils.getStartupOptions(modules);

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
   * @param args Blaze startup options.
   *
   * @return a new BlazeRuntime instance initialized with the given filesystem and directories, and
   *         an error string that, if not null, describes a fatal initialization failure that makes
   *         this runtime unsuitable for real commands
   */
  private static BlazeRuntime newRuntime(Iterable<BlazeModule> blazeModules, List<String> args)
      throws AbruptExitException, OptionsParsingException {
    OptionsProvider options = parseOptions(blazeModules, args);
    for (BlazeModule module : blazeModules) {
      module.globalInit(options);
    }

    BlazeServerStartupOptions startupOptions = options.getOptions(BlazeServerStartupOptions.class);
    String productName = startupOptions.productName.toLowerCase(Locale.US);

    if (startupOptions.oomMoreEagerlyThreshold != 100) {
      new RetainedHeapLimiter(startupOptions.oomMoreEagerlyThreshold).install();
    }
    if (startupOptions.oomMoreEagerly) {
      new OomSignalHandler();
    }
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

    FileSystem fs = null;
    for (BlazeModule module : blazeModules) {
      FileSystem moduleFs = module.getFileSystem(options);
      if (moduleFs != null) {
        Preconditions.checkState(fs == null, "more than one module returns a file system");
        fs = moduleFs;
      }
    }

    if (fs == null) {
      fs = fileSystemImplementation();
    }

    Path.setFileSystemForSerialization(fs);
    SubprocessBuilder.setSubprocessFactory(subprocessFactoryImplementation());

    Path installBasePath = fs.getPath(installBase);
    Path outputBasePath = fs.getPath(outputBase);
    Path workspaceDirectoryPath = null;
    if (!workspaceDirectory.equals(PathFragment.EMPTY_FRAGMENT)) {
      workspaceDirectoryPath = fs.getPath(workspaceDirectory);
    }

    ServerDirectories serverDirectories =
        new ServerDirectories(installBasePath, outputBasePath, startupOptions.installMD5);
    Clock clock = BlazeClock.instance();
    BlazeRuntime.Builder runtimeBuilder = new BlazeRuntime.Builder()
        .setProductName(productName)
        .setServerDirectories(serverDirectories)
        .setStartupOptionsProvider(options)
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

    runtimeBuilder.addBlazeModule(new BuiltinCommandModule());
    for (BlazeModule blazeModule : blazeModules) {
      runtimeBuilder.addBlazeModule(blazeModule);
    }

    BlazeRuntime runtime = runtimeBuilder.build();

    BlazeDirectories directories =
        new BlazeDirectories(
            serverDirectories, workspaceDirectoryPath, startupOptions.deepExecRoot, productName);
    BinTools binTools;
    try {
      binTools = BinTools.forProduction(directories);
    } catch (IOException e) {
      throw new AbruptExitException(
          "Cannot enumerate embedded binaries: " + e.getMessage(),
          ExitCode.LOCAL_ENVIRONMENTAL_ERROR);
    }
    runtime.initWorkspace(directories, binTools);

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
            System.err.println("Remote logging disabled for testing, forcing abrupt shutdown.");
            System.err.printf("%s#%s: %s\n",
                record.getSourceClassName(),
                record.getSourceMethodName(),
                record.getMessage());

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
   * Make sure async threads cannot be orphaned. This method makes sure bugs are reported to
   * telemetry and the proper exit code is reported.
   */
  private static void setupUncaughtHandler(final String[] args) {
    Thread.setDefaultUncaughtExceptionHandler(
        new Thread.UncaughtExceptionHandler() {
          @Override
          public void uncaughtException(Thread thread, Throwable throwable) {
            BugReport.handleCrash(throwable, args);
          }
        });
  }

  public String getProductName() {
    return productName;
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
    private ServerDirectories serverDirectories;
    private Clock clock;
    private OptionsProvider startupOptionsProvider;
    private final List<BlazeModule> blazeModules = new ArrayList<>();
    private SubscriberExceptionHandler eventBusExceptionHandler = new RemoteExceptionHandler();
    private UUID instanceId;
    private String productName;

    public BlazeRuntime build() throws AbruptExitException {
      Preconditions.checkNotNull(productName);
      Preconditions.checkNotNull(serverDirectories);
      Preconditions.checkNotNull(startupOptionsProvider);
      Clock clock = (this.clock == null) ? BlazeClock.instance() : this.clock;
      UUID instanceId =  (this.instanceId == null) ? UUID.randomUUID() : this.instanceId;

      Preconditions.checkNotNull(clock);

      for (BlazeModule module : blazeModules) {
        module.blazeStartup(startupOptionsProvider,
            BlazeVersionInfo.instance(), instanceId, serverDirectories, clock);
      }
      ServerBuilder serverBuilder = new ServerBuilder();
      serverBuilder.addQueryOutputFormatters(OutputFormatter.getDefaultFormatters());
      for (BlazeModule module : blazeModules) {
        module.serverInit(startupOptionsProvider, serverBuilder);
      }

      ConfiguredRuleClassProvider.Builder ruleClassBuilder =
          new ConfiguredRuleClassProvider.Builder();
      for (BlazeModule module : blazeModules) {
        module.initializeRuleClasses(ruleClassBuilder);
      }

      ConfiguredRuleClassProvider ruleClassProvider = ruleClassBuilder.build();

      List<PackageFactory.EnvironmentExtension> extensions = new ArrayList<>();
      for (BlazeModule module : blazeModules) {
        extensions.add(module.getPackageEnvironmentExtension());
      }

      Package.Builder.Helper packageBuilderHelper = null;
      for (BlazeModule module : blazeModules) {
        Package.Builder.Helper candidateHelper =
            module.getPackageBuilderHelper(ruleClassProvider, serverDirectories.getFileSystem());
        if (candidateHelper != null) {
          Preconditions.checkState(packageBuilderHelper == null,
              "more than one module defines a package builder helper");
          packageBuilderHelper = candidateHelper;
        }
      }
      if (packageBuilderHelper == null) {
        packageBuilderHelper = Package.Builder.DefaultHelper.INSTANCE;
      }

      PackageFactory packageFactory =
          new PackageFactory(
              ruleClassProvider,
              ruleClassBuilder.getPlatformRegexps(),
              serverBuilder.getAttributeContainerFactory(),
              extensions,
              BlazeVersionInfo.instance().getVersion(),
              packageBuilderHelper);

      ConfigurationFactory configurationFactory =
          new ConfigurationFactory(
              ruleClassProvider.getConfigurationCollectionFactory(),
              ruleClassProvider.getConfigurationFragments());

      ProjectFile.Provider projectFileProvider = null;
      for (BlazeModule module : blazeModules) {
        ProjectFile.Provider candidate = module.createProjectFileProvider();
        if (candidate != null) {
          Preconditions.checkState(projectFileProvider == null,
              "more than one module defines a project file provider");
          projectFileProvider = candidate;
        }
      }

      return new BlazeRuntime(
          serverBuilder.getQueryEnvironmentFactory(),
          serverBuilder.getQueryFunctions(),
          serverBuilder.getQueryOutputFormatters(),
          packageFactory,
          ruleClassProvider,
          configurationFactory,
          serverBuilder.getInfoItems(),
          clock,
          startupOptionsProvider,
          ImmutableList.copyOf(blazeModules),
          eventBusExceptionHandler,
          projectFileProvider,
          serverBuilder.getInvocationPolicy(),
          serverBuilder.getCommands(),
          productName);
    }

    public Builder setProductName(String productName) {
      this.productName = productName;
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
  }
}
