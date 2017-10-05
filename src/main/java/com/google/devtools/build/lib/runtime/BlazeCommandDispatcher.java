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
import com.google.common.base.Joiner;
import com.google.common.base.Predicates;
import com.google.common.base.Throwables;
import com.google.common.base.Verify;
import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ListMultimap;
import com.google.common.io.Flushables;
import com.google.common.util.concurrent.UncheckedExecutionException;
import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.events.ExtendedEventHandler.Postable;
import com.google.devtools.build.lib.events.PrintingEventHandler;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.runtime.commands.ProjectFileSupport;
import com.google.devtools.build.lib.runtime.proto.InvocationPolicyOuterClass.InvocationPolicy;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.AnsiStrippingOutputStream;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.util.LoggingUtil;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.util.io.DelegatingOutErr;
import com.google.devtools.build.lib.util.io.OutErr;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.common.options.InvocationPolicyEnforcer;
import com.google.devtools.common.options.OpaqueOptionsData;
import com.google.devtools.common.options.OptionDefinition;
import com.google.devtools.common.options.OptionPriority;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;
import com.google.devtools.common.options.OptionsProvider;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Function;
import java.util.logging.Level;
import javax.annotation.Nullable;

/**
 * Dispatches to the Blaze commands; that is, given a command line, this
 * abstraction looks up the appropriate command object, parses the options
 * required by the object, and calls its exec method. Also, this object provides
 * the runtime state (BlazeRuntime) to the commands.
 */
public class BlazeCommandDispatcher {

  /**
   * What to do if the command lock is not available.
   */
  public enum LockingMode {
    WAIT,  // Wait until it is available
    ERROR_OUT,  // Return with an error
  }
  // Keep in sync with options added in OptionProcessor::AddRcfileArgsAndOptions()
  private static final ImmutableSet<String> INTERNAL_COMMAND_OPTIONS =
      ImmutableSet.of(
          "rc_source",
          "default_override",
          "isatty",
          "terminal_columns",
          "ignore_client_env",
          "client_env",
          "client_cwd");

  private static final ImmutableList<String> HELP_COMMAND = ImmutableList.of("help");

  private static final ImmutableSet<String> ALL_HELP_OPTIONS =
      ImmutableSet.of("--help", "-help", "-h");

  /**
   * By throwing this exception, a command indicates that it wants to shutdown
   * the Blaze server process.
   */
  public static class ShutdownBlazeServerException extends Exception {
    private final int exitStatus;

    public ShutdownBlazeServerException(int exitStatus, Throwable cause) {
      super(cause);
      this.exitStatus = exitStatus;
    }

    public ShutdownBlazeServerException(int exitStatus) {
      this.exitStatus = exitStatus;
    }

    public int getExitStatus() {
      return exitStatus;
    }
  }

  private final BlazeRuntime runtime;
  private final Object commandLock;
  private String currentClientDescription = null;
  private String shutdownReason = null;
  private OutputStream logOutputStream = null;
  private Level lastLogVerbosityLevel = null;
  private final LoadingCache<BlazeCommand, OpaqueOptionsData> optionsDataCache =
      CacheBuilder.newBuilder().build(
          new CacheLoader<BlazeCommand, OpaqueOptionsData>() {
            @Override
            public OpaqueOptionsData load(BlazeCommand command) {
              return OptionsParser.getOptionsData(BlazeCommandUtils.getOptions(
                  command.getClass(),
                  runtime.getBlazeModules(),
                  runtime.getRuleClassProvider()));
            }
          });

  /**
   * Create a Blaze dispatcher that uses the specified {@code BlazeRuntime} instance, but overrides
   * the command map with the given commands (plus any commands from modules).
   */
  @VisibleForTesting
  public BlazeCommandDispatcher(BlazeRuntime runtime, BlazeCommand... commands) {
    this(runtime);
    runtime.overrideCommands(Arrays.asList(commands));
  }

  /**
   * Create a Blaze dispatcher that uses the specified {@code BlazeRuntime} instance.
   */
  @VisibleForTesting
  public BlazeCommandDispatcher(BlazeRuntime runtime) {
    this.runtime = runtime;
    this.commandLock = new Object();
  }

  /**
   * Only some commands work if cwd != workspaceSuffix in Blaze. In that case, also check if Blaze
   * was called from the output directory and fail if it was.
   */
  private ExitCode checkCwdInWorkspace(BlazeWorkspace workspace, Command commandAnnotation,
      String commandName, EventHandler eventHandler) {
    if (!commandAnnotation.mustRunInWorkspace()) {
      return ExitCode.SUCCESS;
    }

    if (!workspace.getDirectories().inWorkspace()) {
      eventHandler.handle(
          Event.error(
              "The '" + commandName + "' command is only supported from within a workspace."));
      return ExitCode.COMMAND_LINE_ERROR;
    }

    Path workspacePath = workspace.getWorkspace();
    // TODO(kchodorow): Remove this once spaces are supported.
    if (workspacePath.getPathString().contains(" ")) {
      eventHandler.handle(
          Event.error(
              runtime.getProductName() + " does not currently work properly from paths "
                  + "containing spaces (" + workspace + ")."));
      return ExitCode.LOCAL_ENVIRONMENTAL_ERROR;
    }

    Path doNotBuild = workspacePath.getParentDirectory().getRelative(
        BlazeWorkspace.DO_NOT_BUILD_FILE_NAME);

    if (doNotBuild.exists()) {
      if (!commandAnnotation.canRunInOutputDirectory()) {
        eventHandler.handle(Event.error(getNotInRealWorkspaceError(doNotBuild)));
        return ExitCode.COMMAND_LINE_ERROR;
      } else {
        eventHandler.handle(
            Event.warn(
                runtime.getProductName() + " is run from output directory. This is unsound."));
      }
    }
    return ExitCode.SUCCESS;
  }

  private void parseArgsAndConfigs(Path workspaceDirectory, Path workingDirectory,
      OptionsParser optionsParser, Command commandAnnotation, List<String> args,
      List<String> rcfileNotes, ExtendedEventHandler eventHandler)
          throws OptionsParsingException {
    Function<OptionDefinition, String> commandOptionSourceFunction =
        option -> {
          if (INTERNAL_COMMAND_OPTIONS.contains(option.getOptionName())) {
            return "options generated by " + runtime.getProductName() + " launcher";
          } else {
            return "command line options";
          }
        };

    // Explicit command-line options:
    List<String> cmdLineAfterCommand = args.subList(1, args.size());
    optionsParser.parseWithSourceFunction(OptionPriority.COMMAND_LINE,
        commandOptionSourceFunction, cmdLineAfterCommand);

    // Command-specific options from .blazerc passed in via --default_override
    // and --rc_source. A no-op if none are provided.
    CommonCommandOptions rcFileOptions = optionsParser.getOptions(CommonCommandOptions.class);
    List<Pair<String, ListMultimap<String, String>>> optionsMap =
        getOptionsMap(eventHandler, rcFileOptions.rcSource, rcFileOptions.optionsOverrides,
            runtime.getCommandMap().keySet());

    parseOptionsForCommand(rcfileNotes, commandAnnotation, optionsParser, optionsMap, null, null);
    if (commandAnnotation.builds()) {
      ProjectFileSupport.handleProjectFiles(
          eventHandler, runtime.getProjectFileProvider(), workspaceDirectory, workingDirectory,
          optionsParser, commandAnnotation.name());
    }

    // Fix-point iteration until all configs are loaded.
    List<String> configsLoaded = ImmutableList.of();
    Set<String> unknownConfigs = new LinkedHashSet<>();
    CommonCommandOptions commonOptions = optionsParser.getOptions(CommonCommandOptions.class);
    while (!commonOptions.configs.equals(configsLoaded)) {
      Set<String> missingConfigs = new LinkedHashSet<>(commonOptions.configs);
      missingConfigs.removeAll(configsLoaded);
      parseOptionsForCommand(rcfileNotes, commandAnnotation, optionsParser, optionsMap,
          missingConfigs, unknownConfigs);
      configsLoaded = commonOptions.configs;
      commonOptions = optionsParser.getOptions(CommonCommandOptions.class);
    }
    if (!unknownConfigs.isEmpty()) {
      if (commonOptions.allowUndefinedConfigs) {
        eventHandler.handle(
            Event.warn(
                "Config values are not defined in any .rc file: "
                    + Joiner.on(", ").join(unknownConfigs)));
      } else {
        throw new OptionsParsingException(
            "Config values are not defined in any .rc file: "
                + Joiner.on(", ").join(unknownConfigs));
      }
    }
  }

  /**
   * Executes a single command. Returns the Unix exit status for the Blaze client process, or throws
   * {@link ShutdownBlazeServerException} to indicate that a command wants to shutdown the Blaze
   * server.
   */
  int exec(
      InvocationPolicy invocationPolicy,
      List<String> args,
      OutErr outErr,
      LockingMode lockingMode,
      String clientDescription,
      long firstContactTime,
      Optional<List<Pair<String, String>>> startupOptionsTaggedWithBazelRc)
      throws ShutdownBlazeServerException, InterruptedException {
    OriginalCommandLineEvent originalCommandLine = new OriginalCommandLineEvent(args);
    Preconditions.checkNotNull(clientDescription);
    if (args.isEmpty()) { // Default to help command if no arguments specified.
      args = HELP_COMMAND;
    }

    String commandName = args.get(0);

    // Be gentle to users who want to find out about Blaze invocation.
    if (ALL_HELP_OPTIONS.contains(commandName)) {
      commandName = "help";
    }

    BlazeCommand command = runtime.getCommandMap().get(commandName);
    if (command == null) {
      outErr.printErrLn(String.format(
          "Command '%s' not found. Try '%s help'.", commandName, runtime.getProductName()));
      return ExitCode.COMMAND_LINE_ERROR.getNumericExitCode();
    }

    // Take the exclusive server lock.  If we fail, we busy-wait until the lock becomes available.
    //
    // We used to rely on commandLock.wait() to lazy-wait for the lock to become available, which is
    // theoretically fine, but doing so prevents us from determining if the PID of the server
    // holding the lock has changed under the hood.  There have been multiple bug reports where
    // users (especially macOS ones) mention that the Blaze invocation hangs on a non-existent PID.
    // This should help troubleshoot those scenarios in case there really is a bug somewhere.
    int attempts = 0;
    long clockBefore = BlazeClock.nanoTime();
    String otherClientDescription = "";
    synchronized (commandLock) {
      while (currentClientDescription != null) {
        switch (lockingMode) {
          case WAIT:
            if (!otherClientDescription.equals(currentClientDescription)) {
              if (attempts > 0) {
                outErr.printErrLn(" lock taken by another command");
              }
              outErr.printErr("Another command (" + currentClientDescription + ") is running. "
                  + " Waiting for it to complete on the server...");
              otherClientDescription = currentClientDescription;
            }
            commandLock.wait(500);
            break;

          case ERROR_OUT:
            outErr.printErrLn(String.format("Another command (" + currentClientDescription + ") is "
                + "running. Exiting immediately."));
            return ExitCode.COMMAND_LINE_ERROR.getNumericExitCode();

          default:
            throw new IllegalStateException();
        }

        attempts += 1;
      }
      Verify.verify(currentClientDescription == null);
      currentClientDescription = clientDescription;
    }
    if (attempts > 0) {
      outErr.printErrLn(" done!");
    }
    // If we took the lock on the first try, force the reported wait time to 0 to avoid unnecessary
    // noise in the logs.  In this metric, we are only interested in knowing how long it took for
    // other commands to complete, not how fast acquiring a lock is.
    long waitTimeInMs = attempts == 0 ? 0 : (BlazeClock.nanoTime() - clockBefore) / (1000L * 1000L);

    try {
      if (shutdownReason != null) {
        outErr.printErrLn("Server shut down " + shutdownReason);
        return ExitCode.LOCAL_ENVIRONMENTAL_ERROR.getNumericExitCode();
      }
      return execExclusively(
          originalCommandLine,
          invocationPolicy,
          args,
          outErr,
          firstContactTime,
          commandName,
          command,
          waitTimeInMs,
          startupOptionsTaggedWithBazelRc);
    } catch (ShutdownBlazeServerException e) {
      shutdownReason = "explicitly by client " + currentClientDescription;
      throw e;
    } finally {
      synchronized (commandLock) {
        currentClientDescription = null;
        commandLock.notify();
      }
    }
  }

  private int execExclusively(
      OriginalCommandLineEvent originalCommandLine,
      InvocationPolicy invocationPolicy,
      List<String> args,
      OutErr outErr,
      long firstContactTime,
      String commandName,
      BlazeCommand command,
      long waitTimeInMs,
      Optional<List<Pair<String, String>>> startupOptionsTaggedWithBazelRc)
      throws ShutdownBlazeServerException {
    // Record the start time for the profiler. Do not put anything before this!
    long execStartTimeNanos = runtime.getClock().nanoTime();

    Command commandAnnotation = command.getClass().getAnnotation(Command.class);
    BlazeWorkspace workspace = runtime.getWorkspace();

    StoredEventHandler eventHandler = new StoredEventHandler();
    AtomicReference<OptionsProvider> optionsResult = new AtomicReference<>();
    // Delay output of notes regarding the parsed rc file, so it's possible to disable this in the
    // rc file.
    List<String> rcfileNotes = new ArrayList<>();
    ExitCode earlyExitCode = parseOptions(
        eventHandler, workspace, command, commandAnnotation, commandName, invocationPolicy, args,
        optionsResult, rcfileNotes);
    OptionsProvider options = optionsResult.get();

    // The initCommand call also records the start time for the timestamp granularity monitor.
    CommandEnvironment env = workspace.initCommand(commandAnnotation, options);
    // Record the command's starting time for use by the commands themselves.
    env.recordCommandStartTime(firstContactTime);

    // Temporary: there is one module that outputs events during beforeCommand, but the reporter
    // isn't setup yet. Add the stored event handler to catch those events.
    env.getReporter().addHandler(eventHandler);
    for (BlazeModule module : runtime.getBlazeModules()) {
      try {
        module.beforeCommand(env);
      } catch (AbruptExitException e) {
        // Don't let one module's complaints prevent the other modules from doing necessary
        // setup. We promised to call beforeCommand exactly once per-module before each command
        // and will be calling afterCommand soon in the future - a module's afterCommand might
        // rightfully assume its beforeCommand has already been called.
        eventHandler.handle(Event.error(e.getMessage()));
        // It's not ideal but we can only return one exit code, so we just pick the code of the
        // last exception.
        earlyExitCode = e.getExitCode();
      }
    }
    env.getReporter().removeHandler(eventHandler);

    // We may only start writing to outErr once we've given the modules the chance to hook into it.
    for (BlazeModule module : runtime.getBlazeModules()) {
      OutErr listener = module.getOutputListener();
      if (listener != null) {
        outErr = tee(outErr, listener);
      }
    }

    // Early exit. We need to guarantee that the ErrOut and Reporter setup below never error out, so
    // any invariants they need must be checked before this point.
    if (!earlyExitCode.equals(ExitCode.SUCCESS)) {
      // Partial replay of the printed events before we exit.
      PrintingEventHandler printingEventHandler =
          new PrintingEventHandler(outErr, EventKind.ALL_EVENTS);
      for (String note : rcfileNotes) {
        printingEventHandler.handle(Event.info(note));
      }
      for (Event event : eventHandler.getEvents()) {
        printingEventHandler.handle(event);
      }
      for (Postable post : eventHandler.getPosts()) {
        env.getEventBus().post(post);
      }
      // TODO(ulfjack): We're not calling BlazeModule.afterCommand here, even though we should.
      return earlyExitCode.getNumericExitCode();
    }

    // Setup log filtering
    BlazeCommandEventHandler.Options eventHandlerOptions =
        options.getOptions(BlazeCommandEventHandler.Options.class);
    OutErr colorfulOutErr = outErr;

    if (!eventHandlerOptions.useColor()) {
      outErr = ansiStripOut(ansiStripErr(outErr));
      if (!commandAnnotation.binaryStdOut()) {
        colorfulOutErr = ansiStripOut(colorfulOutErr);
      }
      if (!commandAnnotation.binaryStdErr()) {
        colorfulOutErr = ansiStripErr(colorfulOutErr);
      }
    }

    if (!commandAnnotation.binaryStdOut()) {
      outErr = bufferOut(outErr, eventHandlerOptions.experimentalUi);
    }

    if (!commandAnnotation.binaryStdErr()) {
      outErr = bufferErr(outErr, eventHandlerOptions.experimentalUi);
    }

    CommonCommandOptions commonOptions = options.getOptions(CommonCommandOptions.class);
    if (!commonOptions.verbosity.equals(lastLogVerbosityLevel)) {
      BlazeRuntime.setupLogging(commonOptions.verbosity);
      lastLogVerbosityLevel = commonOptions.verbosity;
    }

    // Do this before an actual crash so we don't have to worry about
    // allocating memory post-crash.
    String[] crashData = env.getCrashData();
    int numericExitCode = ExitCode.BLAZE_INTERNAL_ERROR.getNumericExitCode();
    PrintStream savedOut = System.out;
    PrintStream savedErr = System.err;

    EventHandler handler = createEventHandler(outErr, eventHandlerOptions);
    Reporter reporter = env.getReporter();
    reporter.addHandler(handler);
    env.getEventBus().register(handler);

    int oomMoreEagerlyThreshold = commonOptions.oomMoreEagerlyThreshold;
    if (oomMoreEagerlyThreshold == 100) {
      oomMoreEagerlyThreshold =
          runtime
              .getStartupOptionsProvider()
              .getOptions(BlazeServerStartupOptions.class)
              .oomMoreEagerlyThreshold;
    }
    if (oomMoreEagerlyThreshold < 0 || oomMoreEagerlyThreshold > 100) {
      reporter.handle(Event.error("--oom_more_eagerly_threshold must be non-negative percent"));
      return ExitCode.COMMAND_LINE_ERROR.getNumericExitCode();
    }
    if (oomMoreEagerlyThreshold != 100) {
      try {
        RetainedHeapLimiter.maybeInstallRetainedHeapLimiter(oomMoreEagerlyThreshold);
      } catch (OptionsParsingException e) {
        reporter.handle(Event.error(e.getMessage()));
        return ExitCode.COMMAND_LINE_ERROR.getNumericExitCode();
      }
    }

    // We register an ANSI-allowing handler associated with {@code handler} so that ANSI control
    // codes can be re-introduced later even if blaze is invoked with --color=no. This is useful
    // for commands such as 'blaze run' where the output of the final executable shouldn't be
    // modified.
    EventHandler ansiAllowingHandler = null;
    if (!eventHandlerOptions.useColor()) {
      ansiAllowingHandler = createEventHandler(colorfulOutErr, eventHandlerOptions);
      reporter.registerAnsiAllowingHandler(handler, ansiAllowingHandler);
      if (ansiAllowingHandler instanceof ExperimentalEventHandler) {
        env.getEventBus()
            .register(
                new PassiveExperimentalEventHandler(
                    (ExperimentalEventHandler) ansiAllowingHandler));
      }
    }

    // Now we're ready to replay the events.
    eventHandler.replayOn(reporter);

    try {
      // While a Blaze command is active, direct all errors to the client's
      // event handler (and out/err streams).
      OutErr reporterOutErr = reporter.getOutErr();
      System.setOut(new PrintStream(reporterOutErr.getOutputStream(), /*autoflush=*/true));
      System.setErr(new PrintStream(reporterOutErr.getErrorStream(), /*autoflush=*/true));

      for (BlazeModule module : runtime.getBlazeModules()) {
        module.checkEnvironment(env);
      }

      if (commonOptions.announceRcOptions) {
        if (startupOptionsTaggedWithBazelRc.isPresent()) {
          String lastBlazerc = "";
          List<String> accumulatedStartupOptions = new ArrayList<>();
          for (Pair<String, String> option : startupOptionsTaggedWithBazelRc.get()) {
            // Do not include the command line options, marked by the empty string.
            if (option.getFirst().isEmpty()) {
              continue;
            }

            // If we've moved to a new blazerc in the list, print out the info from the last one,
            // and clear the accumulated list.
            if (!lastBlazerc.isEmpty() && !option.getFirst().equals(lastBlazerc)) {
              String logMessage =
                  String.format(
                      "Reading 'startup' options from %s: %s",
                      lastBlazerc, String.join(", ", accumulatedStartupOptions));
              reporter.handle(Event.info(logMessage));
              accumulatedStartupOptions = new ArrayList<>();
            }

            lastBlazerc = option.getFirst();
            accumulatedStartupOptions.add(option.getSecond());
          }
          // Print out the final blazerc-grouped list, if any startup options were provided by
          // blazerc.
          if (!lastBlazerc.isEmpty()) {
            String logMessage =
                String.format(
                    "Reading 'startup' options from %s: %s",
                    lastBlazerc, String.join(", ", accumulatedStartupOptions));
            reporter.handle(Event.info(logMessage));
          }
        }
        for (String note : rcfileNotes) {
          reporter.handle(Event.info(note));
        }
      }

      try {
        // Notify the BlazeRuntime, so it can do some initial setup.
        env.beforeCommand(
            options,
            commonOptions,
            execStartTimeNanos,
            waitTimeInMs,
            invocationPolicy);
      } catch (AbruptExitException e) {
        reporter.handle(Event.error(e.getMessage()));
        return e.getExitCode().getNumericExitCode();
      }

      env.getEventBus().post(originalCommandLine);

      for (BlazeModule module : runtime.getBlazeModules()) {
        env.getSkyframeExecutor().injectExtraPrecomputedValues(module.getPrecomputedValues());
      }

      ExitCode outcome = command.exec(env, options);
      outcome = env.precompleteCommand(outcome);
      numericExitCode = outcome.getNumericExitCode();
      return numericExitCode;
    } catch (ShutdownBlazeServerException e) {
      numericExitCode = e.getExitStatus();
      throw e;
    } catch (Throwable e) {
      e.printStackTrace();
      BugReport.printBug(outErr, e);
      BugReport.sendBugReport(e, args, crashData);
      numericExitCode = BugReport.getExitCodeForThrowable(e);
      throw new ShutdownBlazeServerException(numericExitCode, e);
    } finally {
      env.getEventBus().post(new AfterCommandEvent());
      runtime.afterCommand(env, numericExitCode);
      // Swallow IOException, as we are already in a finally clause
      Flushables.flushQuietly(outErr.getOutputStream());
      Flushables.flushQuietly(outErr.getErrorStream());

      System.setOut(savedOut);
      System.setErr(savedErr);
      reporter.removeHandler(handler);
      releaseHandler(handler);
      if (!eventHandlerOptions.useColor()) {
        reporter.removeHandler(ansiAllowingHandler);
        releaseHandler(ansiAllowingHandler);
      }
      env.getTimestampGranularityMonitor().waitForTimestampGranularity(outErr);
    }
  }

  /**
   * For testing ONLY. Same as {@link #exec(InvocationPolicy, List, OutErr, LockingMode, String,
   * long, Optional<List<Pair<String, String>>>)}, but automatically uses the current time.
   */
  @VisibleForTesting
  public int exec(
      List<String> args, LockingMode lockingMode, String clientDescription, OutErr originalOutErr)
      throws ShutdownBlazeServerException, InterruptedException {
    return exec(
        InvocationPolicy.getDefaultInstance(),
        args,
        originalOutErr,
        LockingMode.ERROR_OUT,
        clientDescription,
        runtime.getClock().currentTimeMillis(),
        Optional.empty() /* startupOptionBundles */);
  }

  /**
   * Parses the options, taking care not to generate any output to outErr, return, or throw an
   * exception.
   *
   * @return ExitCode.SUCCESS if everything went well, or some other value if not
   */
  private ExitCode parseOptions(
      ExtendedEventHandler eventHandler,
      BlazeWorkspace workspace,
      BlazeCommand command,
      Command commandAnnotation,
      String commandName,
      InvocationPolicy invocationPolicy,
      List<String> args,
      // Declare options as OptionsProvider so the options can't be easily modified after we've
      // applied the invocation policy.
      AtomicReference<OptionsProvider> parsedOptions,
      List<String> rcfileNotes) {
    OptionsParser optionsParser;
    try {
      optionsParser = createOptionsParser(command);
      // We need to set this early so it's not null when we return.
      parsedOptions.set(optionsParser);
    } catch (OptionsParser.ConstructionException e) {
      // This should never happen.
      throw new IllegalStateException(e);
    }

    // The initialization code here was carefully written to parse the options early before we call
    // into the BlazeModule APIs, which means we must not generate any output to outErr, return, or
    // throw an exception. All the events happening here are instead stored in a temporary event
    // handler, and later replayed.
    ExitCode earlyExitCode =
        checkCwdInWorkspace(workspace, commandAnnotation, commandName, eventHandler);
    if (!earlyExitCode.equals(ExitCode.SUCCESS)) {
      return earlyExitCode;
    }

    try {
      // TODO(ulfjack): The second parameter is supposed to be the working directory, except that
      // the client passes that as part of CommonCommandOptions, and we can't know those until
      // after we've parsed them.
      parseArgsAndConfigs(
          workspace.getWorkspace(), /*workingDirectory=*/workspace.getWorkspace(), optionsParser,
          commandAnnotation, args, rcfileNotes, eventHandler);
      // Allow the command to edit the options.
      command.editOptions(optionsParser);
      // Migration of --watchfs to a command option.
      // TODO(ulfjack): Get rid of the startup option and drop this code.
      if (runtime.getStartupOptionsProvider().getOptions(BlazeServerStartupOptions.class).watchFS) {
        try {
          optionsParser.parse("--watchfs");
        } catch (OptionsParsingException e) {
          // This should never happen.
          throw new IllegalStateException(e);
        }
      }
      // Merge the invocation policy that is user-supplied, from the command line, and any
      // invocation policy that was added by a module. The module one goes 'first,' so the user
      // one has priority.
      InvocationPolicy combinedPolicy =
          InvocationPolicy.newBuilder()
              .mergeFrom(runtime.getModuleInvocationPolicy())
              .mergeFrom(invocationPolicy)
              .build();
      InvocationPolicyEnforcer optionsPolicyEnforcer =
          new InvocationPolicyEnforcer(combinedPolicy, Level.INFO);
      // Enforce the invocation policy. It is intentional that this is the last step in preparing
      // the options. The invocation policy is used in security-critical contexts, and may be used
      // as a last resort to override flags. That means that the policy can override flags set in
      // BlazeCommand.editOptions, so the code needs to be safe regardless of the actual flag
      // values. At the time of this writing, editOptions was only used as a convenience feature or
      // to improve the user experience, but not required for safety or correctness.
      optionsPolicyEnforcer.enforce(optionsParser, commandName);
      // Print warnings for odd options usage
      for (String warning : optionsParser.getWarnings()) {
        eventHandler.handle(Event.warn(warning));
      }
    } catch (OptionsParsingException e) {
      eventHandler.handle(Event.error(e.getMessage()));
      return ExitCode.COMMAND_LINE_ERROR;
    }
    return ExitCode.SUCCESS;
  }

  /**
   * Parses the options from .rc files for a command invocation. It works in one of two modes;
   * either it loads the non-config options, or the config options that are specified in the {@code
   * configs} parameter.
   *
   * <p>This method adds every option pertaining to the specified command to the options parser. To
   * do that, it needs the command -> option mapping that is generated from the .rc files.
   *
   * <p>It is not as trivial as simply taking the list of options for the specified command because
   * commands can inherit arguments from each other, and we have to respect that (e.g. if an option
   * is specified for 'build', it needs to take effect for the 'test' command, too).
   *
   * <p>Note that the order in which the options are parsed is well-defined: all options from the
   * same rc file are parsed at the same time, and the rc files are handled in the order in which
   * they were passed in from the client.
   *
   * @param rcfileNotes note message that would be printed during parsing
   * @param commandAnnotation the command for which options should be parsed.
   * @param optionsParser parser to receive parsed options.
   * @param optionsMap .rc files in structured format: a list of pairs, where the first part is the
   *     name of the rc file, and the second part is a multimap of command name (plus config, if
   *     present) to the list of options for that command
   * @param configs the configs for which to parse options; if {@code null}, non-config options are
   *     parsed
   * @param unknownConfigs optional; a collection that the method will populate with the config
   *     values in {@code configs} that none of the .rc files had entries for
   * @throws OptionsParsingException
   */
  protected static void parseOptionsForCommand(List<String> rcfileNotes, Command commandAnnotation,
      OptionsParser optionsParser, List<Pair<String, ListMultimap<String, String>>> optionsMap,
      @Nullable Collection<String> configs, @Nullable Collection<String> unknownConfigs)
      throws OptionsParsingException {
    Set<String> knownConfigs = new HashSet<>();
    for (String commandToParse : getCommandNamesToParse(commandAnnotation)) {
      for (Pair<String, ListMultimap<String, String>> entry : optionsMap) {
        List<String> allOptions = new ArrayList<>();
        if (configs == null) {
          allOptions.addAll(entry.second.get(commandToParse));
        } else {
          for (String config : configs) {
            Collection<String> values = entry.second.get(commandToParse + ":" + config);
            if (!values.isEmpty()) {
              allOptions.addAll(values);
              knownConfigs.add(config);
            }
          }
        }
        processOptionList(optionsParser, commandToParse,
            commandAnnotation.name(), rcfileNotes, entry.first, allOptions);
      }
    }
    if (unknownConfigs != null && configs != null && configs.size() > knownConfigs.size()) {
      configs
          .stream()
          .filter(Predicates.not(Predicates.in(knownConfigs)))
          .forEachOrdered(unknownConfigs::add);
    }
  }

  // Processes the option list for an .rc file - command pair.
  private static void processOptionList(OptionsParser optionsParser, String commandToParse,
      String originalCommand, List<String> rcfileNotes, String rcfile, List<String> rcfileOptions)
      throws OptionsParsingException {
    if (!rcfileOptions.isEmpty()) {
      String inherited = commandToParse.equals(originalCommand) ? "" : "Inherited ";
      String source = rcfile.equals("client") ? "Options provided by the client"
          : "Reading options for '" + originalCommand + "' from " + rcfile;
      rcfileNotes.add(source + ":\n"
          + "  " + inherited + "'" + commandToParse + "' options: "
          + Joiner.on(' ').join(rcfileOptions));
      optionsParser.parse(OptionPriority.RC_FILE, rcfile, rcfileOptions);
    }
  }

  private static List<String> getCommandNamesToParse(Command commandAnnotation) {
    List<String> result = new ArrayList<>();
    result.add("common");
    getCommandNamesToParseHelper(commandAnnotation, result);
    return result;
  }

  private static void getCommandNamesToParseHelper(Command commandAnnotation,
      List<String> accumulator) {
    for (Class<? extends BlazeCommand> base : commandAnnotation.inherits()) {
      getCommandNamesToParseHelper(base.getAnnotation(Command.class), accumulator);
    }
    accumulator.add(commandAnnotation.name());
  }

  private OutErr bufferOut(OutErr outErr, boolean fully) {
    OutputStream wrappedOut;
    if (fully) {
      wrappedOut = new BufferedOutputStream(outErr.getOutputStream());
    } else {
      wrappedOut = new LineBufferedOutputStream(outErr.getOutputStream());
    }
    return OutErr.create(wrappedOut, outErr.getErrorStream());
  }

  private OutErr bufferErr(OutErr outErr, boolean fully) {
    OutputStream wrappedErr;
    if (fully) {
      wrappedErr = new BufferedOutputStream(outErr.getErrorStream());
    } else {
      wrappedErr = new LineBufferedOutputStream(outErr.getErrorStream());
    }
    return OutErr.create(outErr.getOutputStream(), wrappedErr);
  }

  private OutErr ansiStripOut(OutErr outErr) {
    OutputStream wrappedOut = new AnsiStrippingOutputStream(outErr.getOutputStream());
    return OutErr.create(wrappedOut, outErr.getErrorStream());
  }

  private OutErr ansiStripErr(OutErr outErr) {
    OutputStream wrappedErr = new AnsiStrippingOutputStream(outErr.getErrorStream());
    return OutErr.create(outErr.getOutputStream(), wrappedErr);
  }

  private String getNotInRealWorkspaceError(Path doNotBuildFile) {
    String message =
        String.format(
            "%1$s should not be called from a %1$s output directory. ", runtime.getProductName());
    try {
      String realWorkspace =
          new String(FileSystemUtils.readContentAsLatin1(doNotBuildFile));
      message += String.format("The pertinent workspace directory is: '%s'",
          realWorkspace);
    } catch (IOException e) {
      // We are exiting anyway.
    }

    return message;
  }

  private OutErr tee(OutErr outErr1, OutErr outErr2) {
    DelegatingOutErr outErr = new DelegatingOutErr();
    outErr.addSink(outErr1);
    outErr.addSink(outErr2);
    return outErr;
  }

  private void closeSilently(OutputStream logOutputStream) {
    if (logOutputStream != null) {
      try {
        logOutputStream.close();
      } catch (IOException e) {
        LoggingUtil.logToRemote(Level.WARNING, "Unable to close command.log", e);
      }
    }
  }

  /**
   * Creates an option parser using the common options classes and the command-specific options
   * classes.
   *
   * <p>An overriding method should first call this method and can then override default values
   * directly or by calling {@link #parseOptionsForCommand} for command-specific options.
   */
  protected OptionsParser createOptionsParser(BlazeCommand command)
      throws OptionsParser.ConstructionException {
    OpaqueOptionsData optionsData = null;
    try {
      optionsData = optionsDataCache.getUnchecked(command);
    } catch (UncheckedExecutionException e) {
      Throwables.throwIfInstanceOf(e.getCause(), OptionsParser.ConstructionException.class);
      throw new IllegalStateException(e);
    }
    Command annotation = command.getClass().getAnnotation(Command.class);
    OptionsParser parser = OptionsParser.newOptionsParser(optionsData);
    parser.setAllowResidue(annotation.allowResidue());
    return parser;
  }

  /**
   * Convert a list of option override specifications to a more easily digestible
   * form.
   *
   * @param overrides list of option override specifications
   */
  @VisibleForTesting
  static List<Pair<String, ListMultimap<String, String>>> getOptionsMap(
      EventHandler eventHandler,
      List<String> rcFiles,
      List<CommonCommandOptions.OptionOverride> overrides,
      Set<String> validCommands) {
    List<Pair<String, ListMultimap<String, String>>> result = new ArrayList<>();

    String lastRcFile = null;
    ListMultimap<String, String> lastMap = null;
    for (CommonCommandOptions.OptionOverride override : overrides) {
      if (override.blazeRc < 0 || override.blazeRc >= rcFiles.size()) {
        eventHandler.handle(
            Event.warn("inconsistency in generated command line args. Ignoring bogus argument\n"));
        continue;
      }
      String rcFile = rcFiles.get(override.blazeRc);

      String command = override.command;
      int index = command.indexOf(':');
      if (index > 0) {
        command = command.substring(0, index);
      }
      if (!validCommands.contains(command) && !command.equals("common")) {
        eventHandler.handle(
            Event.warn(
                "while reading option defaults file '" + rcFile + "':\n"
                    + "  invalid command name '" + override.command + "'."));
        continue;
      }

      if (!rcFile.equals(lastRcFile)) {
        if (lastRcFile != null) {
          result.add(Pair.of(lastRcFile, lastMap));
        }
        lastRcFile = rcFile;
        lastMap = ArrayListMultimap.create();
      }
      lastMap.put(override.command, override.option);
    }
    if (lastRcFile != null) {
      result.add(Pair.of(lastRcFile, lastMap));
    }

    return result;
  }

  /**
   * Returns the event handler to use for this Blaze command.
   */
  private EventHandler createEventHandler(OutErr outErr,
      BlazeCommandEventHandler.Options eventOptions) {
    EventHandler eventHandler;
    if (eventOptions.experimentalUi) {
      // The experimental event handler is not to be rate limited.
      return new ExperimentalEventHandler(outErr, eventOptions, runtime.getClock());
    } else if ((eventOptions.useColor() || eventOptions.useCursorControl())) {
      eventHandler = new FancyTerminalEventHandler(outErr, eventOptions);
    } else {
      eventHandler = new BlazeCommandEventHandler(outErr, eventOptions);
    }

    return RateLimitingEventHandler.create(eventHandler, eventOptions.showProgressRateLimit);
  }

  /** Unsets the event handler. */
  private void releaseHandler(EventHandler eventHandler) {
    if (eventHandler instanceof FancyTerminalEventHandler) {
      // Make sure that the terminal state of the old event handler is clear
      // before creating a new one.
      ((FancyTerminalEventHandler) eventHandler).resetTerminal();
    }
  }

  /**
   * Returns the runtime instance shared by the commands that this dispatcher
   * dispatches to.
   */
  public BlazeRuntime getRuntime() {
    return runtime;
  }

  /**
   * Shuts down all the registered commands to give them a chance to cleanup or
   * close resources. Should be called by the owner of this command dispatcher
   * in all termination cases.
   */
  public void shutdown() {
    closeSilently(logOutputStream);
    logOutputStream = null;
  }
}
