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

import static com.google.devtools.build.lib.runtime.BlazeOptionHandler.BAD_OPTION_TAG;
import static com.google.devtools.build.lib.runtime.BlazeOptionHandler.ERROR_SEPARATOR;
import static com.google.devtools.common.options.Converters.BLAZE_ALIASING_FLAG;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.base.Strings;
import com.google.common.base.Throwables;
import com.google.common.base.Verify;
import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.flogger.GoogleLogger;
import com.google.common.io.Flushables;
import com.google.common.util.concurrent.UncheckedExecutionException;
import com.google.devtools.build.lib.analysis.NoBuildEvent;
import com.google.devtools.build.lib.bugreport.BugReport;
import com.google.devtools.build.lib.bugreport.BugReporter;
import com.google.devtools.build.lib.buildeventstream.BuildEventProtocolOptions;
import com.google.devtools.build.lib.buildtool.buildevent.ProfilerStartedEvent;
import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.events.ExtendedEventHandler.Postable;
import com.google.devtools.build.lib.events.PrintingEventHandler;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.runtime.proto.InvocationPolicyOuterClass.InvocationPolicy;
import com.google.devtools.build.lib.server.FailureDetails;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.Interrupted.Code;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.AnsiStrippingOutputStream;
import com.google.devtools.build.lib.util.DebugLoggerConfigurator;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.util.InterruptedFailureDetails;
import com.google.devtools.build.lib.util.LoggingUtil;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.io.DelegatingOutErr;
import com.google.devtools.build.lib.util.io.OutErr;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.common.options.OpaqueOptionsData;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingResult;
import java.io.BufferedOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.logging.Level;
import net.starlark.java.eval.Starlark;

/**
 * Dispatches to the Blaze commands; that is, given a command line, this abstraction looks up the
 * appropriate command object, parses the options required by the object, and calls its exec method.
 * Also, this object provides the runtime state (BlazeRuntime) to the commands.
 */
public class BlazeCommandDispatcher implements CommandDispatcher {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  public static final int UNKNOWN_SERVER_PID = -1;

  private static final ImmutableList<String> HELP_COMMAND = ImmutableList.of("help");

  private static final ImmutableSet<String> ALL_HELP_OPTIONS =
      ImmutableSet.of("--help", "-help", "-h");

  private final BlazeRuntime runtime;
  private final int serverPid;
  private final BugReporter bugReporter;
  private final Object commandLock;
  private String currentClientDescription = null;
  private String shutdownReason = null;
  private OutputStream logOutputStream = null;
  private final LoadingCache<BlazeCommand, OpaqueOptionsData> optionsDataCache =
      CacheBuilder.newBuilder()
          .build(
              new CacheLoader<BlazeCommand, OpaqueOptionsData>() {
                @Override
                public OpaqueOptionsData load(BlazeCommand command) {
                  return OptionsParser.getOptionsData(
                      BlazeCommandUtils.getOptions(
                          command.getClass(),
                          runtime.getBlazeModules(),
                          runtime.getRuleClassProvider()));
                }
              });

  BlazeCommandDispatcher(BlazeRuntime runtime, int serverPid) {
    this(runtime, serverPid, runtime.getBugReporter());
  }

  /** Convenience test-only constructor. */
  @VisibleForTesting
  public BlazeCommandDispatcher(BlazeRuntime runtime) {
    this(runtime, UNKNOWN_SERVER_PID, runtime.getBugReporter());
  }

  /** Convenience test-only constructor. */
  @VisibleForTesting
  BlazeCommandDispatcher(BlazeRuntime runtime, BugReporter bugReporter) {
    this(runtime, UNKNOWN_SERVER_PID, bugReporter);
  }

  private BlazeCommandDispatcher(BlazeRuntime runtime, int serverPid, BugReporter bugReporter) {
    this.runtime = runtime;
    this.serverPid = serverPid;
    this.bugReporter = bugReporter;
    this.commandLock = new Object();
  }

  @Override
  public BlazeCommandResult exec(
      InvocationPolicy invocationPolicy,
      List<String> args,
      OutErr outErr,
      LockingMode lockingMode,
      String clientDescription,
      long firstContactTimeMillis,
      Optional<List<Pair<String, String>>> startupOptionsTaggedWithBazelRc)
      throws InterruptedException {
    OriginalUnstructuredCommandLineEvent originalCommandLine =
        new OriginalUnstructuredCommandLineEvent(args);
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
      outErr.printErrLn(
          String.format(
              "Command '%s' not found. Try '%s help'.", commandName, runtime.getProductName()));
      return createDetailedCommandResult(
          String.format("Command '%s' not found.", commandName),
          ExitCode.COMMAND_LINE_ERROR,
          FailureDetails.Command.Code.COMMAND_NOT_FOUND);
    }

    // Take the exclusive server lock.  If we fail, we busy-wait until the lock becomes available.
    //
    // We used to rely on commandLock.wait() to lazy-wait for the lock to become available, which is
    // theoretically fine, but doing so prevents us from determining if the PID of the server
    // holding the lock has changed under the hood.  There have been multiple bug reports where
    // users (especially macOS ones) mention that the Blaze invocation hangs on a non-existent PID.
    // This should help troubleshoot those scenarios in case there really is a bug somewhere.
    boolean multipleAttempts = false;
    long clockBefore = BlazeClock.nanoTime();
    String otherClientDescription = "";
    // TODO(ulfjack): Add lock acquisition to the profiler.
    synchronized (commandLock) {
      while (currentClientDescription != null) {
        switch (lockingMode) {
          case WAIT:
            if (!otherClientDescription.equals(currentClientDescription)) {
              String serverDescription =
                  serverPid == UNKNOWN_SERVER_PID ? "" : (" (server_pid=" + serverPid + ")");
              outErr.printErrLn(
                  String.format(
                      "Another command (%s) is running. Waiting for it to complete on the"
                          + " server%s...",
                      currentClientDescription, serverDescription));
              otherClientDescription = currentClientDescription;
            }
            commandLock.wait(500);
            break;

          case ERROR_OUT:
            String message =
                String.format(
                    "Another command (%s) is running. Exiting immediately.",
                    currentClientDescription);
            outErr.printErrLn(message);
            return createDetailedCommandResult(
                message,
                ExitCode.LOCK_HELD_NOBLOCK_FOR_LOCK,
                FailureDetails.Command.Code.ANOTHER_COMMAND_RUNNING);

          default:
            throw new IllegalStateException();
        }

        multipleAttempts = true;
      }
      Verify.verify(currentClientDescription == null);
      currentClientDescription = clientDescription;
    }
    // If we took the lock on the first try, force the reported wait time to 0 to avoid unnecessary
    // noise in the logs.  In this metric, we are only interested in knowing how long it took for
    // other commands to complete, not how fast acquiring a lock is.
    long waitTimeInMs =
        !multipleAttempts ? 0 : (BlazeClock.nanoTime() - clockBefore) / (1000L * 1000L);

    try {
      if (shutdownReason != null) {
        String message = "Server shut down " + shutdownReason;
        outErr.printErrLn(message);
        return createDetailedCommandResult(
            message,
            ExitCode.LOCAL_ENVIRONMENTAL_ERROR,
            FailureDetails.Command.Code.PREVIOUSLY_SHUTDOWN);
      }
      BlazeCommandResult result =
          execExclusively(
              originalCommandLine,
              invocationPolicy,
              args,
              outErr,
              firstContactTimeMillis,
              commandName,
              command,
              waitTimeInMs,
              startupOptionsTaggedWithBazelRc);
      if (result.shutdown()) {
        // TODO(lberki): This also handles the case where we catch an uncaught Throwable in
        // execExclusively() which is not an explicit shutdown.
        shutdownReason = "explicitly by client " + clientDescription;
      }
      if (!result.getDetailedExitCode().isSuccess()) {
        logger.atInfo().log("Exit status was %s", result.getDetailedExitCode());
      }
      return result;
    } finally {
      synchronized (commandLock) {
        currentClientDescription = null;
        commandLock.notify();
      }
    }
  }

  /**
   * For testing ONLY. Same as {@link #exec(InvocationPolicy, List, OutErr, LockingMode, String,
   * long, Optional<List<Pair<String, String>>>)}, but automatically uses the current time.
   */
  @VisibleForTesting
  public BlazeCommandResult exec(List<String> args, String clientDescription, OutErr originalOutErr)
      throws InterruptedException {
    return exec(
        InvocationPolicy.getDefaultInstance(),
        args,
        originalOutErr,
        LockingMode.ERROR_OUT,
        clientDescription,
        runtime.getClock().currentTimeMillis(),
        Optional.empty() /* startupOptionBundles */);
  }

  private BlazeCommandResult execExclusively(
      OriginalUnstructuredCommandLineEvent unstructuredServerCommandLineEvent,
      InvocationPolicy invocationPolicy,
      List<String> args,
      OutErr outErr,
      long firstContactTime,
      String commandName,
      BlazeCommand command,
      long waitTimeInMs,
      Optional<List<Pair<String, String>>> startupOptionsTaggedWithBazelRc) {
    // Record the start time for the profiler. Do not put anything before this!
    long execStartTimeNanos = runtime.getClock().nanoTime();

    Command commandAnnotation = command.getClass().getAnnotation(Command.class);
    BlazeWorkspace workspace = runtime.getWorkspace();

    StoredEventHandler storedEventHandler = new StoredEventHandler();
    BlazeOptionHandler optionHandler =
        new BlazeOptionHandler(
            runtime,
            workspace,
            command,
            commandAnnotation,
            // Provide the options parser so that we can cache OptionsData here.
            createOptionsParser(command),
            invocationPolicy);
    DetailedExitCode earlyExitCode = optionHandler.parseOptions(args, storedEventHandler);
    OptionsParsingResult options = optionHandler.getOptionsResult();

    CommandLineEvent originalCommandLineEvent =
        new CommandLineEvent.OriginalCommandLineEvent(
            runtime, commandName, options, startupOptionsTaggedWithBazelRc);
    CommandLineEvent canonicalCommandLineEvent =
        new CommandLineEvent.CanonicalCommandLineEvent(runtime, commandName, options);

    // The initCommand call also records the start time for the timestamp granularity monitor.
    List<String> commandEnvWarnings = new ArrayList<>();
    CommandEnvironment env =
        workspace.initCommand(
            commandAnnotation, options, commandEnvWarnings, waitTimeInMs, firstContactTime);
    CommonCommandOptions commonOptions = options.getOptions(CommonCommandOptions.class);
    // We cannot flip an incompatible flag that expands to other flags, so we do it manually here.
    // If an option is specified explicitly, we give that preference.
    boolean commandSupportsProfile =
        (commandAnnotation.builds() || "query".equals(commandName))
            && !"clean".equals(commandName)
            && !"info".equals(commandName);
    boolean profileExplicitlyDisabled =
        options.containsExplicitOption("experimental_generate_json_trace_profile")
            && !commonOptions.enableTracer;
    if (commandSupportsProfile && !profileExplicitlyDisabled) {
      commonOptions.enableTracer = true;
      if (!options.containsExplicitOption("experimental_profile_cpu_usage")) {
        commonOptions.enableCpuUsageProfiling = true;
      }
    }
    // TODO(ulfjack): Move the profiler initialization as early in the startup sequence as possible.
    // Profiler setup and shutdown must always happen in pairs. Shutdown is currently performed in
    // the afterCommand call in the finally block below.
    ProfilerStartedEvent profilerStartedEvent =
        runtime.initProfiler(
            storedEventHandler,
            workspace,
            commonOptions,
            options.getOptions(BuildEventProtocolOptions.class),
            env,
            execStartTimeNanos,
            waitTimeInMs);
    storedEventHandler.post(profilerStartedEvent);

    // Enable Starlark CPU profiling (--starlark_cpu_profile=/tmp/foo.pprof.gz)
    if (!commonOptions.starlarkCpuProfile.isEmpty()) {
      FileOutputStream out;
      try {
        out = new FileOutputStream(commonOptions.starlarkCpuProfile);
      } catch (IOException ex) {
        String message = "Starlark CPU profiler: " + ex.getMessage();
        outErr.printErrLn(message);
        return createDetailedCommandResult(
            message,
            ExitCode.LOCAL_ENVIRONMENTAL_ERROR,
            FailureDetails.Command.Code.STARLARK_CPU_PROFILE_FILE_INITIALIZATION_FAILURE);
      }
      try {
        Starlark.startCpuProfile(out, Duration.ofMillis(10));
      } catch (IllegalStateException ex) { // e.g. SIGPROF in use
        String message = Strings.nullToEmpty(ex.getMessage());
        outErr.printErrLn(message);
        return createDetailedCommandResult(
            message,
            ExitCode.LOCAL_ENVIRONMENTAL_ERROR,
            FailureDetails.Command.Code.STARLARK_CPU_PROFILING_INITIALIZATION_FAILURE);
      }
    }

    BlazeCommandResult result =
        createDetailedCommandResult(
            "Unknown command failure",
            ExitCode.BLAZE_INTERNAL_ERROR,
            FailureDetails.Command.Code.COMMAND_FAILURE_UNKNOWN);
    boolean afterCommandCalled = false;
    Reporter reporter = env.getReporter();
    try (OutErr.SystemPatcher systemOutErrPatcher = reporter.getOutErr().getSystemPatcher()) {
      // Temporary: there are modules that output events during beforeCommand, but the reporter
      // isn't setup yet. Add the stored event handler to catch those events.
      reporter.addHandler(storedEventHandler);
      for (BlazeModule module : runtime.getBlazeModules()) {
        try (SilentCloseable closeable = Profiler.instance().profile(module + ".beforeCommand")) {
          module.beforeCommand(env);
        } catch (AbruptExitException e) {
          logger.atInfo().withCause(e).log("Error in %s", module);
          // Don't let one module's complaints prevent the other modules from doing necessary
          // setup. We promised to call beforeCommand exactly once per-module before each command
          // and will be calling afterCommand soon in the future - a module's afterCommand might
          // rightfully assume its beforeCommand has already been called.
          storedEventHandler.handle(Event.error(e.getMessage()));
          // It's not ideal but we can only return one exit code, so we just pick the code of the
          // last exception.
          earlyExitCode = e.getDetailedExitCode();
        }
      }
      reporter.removeHandler(storedEventHandler);

      // Setup stdout / stderr.
      outErr = tee(outErr, env.getOutputListeners());

      // Early exit. We need to guarantee that the ErrOut and Reporter setup below never error out,
      // so any invariants they need must be checked before this point.
      if (!earlyExitCode.isSuccess()) {
        replayEarlyExitEvents(
            outErr,
            optionHandler,
            storedEventHandler,
            env,
            new NoBuildEvent(
                commandName, firstContactTime, false, true, env.getCommandId().toString()));
        result = BlazeCommandResult.detailedExitCode(earlyExitCode);
        return result;
      }

      try (SilentCloseable closeable = Profiler.instance().profile("setup event handler")) {
        UiOptions eventHandlerOptions = options.getOptions(UiOptions.class);
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
          outErr = bufferOut(outErr);
        }

        if (!commandAnnotation.binaryStdErr()) {
          outErr = bufferErr(outErr);
        }

        DebugLoggerConfigurator.setupLogging(commonOptions.verbosity);

        EventHandler handler = createEventHandler(outErr, eventHandlerOptions);
        reporter.addHandler(handler);
        env.getEventBus().register(handler);

        runtime
            .getRetainedHeapLimiter()
            .update(commonOptions.oomMoreEagerlyThreshold, commonOptions.oomMessage, reporter);

        // We register an ANSI-allowing handler associated with {@code handler} so that ANSI control
        // codes can be re-introduced later even if blaze is invoked with --color=no. This is useful
        // for commands such as 'blaze run' where the output of the final executable shouldn't be
        // modified.
        if (!eventHandlerOptions.useColor()) {
          UiEventHandler ansiAllowingHandler =
              createEventHandler(colorfulOutErr, eventHandlerOptions);
          reporter.registerAnsiAllowingHandler(handler, ansiAllowingHandler);
          env.getEventBus().register(new PassiveExperimentalEventHandler(ansiAllowingHandler));
        }
      }

      try (SilentCloseable closeable = Profiler.instance().profile("replay stored events")) {
        // Now we're ready to replay the events.
        storedEventHandler.replayOn(reporter);
        for (String warning : commandEnvWarnings) {
          reporter.handle(Event.warn(warning));
        }
      }

      try (SilentCloseable closeable = Profiler.instance().profile("announce rc options")) {
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
          for (String note : optionHandler.getRcfileNotes()) {
            reporter.handle(Event.info(note));
          }
        }
      }

      // While a Blaze command is active, direct all errors to the client's event handler (and
      // out/err streams).
      systemOutErrPatcher.start();

      try (SilentCloseable closeable = Profiler.instance().profile("CommandEnv.beforeCommand")) {
        // Notify the BlazeRuntime, so it can do some initial setup.
        env.beforeCommand(invocationPolicy);
      } catch (AbruptExitException e) {
        logger.atInfo().withCause(e).log("Error before command");
        reporter.handle(Event.error(e.getMessage()));
        result = BlazeCommandResult.detailedExitCode(e.getDetailedExitCode());
        return result;
      }

      // Log the command line now that the modules have all had a change to register their listeners
      // to the event bus.
      env.getEventBus().post(unstructuredServerCommandLineEvent);
      env.getEventBus().post(originalCommandLineEvent);
      env.getEventBus().post(canonicalCommandLineEvent);
      env.getEventBus().post(commonOptions.toolCommandLine);

      for (BlazeModule module : runtime.getBlazeModules()) {
        try (SilentCloseable closeable =
            Profiler.instance().profile(module + ".injectExtraPrecomputedValues")) {
          env.getSkyframeExecutor().injectExtraPrecomputedValues(module.getPrecomputedValues());
        }
      }

      // {@link CleanCommand} is annotated with {@code builds = true}
      // to have access to relevant build options but don't actually do building.
      // {@link InfoCommand} is annotated with {@code builds = true} but only conditionally
      // does this step based on some complicated logic.
      if (commandAnnotation.builds()
          && !commandAnnotation.name().equals("clean")
          && !commandAnnotation.name().equals("info")) {
        try {
          env.syncPackageLoading(options);
        } catch (InterruptedException e) {
          Thread.currentThread().interrupt();
          String message = "command interrupted while syncing package loading";
          reporter.handle(Event.error(message));
          earlyExitCode =
              InterruptedFailureDetails.detailedExitCode(message, Code.PACKAGE_LOADING_SYNC);
        } catch (AbruptExitException e) {
          logger.atInfo().withCause(e).log("Error package loading");
          reporter.handle(Event.error(e.getMessage()));
          earlyExitCode = e.getDetailedExitCode();
        }
        if (!earlyExitCode.isSuccess()) {
          replayEarlyExitEvents(
              outErr,
              optionHandler,
              storedEventHandler,
              env,
              new NoBuildEvent(
                  commandName, firstContactTime, false, true, env.getCommandId().toString()));
          result = BlazeCommandResult.detailedExitCode(earlyExitCode);
          return result;
        }
      }

      // Parse starlark options.
      earlyExitCode = optionHandler.parseStarlarkOptions(env, storedEventHandler);
      if (!earlyExitCode.isSuccess()) {
        replayEarlyExitEvents(
            outErr,
            optionHandler,
            storedEventHandler,
            env,
            new NoBuildEvent(
                commandName, firstContactTime, false, true, env.getCommandId().toString()));
        result = BlazeCommandResult.detailedExitCode(earlyExitCode);
        return result;
      }
      options = optionHandler.getOptionsResult();

      // Run the command.
      result = command.exec(env, options);

      DetailedExitCode moduleExitCode = env.precompleteCommand(result.getDetailedExitCode());
      // If Blaze did not suffer an infrastructure failure, check for errors in modules.
      if (!result.getExitCode().isInfrastructureFailure() && moduleExitCode != null) {
        result = BlazeCommandResult.detailedExitCode(moduleExitCode);
      }

      // Finalize the Starlark CPU profile.
      if (!commonOptions.starlarkCpuProfile.isEmpty()) {
        try {
          Starlark.stopCpuProfile();
        } catch (IOException ex) {
          String message = "Starlark CPU profiler: " + ex.getMessage();
          reporter.handle(Event.error(message));
          if (result.getDetailedExitCode().isSuccess()) { // don't clobber existing error
            result =
                createDetailedCommandResult(
                    message,
                    ExitCode.LOCAL_ENVIRONMENTAL_ERROR,
                    FailureDetails.Command.Code.STARLARK_CPU_PROFILE_FILE_WRITE_FAILURE);
          }
        }
      }

      afterCommandCalled = true;
      return runtime.afterCommand(env, result);
    } catch (Throwable e) {
      outErr.printErr(
          "Internal error thrown during build. Printing stack trace: "
              + Throwables.getStackTraceAsString(e));
      e.printStackTrace();
      BugReport.printBug(outErr, e, commonOptions.oomMessage);
      bugReporter.sendBugReport(e, args);
      logger.atSevere().withCause(e).log("Shutting down due to exception");
      result = BlazeCommandResult.createShutdown(e);
      return result;
    } finally {
      if (!afterCommandCalled) {
        BlazeCommandResult newResult = runtime.afterCommand(env, result);
        if (!newResult.equals(result)) {
          logger.atWarning().log("afterCommand yielded different result: %s %s", result, newResult);
        }
      }
      // Swallow IOException, as we are already in a finally clause
      Flushables.flushQuietly(outErr.getOutputStream());
      Flushables.flushQuietly(outErr.getErrorStream());

      env.getTimestampGranularityMonitor().waitForTimestampGranularity(outErr);
    }
  }

  private static void replayEarlyExitEvents(
      OutErr outErr,
      BlazeOptionHandler optionHandler,
      StoredEventHandler storedEventHandler,
      CommandEnvironment env,
      NoBuildEvent noBuildEvent) {
    PrintingEventHandler printingEventHandler =
        new PrintingEventHandler(outErr, EventKind.ALL_EVENTS);

    Optional<String> badOption = retrieveBadOption(storedEventHandler.getEvents());

    for (String note : optionHandler.getRcfileNotes()) {
      if (badOption.isPresent()) {
        if (note.contains(badOption.get())) {
          printingEventHandler.handle(Event.info(note));
        }
      }
    }
    for (Event event : storedEventHandler.getEvents()) {
      printingEventHandler.handle(event);
    }
    for (Postable post : storedEventHandler.getPosts()) {
      env.getEventBus().post(post);
    }
    env.getEventBus().post(noBuildEvent);
  }

  private static Optional<String> retrieveBadOption(ImmutableList<Event> events) {
    return events.stream()
        .filter(e -> e.getTag() != null && e.getTag().equals(BAD_OPTION_TAG))
        .map(Event::getMessage)
        .filter(message -> message.contains(ERROR_SEPARATOR))
        .map(message -> message.substring(0, message.indexOf(ERROR_SEPARATOR)))
        .findFirst();
  }

  private OutErr bufferOut(OutErr outErr) {
    OutputStream wrappedOut = new BufferedOutputStream(outErr.getOutputStream());
    return OutErr.create(wrappedOut, outErr.getErrorStream());
  }

  private OutErr bufferErr(OutErr outErr) {
    OutputStream wrappedErr = new BufferedOutputStream(outErr.getErrorStream());
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

  private OutErr tee(OutErr outErr, List<OutErr> additionalOutErrs) {
    if (additionalOutErrs.isEmpty()) {
      return outErr;
    }
    DelegatingOutErr result = new DelegatingOutErr();
    result.addSink(outErr);
    for (OutErr additionalOutErr : additionalOutErrs) {
      result.addSink(additionalOutErr);
    }
    return result;
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
   * directly or by calling {@link BlazeOptionHandler#parseOptions} for command-specific options.
   */
  private OptionsParser createOptionsParser(BlazeCommand command)
      throws OptionsParser.ConstructionException {
    OpaqueOptionsData optionsData;
    try {
      optionsData = optionsDataCache.getUnchecked(command);
    } catch (UncheckedExecutionException e) {
      Throwables.throwIfInstanceOf(e.getCause(), OptionsParser.ConstructionException.class);
      throw new IllegalStateException(e);
    }
    Command annotation = command.getClass().getAnnotation(Command.class);
    OptionsParser parser =
        OptionsParser.builder()
            .optionsData(optionsData)
            .skipStarlarkOptionPrefixes()
            .allowResidue(annotation.allowResidue())
            .withAliasFlag(BLAZE_ALIASING_FLAG)
            .build();
    return parser;
  }

  /** Returns the event handler to use for this Blaze command. */
  private UiEventHandler createEventHandler(OutErr outErr, UiOptions eventOptions) {
    Path workspacePath = runtime.getWorkspace().getDirectories().getWorkspace();
    PathFragment workspacePathFragment = workspacePath == null ? null : workspacePath.asFragment();
    return new UiEventHandler(outErr, eventOptions, runtime.getClock(), workspacePathFragment);
  }

  /** Returns the runtime instance shared by the commands that this dispatcher dispatches to. */
  @VisibleForTesting
  public BlazeRuntime getRuntime() {
    return runtime;
  }

  /**
   * Shuts down all the registered commands to give them a chance to cleanup or close resources.
   * Should be called by the owner of this command dispatcher in all termination cases.
   */
  public void shutdown() {
    closeSilently(logOutputStream);
    logOutputStream = null;
  }

  private static BlazeCommandResult createDetailedCommandResult(
      String message, ExitCode exitCode, FailureDetails.Command.Code detailedCode) {
    return BlazeCommandResult.detailedExitCode(
        DetailedExitCode.of(
            exitCode,
            FailureDetail.newBuilder()
                .setMessage(message)
                .setCommand(FailureDetails.Command.newBuilder().setCode(detailedCode))
                .build()));
  }
}
