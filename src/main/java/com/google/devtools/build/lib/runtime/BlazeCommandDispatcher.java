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
import com.google.common.base.Preconditions;
import com.google.common.base.Throwables;
import com.google.common.base.Verify;
import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.io.Flushables;
import com.google.common.util.concurrent.UncheckedExecutionException;
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
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.AnsiStrippingOutputStream;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.util.LoggingUtil;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.io.DelegatingOutErr;
import com.google.devtools.build.lib.util.io.OutErr;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.common.options.OpaqueOptionsData;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;
import com.google.devtools.common.options.OptionsParsingResult;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Dispatches to the Blaze commands; that is, given a command line, this
 * abstraction looks up the appropriate command object, parses the options
 * required by the object, and calls its exec method. Also, this object provides
 * the runtime state (BlazeRuntime) to the commands.
 */
public class BlazeCommandDispatcher {
  private static final Logger logger = Logger.getLogger(BlazeCommandDispatcher.class.getName());

  /**
   * What to do if the command lock is not available.
   */
  public enum LockingMode {
    WAIT,  // Wait until it is available
    ERROR_OUT,  // Return with an error
  }

  private static final ImmutableList<String> HELP_COMMAND = ImmutableList.of("help");

  private static final ImmutableSet<String> ALL_HELP_OPTIONS =
      ImmutableSet.of("--help", "-help", "-h");

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
   * Executes a single command. Returns a {@link BlazeCommandResult} to indicate either an exit
   * code, the desire to shut down the server, or that a given binary should be executed by the
   * client.
   */
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
      outErr.printErrLn(String.format(
          "Command '%s' not found. Try '%s help'.", commandName, runtime.getProductName()));
      return BlazeCommandResult.exitCode(ExitCode.COMMAND_LINE_ERROR);
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
              outErr.printErrLn("Another command (" + currentClientDescription + ") is running. "
                  + " Waiting for it to complete on the server...");
              otherClientDescription = currentClientDescription;
            }
            commandLock.wait(500);
            break;

          case ERROR_OUT:
            outErr.printErrLn(String.format("Another command (" + currentClientDescription + ") is "
                + "running. Exiting immediately."));
            return BlazeCommandResult.exitCode(ExitCode.LOCK_HELD_NOBLOCK_FOR_LOCK);

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
        outErr.printErrLn("Server shut down " + shutdownReason);
        return BlazeCommandResult.exitCode(ExitCode.LOCAL_ENVIRONMENTAL_ERROR);
      }
      BlazeCommandResult result = execExclusively(
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
      return result;
    } finally {
      synchronized (commandLock) {
        currentClientDescription = null;
        commandLock.notify();
      }
    }
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
    ExitCode earlyExitCode = optionHandler.parseOptions(args, storedEventHandler);
    OptionsParsingResult options = optionHandler.getOptionsResult();

    CommandLineEvent originalCommandLineEvent =
        new CommandLineEvent.OriginalCommandLineEvent(
            runtime, commandName, options, startupOptionsTaggedWithBazelRc);
    CommandLineEvent canonicalCommandLineEvent =
        new CommandLineEvent.CanonicalCommandLineEvent(runtime, commandName, options);

    // The initCommand call also records the start time for the timestamp granularity monitor.
    List<String> commandEnvWarnings = new ArrayList<>();
    CommandEnvironment env = workspace.initCommand(commandAnnotation, options, commandEnvWarnings);
    // Record the command's starting time for use by the commands themselves.
    env.recordCommandStartTime(firstContactTime);
    CommonCommandOptions commonOptions = options.getOptions(CommonCommandOptions.class);
    // TODO(ulfjack): Move the profiler initialization as early in the startup sequence as possible.
    // Profiler setup and shutdown must always happen in pairs. Shutdown is currently performed in
    // the afterCommand call in the finally block below.
    Path profilePath =
        runtime.initProfiler(
            storedEventHandler,
            workspace,
            commonOptions,
            env.getCommandId(),
            execStartTimeNanos,
            waitTimeInMs);
    if (commonOptions.postProfileStartedEvent) {
      storedEventHandler.post(new ProfilerStartedEvent(profilePath));
    }

    BlazeCommandResult result = BlazeCommandResult.exitCode(ExitCode.BLAZE_INTERNAL_ERROR);
    PrintStream savedOut = System.out;
    PrintStream savedErr = System.err;
    boolean afterCommandCalled = false;
    try {
      // Temporary: there are modules that output events during beforeCommand, but the reporter
      // isn't setup yet. Add the stored event handler to catch those events.
      env.getReporter().addHandler(storedEventHandler);
      for (BlazeModule module : runtime.getBlazeModules()) {
        try (SilentCloseable closeable = Profiler.instance().profile(module + ".beforeCommand")) {
          module.beforeCommand(env);
        } catch (AbruptExitException e) {
          // Don't let one module's complaints prevent the other modules from doing necessary
          // setup. We promised to call beforeCommand exactly once per-module before each command
          // and will be calling afterCommand soon in the future - a module's afterCommand might
          // rightfully assume its beforeCommand has already been called.
          storedEventHandler.handle(Event.error(e.getMessage()));
          // It's not ideal but we can only return one exit code, so we just pick the code of the
          // last exception.
          earlyExitCode = e.getExitCode();
        }
      }
      env.getReporter().removeHandler(storedEventHandler);

      // Setup stdout / stderr.
      outErr = tee(outErr, env.getOutputListeners());

      // Early exit. We need to guarantee that the ErrOut and Reporter setup below never error out,
      // so any invariants they need must be checked before this point.
      if (!earlyExitCode.equals(ExitCode.SUCCESS)) {
        // Partial replay of the printed events before we exit.
        PrintingEventHandler printingEventHandler =
            new PrintingEventHandler(outErr, EventKind.ALL_EVENTS);
        for (String note : optionHandler.getRcfileNotes()) {
          printingEventHandler.handle(Event.info(note));
        }
        for (Event event : storedEventHandler.getEvents()) {
          printingEventHandler.handle(event);
        }
        for (Postable post : storedEventHandler.getPosts()) {
          env.getEventBus().post(post);
        }
        result = BlazeCommandResult.exitCode(earlyExitCode);
        return result;
      }

      Reporter reporter = env.getReporter();
      try (SilentCloseable closeable = Profiler.instance().profile("setup event handler")) {
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

        if (!commonOptions.verbosity.equals(lastLogVerbosityLevel)) {
          BlazeRuntime.setupLogging(commonOptions.verbosity);
          lastLogVerbosityLevel = commonOptions.verbosity;
        }

        EventHandler handler = createEventHandler(outErr, eventHandlerOptions);
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
          return BlazeCommandResult.exitCode(ExitCode.COMMAND_LINE_ERROR);
        }
        if (oomMoreEagerlyThreshold != 100) {
          try {
            RetainedHeapLimiter.maybeInstallRetainedHeapLimiter(oomMoreEagerlyThreshold);
          } catch (OptionsParsingException e) {
            reporter.handle(Event.error(e.getMessage()));
            return BlazeCommandResult.exitCode(ExitCode.COMMAND_LINE_ERROR);
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

      // While a Blaze command is active, direct all errors to the client's
      // event handler (and out/err streams).
      OutErr reporterOutErr = reporter.getOutErr();
      System.setOut(new PrintStream(reporterOutErr.getOutputStream(), /*autoflush=*/ true));
      System.setErr(new PrintStream(reporterOutErr.getErrorStream(), /*autoflush=*/ true));

      try (SilentCloseable closeable = Profiler.instance().profile("CommandEnv.beforeCommand")) {
        // Notify the BlazeRuntime, so it can do some initial setup.
        env.beforeCommand(
            options,
            commonOptions,
            waitTimeInMs,
            invocationPolicy);
      } catch (AbruptExitException e) {
        reporter.handle(Event.error(e.getMessage()));
        return BlazeCommandResult.exitCode(e.getExitCode());
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

      result = command.exec(env, options);
      ExitCode moduleExitCode = env.precompleteCommand(result.getExitCode());
      // If Blaze did not suffer an infrastructure failure, check for errors in modules.
      if (!result.getExitCode().isInfrastructureFailure() && moduleExitCode != null) {
        result = BlazeCommandResult.exitCode(moduleExitCode);
      }

      afterCommandCalled = true;
      return runtime.afterCommand(env, result);
    } catch (Throwable e) {
      outErr.printErr(
          "Internal error thrown during build. Printing stack trace: "
              + Throwables.getStackTraceAsString(e));
      e.printStackTrace();
      BugReport.printBug(outErr, e);
      BugReport.sendBugReport(e, args, env.getCrashData());
      logger.log(Level.SEVERE, "Shutting down due to exception", e);
      return BlazeCommandResult.shutdown(BugReport.getExitCodeForThrowable(e));
    } finally {
      if (!afterCommandCalled) {
        runtime.afterCommand(env, result);
      }
      // Swallow IOException, as we are already in a finally clause
      Flushables.flushQuietly(outErr.getOutputStream());
      Flushables.flushQuietly(outErr.getErrorStream());

      System.setOut(savedOut);
      System.setErr(savedErr);
      env.getTimestampGranularityMonitor().waitForTimestampGranularity(outErr);
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
    OptionsParser parser = OptionsParser.newOptionsParser(optionsData);
    parser.setAllowResidue(annotation.allowResidue());
    return parser;
  }

  /** Returns the event handler to use for this Blaze command. */
  private EventHandler createEventHandler(
      OutErr outErr, BlazeCommandEventHandler.Options eventOptions) {
    Path workspacePath = runtime.getWorkspace().getDirectories().getWorkspace();
    PathFragment workspacePathFragment = workspacePath == null ? null : workspacePath.asFragment();
    EventHandler eventHandler;
    if (eventOptions.experimentalUi) {
      // The experimental event handler is not to be rate limited, so don't wrap it in a
      // RateLimitingEventHandler.
      return new ExperimentalEventHandler(
          outErr, eventOptions, runtime.getClock(), workspacePathFragment);
    } else if ((eventOptions.useColor() || eventOptions.useCursorControl())) {
      eventHandler = new FancyTerminalEventHandler(outErr, eventOptions, workspacePathFragment);
    } else {
      eventHandler = new BlazeCommandEventHandler(outErr, eventOptions, workspacePathFragment);
    }

    return RateLimitingEventHandler.create(eventHandler, eventOptions.showProgressRateLimit);
  }

  /**
   * Returns the runtime instance shared by the commands that this dispatcher
   * dispatches to.
   */
  @VisibleForTesting
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
