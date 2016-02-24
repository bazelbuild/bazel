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
import com.google.common.base.Joiner;
import com.google.common.base.Predicates;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.ListMultimap;
import com.google.common.collect.Lists;
import com.google.common.io.Flushables;
import com.google.devtools.build.lib.Constants;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.AnsiStrippingOutputStream;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.util.LoggingUtil;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.io.DelegatingOutErr;
import com.google.devtools.build.lib.util.io.OutErr;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.common.options.OptionPriority;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;

import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.logging.Level;

import javax.annotation.Nullable;

/**
 * Dispatches to the Blaze commands; that is, given a command line, this
 * abstraction looks up the appropriate command object, parses the options
 * required by the object, and calls its exec method. Also, this object provides
 * the runtime state (BlazeRuntime) to the commands.
 */
public class BlazeCommandDispatcher {

  // Keep in sync with options added in OptionProcessor::AddRcfileArgsAndOptions()
  private static final Set<String> INTERNAL_COMMAND_OPTIONS = ImmutableSet.of(
      "rc_source", "default_override", "isatty", "terminal_columns", "ignore_client_env",
      "client_env", "client_cwd");

  private static final ImmutableList<String> HELP_COMMAND = ImmutableList.of("help");

  private static final Set<String> ALL_HELP_OPTIONS = ImmutableSet.of("--help", "-help", "-h");

  /**
   * By throwing this exception, a command indicates that it wants to shutdown
   * the Blaze server process.
   * See {@link BlazeCommandDispatcher#exec(List, OutErr, long)}.
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

  private OutputStream logOutputStream = null;

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
  }

  /**
   * Only some commands work if cwd != workspaceSuffix in Blaze. In that case, also check if Blaze
   * was called from the output directory and fail if it was.
   */
  private ExitCode checkCwdInWorkspace(Command commandAnnotation, String commandName,
      OutErr outErr) {
    if (!commandAnnotation.mustRunInWorkspace()) {
      return ExitCode.SUCCESS;
    }

    if (!runtime.inWorkspace()) {
      outErr.printErrLn("The '" + commandName + "' command is only supported from within a "
          + "workspace.");
      return ExitCode.COMMAND_LINE_ERROR;
    }

    Path workspace = runtime.getWorkspace();
    // TODO(kchodorow): Remove this once spaces are supported.
    if (workspace.getPathString().contains(" ")) {
      outErr.printErrLn(Constants.PRODUCT_NAME + " does not currently work properly from paths "
          + "containing spaces (" + workspace + ").");
      return ExitCode.LOCAL_ENVIRONMENTAL_ERROR;
    }

    Path doNotBuild = workspace.getParentDirectory().getRelative(
        BlazeRuntime.DO_NOT_BUILD_FILE_NAME);

    if (doNotBuild.exists()) {
      if (!commandAnnotation.canRunInOutputDirectory()) {
        outErr.printErrLn(getNotInRealWorkspaceError(doNotBuild));
        return ExitCode.COMMAND_LINE_ERROR;
      } else {
        outErr.printErrLn("WARNING: Blaze is run from output directory. This is unsound.");
      }
    }
    return ExitCode.SUCCESS;
  }

  private void parseArgsAndConfigs(OptionsParser optionsParser, Command commandAnnotation,
      List<String> args, List<String> rcfileNotes, OutErr outErr)
          throws OptionsParsingException {

    Function<String, String> commandOptionSourceFunction = new Function<String, String>() {
      @Override
      public String apply(String input) {
        if (INTERNAL_COMMAND_OPTIONS.contains(input)) {
          return "options generated by Blaze launcher";
        } else {
          return "command line options";
        }
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
        getOptionsMap(outErr, rcFileOptions.rcSource, rcFileOptions.optionsOverrides,
            runtime.getCommandMap().keySet());

    parseOptionsForCommand(rcfileNotes, commandAnnotation, optionsParser, optionsMap, null, null);

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
      outErr.printErrLn("WARNING: Config values are not defined in any .rc file: "
          + Joiner.on(", ").join(unknownConfigs));
    }
  }

  /**
   * Executes a single command. Returns the Unix exit status for the Blaze
   * client process, or throws {@link ShutdownBlazeServerException} to
   * indicate that a command wants to shutdown the Blaze server.
   */
  int exec(List<String> args, OutErr outErr, long firstContactTime)
      throws ShutdownBlazeServerException {
    // Record the start time for the profiler and the timestamp granularity monitor. Do not put
    // anything before this!
    long execStartTimeNanos = runtime.getClock().nanoTime();

    // Record the command's starting time again, for use by
    // TimestampGranularityMonitor.waitForTimestampGranularity().
    // This should be done as close as possible to the start of
    // the command's execution - that's why we do this separately,
    // rather than in runtime.beforeCommand().
    runtime.getTimestampGranularityMonitor().setCommandStartTime();
    CommandEnvironment env = runtime.initCommand();
    // Record the command's starting time for use by the commands themselves.
    env.recordCommandStartTime(firstContactTime);

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
          "Command '%s' not found. Try '%s help'.", commandName, Constants.PRODUCT_NAME));
      return ExitCode.COMMAND_LINE_ERROR.getNumericExitCode();
    }
    Command commandAnnotation = command.getClass().getAnnotation(Command.class);

    AbruptExitException exitCausingException = null;
    for (BlazeModule module : runtime.getBlazeModules()) {
      try {
        module.beforeCommand(commandAnnotation, env);
      } catch (AbruptExitException e) {
        // Don't let one module's complaints prevent the other modules from doing necessary
        // setup. We promised to call beforeCommand exactly once per-module before each command
        // and will be calling afterCommand soon in the future - a module's afterCommand might
        // rightfully assume its beforeCommand has already been called.
        outErr.printErrLn(e.getMessage());
        // It's not ideal but we can only return one exit code, so we just pick the code of the
        // last exception.
        exitCausingException = e;
      }
    }
    if (exitCausingException != null) {
      return exitCausingException.getExitCode().getNumericExitCode();
    }

    try {
      Path commandLog = getCommandLogPath(runtime.getOutputBase());

      // Unlink old command log from previous build, if present, so scripts
      // reading it don't conflate it with the command log we're about to write.
      commandLog.delete();

      logOutputStream = commandLog.getOutputStream();
      outErr = tee(outErr, OutErr.create(logOutputStream, logOutputStream));
    } catch (IOException ioException) {
      LoggingUtil.logToRemote(
          Level.WARNING, "Unable to delete or open command.log", ioException);
    }

    ExitCode result = checkCwdInWorkspace(commandAnnotation, commandName, outErr);
    if (result != ExitCode.SUCCESS) {
      return result.getNumericExitCode();
    }

    OptionsParser optionsParser;
    // Delay output of notes regarding the parsed rc file, so it's possible to disable this in the
    // rc file.
    List<String> rcfileNotes = new ArrayList<>();
    try {
      optionsParser = createOptionsParser(command);
      parseArgsAndConfigs(optionsParser, commandAnnotation, args, rcfileNotes, outErr);

      InvocationPolicyEnforcer optionsPolicyEnforcer =
          InvocationPolicyEnforcer.create(getRuntime().getStartupOptionsProvider());
      optionsPolicyEnforcer.enforce(optionsParser, commandName);
    } catch (OptionsParsingException e) {
      for (String note : rcfileNotes) {
        outErr.printErrLn("INFO: " + note);
      }
      outErr.printErrLn(e.getMessage());
      return ExitCode.COMMAND_LINE_ERROR.getNumericExitCode();
    }

    // Setup log filtering
    BlazeCommandEventHandler.Options eventHandlerOptions =
        optionsParser.getOptions(BlazeCommandEventHandler.Options.class);
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

    CommonCommandOptions commonOptions = optionsParser.getOptions(CommonCommandOptions.class);
    BlazeRuntime.setupLogging(commonOptions.verbosity);

    // Do this before an actual crash so we don't have to worry about
    // allocating memory post-crash.
    String[] crashData = runtime.getCrashData(env);
    int numericExitCode = ExitCode.BLAZE_INTERNAL_ERROR.getNumericExitCode();
    PrintStream savedOut = System.out;
    PrintStream savedErr = System.err;

    EventHandler handler = createEventHandler(outErr, eventHandlerOptions);
    Reporter reporter = env.getReporter();
    reporter.addHandler(handler);

    // We register an ANSI-allowing handler associated with {@code handler} so that ANSI control
    // codes can be re-introduced later even if blaze is invoked with --color=no. This is useful
    // for commands such as 'blaze run' where the output of the final executable shouldn't be
    // modified.
    EventHandler ansiAllowingHandler = null;
    if (!eventHandlerOptions.useColor()) {
      ansiAllowingHandler = createEventHandler(colorfulOutErr, eventHandlerOptions);
      reporter.registerAnsiAllowingHandler(handler, ansiAllowingHandler);
    }

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
        for (String note : rcfileNotes) {
          reporter.handle(Event.info(note));
        }
      }

      try {
        // Notify the BlazeRuntime, so it can do some initial setup.
        env.beforeCommand(commandAnnotation, optionsParser, commonOptions, execStartTimeNanos);
        // Allow the command to edit options after parsing:
        command.editOptions(env, optionsParser);
      } catch (AbruptExitException e) {
        reporter.handle(Event.error(e.getMessage()));
        return e.getExitCode().getNumericExitCode();
      }

      // Print warnings for odd options usage
      for (String warning : optionsParser.getWarnings()) {
        reporter.handle(Event.warn(warning));
      }

      ExitCode outcome = command.exec(env, optionsParser);
      outcome = env.precompleteCommand(outcome);
      numericExitCode = outcome.getNumericExitCode();
      return numericExitCode;
    } catch (ShutdownBlazeServerException e) {
      numericExitCode = e.getExitStatus();
      throw e;
    } catch (Throwable e) {
      BugReport.printBug(outErr, e);
      BugReport.sendBugReport(e, args, crashData);
      numericExitCode = e instanceof OutOfMemoryError
          ? ExitCode.OOM_ERROR.getNumericExitCode()
          : ExitCode.BLAZE_INTERNAL_ERROR.getNumericExitCode();
      throw new ShutdownBlazeServerException(numericExitCode, e);
    } finally {
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
      runtime.getTimestampGranularityMonitor().waitForTimestampGranularity(outErr);
    }
  }

  /**
   * For testing ONLY. Same as {@link #exec(List, OutErr, long)}, but automatically uses the current
   * time.
   */
  @VisibleForTesting
  public int exec(List<String> args, OutErr originalOutErr) throws ShutdownBlazeServerException {
    return exec(args, originalOutErr, runtime.getClock().currentTimeMillis());
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
      Iterables.addAll(
          unknownConfigs,
          Iterables.filter(configs, Predicates.not(Predicates.in(knownConfigs))));
    }
  }

  // Processes the option list for an .rc file - command pair.
  private static void processOptionList(OptionsParser optionsParser, String commandToParse,
      String originalCommand, List<String> rcfileNotes, String rcfile, List<String> rcfileOptions)
      throws OptionsParsingException {
    if (!rcfileOptions.isEmpty()) {
      String inherited = commandToParse.equals(originalCommand) ? "" : "Inherited ";
      rcfileNotes.add("Reading options for '" + originalCommand +
          "' from " + rcfile + ":\n" +
          "  " + inherited + "'" + commandToParse + "' options: "
        + Joiner.on(' ').join(rcfileOptions));
      optionsParser.parse(OptionPriority.RC_FILE, rcfile, rcfileOptions);
    }
  }

  private static List<String> getCommandNamesToParse(Command commandAnnotation) {
    List<String> result = new ArrayList<>();
    getCommandNamesToParseHelper(commandAnnotation, result);
    result.add("common");
    // TODO(bazel-team): This statement is a NO-OP: Lists.reverse(result);
    return result;
  }

  private static void getCommandNamesToParseHelper(Command commandAnnotation,
      List<String> accumulator) {
    for (Class<? extends BlazeCommand> base : commandAnnotation.inherits()) {
      getCommandNamesToParseHelper(base.getAnnotation(Command.class), accumulator);
    }
    accumulator.add(commandAnnotation.name());
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
    String message = "Blaze should not be called from a Blaze output directory. ";
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

  /**
   * For a given output_base directory, returns the command log file path.
   */
  public static Path getCommandLogPath(Path outputBase) {
    return outputBase.getRelative("command.log");
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
   * Creates an option parser using the common options classes and the
   * command-specific options classes.
   *
   * <p>An overriding method should first call this method and can then
   * override default values directly or by calling {@link
   * #parseOptionsForCommand} for command-specific options.
   *
   * @throws OptionsParsingException
   */
  protected OptionsParser createOptionsParser(BlazeCommand command)
      throws OptionsParsingException {
    Command annotation = command.getClass().getAnnotation(Command.class);
    List<Class<? extends OptionsBase>> allOptions = Lists.newArrayList();
    allOptions.addAll(BlazeCommandUtils.getOptions(
        command.getClass(), getRuntime().getBlazeModules(), getRuntime().getRuleClassProvider()));
    OptionsParser parser = OptionsParser.newOptionsParser(allOptions);
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
      OutErr outErr,
      List<String> rcFiles,
      List<CommonCommandOptions.OptionOverride> overrides,
      Set<String> validCommands) {
    List<Pair<String, ListMultimap<String, String>>> result = new ArrayList<>();

    String lastRcFile = null;
    ListMultimap<String, String> lastMap = null;
    for (CommonCommandOptions.OptionOverride override : overrides) {
      if (override.blazeRc < 0 || override.blazeRc >= rcFiles.size()) {
        outErr.printErrLn("WARNING: inconsistency in generated command line "
            + "args. Ignoring bogus argument\n");
        continue;
      }
      String rcFile = rcFiles.get(override.blazeRc);

      String command = override.command;
      int index = command.indexOf(':');
      if (index > 0) {
        command = command.substring(0, index);
      }
      if (!validCommands.contains(command) && !command.equals("common")) {
        outErr.printErrLn("WARNING: while reading option defaults file '"
            + rcFile + "':\n"
            + "  invalid command name '" + override.command + "'.");
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
      return new ExperimentalEventHandler(outErr, eventOptions);
    } else if ((eventOptions.useColor() || eventOptions.useCursorControl())) {
      eventHandler = new FancyTerminalEventHandler(outErr, eventOptions);
    } else {
      eventHandler = new BlazeCommandEventHandler(outErr, eventOptions);
    }

    return RateLimitingEventHandler.create(eventHandler, eventOptions.showProgressRateLimit);
  }

  /**
   * Unsets the event handler.
   */
  private void releaseHandler(EventHandler eventHandler) {
    if (eventHandler instanceof FancyTerminalEventHandler) {
      // Make sure that the terminal state of the old event handler is clear
      // before creating a new one.
      ((FancyTerminalEventHandler) eventHandler).resetTerminal();
    }
    if (eventHandler instanceof ExperimentalEventHandler) {
      ((ExperimentalEventHandler) eventHandler).resetTerminal();
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
