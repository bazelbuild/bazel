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
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ListMultimap;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.runtime.commands.ProjectFileSupport;
import com.google.devtools.build.lib.runtime.proto.InvocationPolicyOuterClass.InvocationPolicy;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.common.options.InvocationPolicyEnforcer;
import com.google.devtools.common.options.OptionDefinition;
import com.google.devtools.common.options.OptionPriority.PriorityCategory;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;
import com.google.devtools.common.options.OptionsParsingResult;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.function.Function;
import java.util.logging.Level;

/**
 * Handles parsing the blaze command arguments.
 *
 * <p>This class manages rc options, configs, default options, and invocation policy.
 */
public final class BlazeOptionHandler {
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

  private final BlazeRuntime runtime;
  private final OptionsParser optionsParser;
  private final BlazeWorkspace workspace;
  private final BlazeCommand command;
  private final Command commandAnnotation;
  private final InvocationPolicy invocationPolicy;
  private final List<String> rcfileNotes = new ArrayList<>();

  BlazeOptionHandler(
      BlazeRuntime runtime,
      BlazeWorkspace workspace,
      BlazeCommand command,
      Command commandAnnotation,
      OptionsParser optionsParser,
      InvocationPolicy invocationPolicy) {
    this.runtime = runtime;
    this.workspace = workspace;
    this.command = command;
    this.commandAnnotation = commandAnnotation;
    this.optionsParser = optionsParser;
    this.invocationPolicy = invocationPolicy;
  }

  /**
   * Return options as {@link OptionsParsingResult} so the options can't be easily modified after
   * we've applied the invocation policy.
   */
  OptionsParsingResult getOptionsResult() {
    return optionsParser;
  }

  public List<String> getRcfileNotes() {
    return rcfileNotes;
  }

  /**
   * Only some commands work if cwd != workspaceSuffix in Blaze. In that case, also check if Blaze
   * was called from the output directory and fail if it was.
   */
  private ExitCode checkCwdInWorkspace(EventHandler eventHandler) {
    if (!commandAnnotation.mustRunInWorkspace()) {
      return ExitCode.SUCCESS;
    }

    if (!workspace.getDirectories().inWorkspace()) {
      eventHandler.handle(
          Event.error(
              "The '"
                  + commandAnnotation.name()
                  + "' command is only supported from within a workspace"
                  + " (below a directory having a WORKSPACE file).\n"
                  + "See documentation at"
                  + " https://docs.bazel.build/versions/master/build-ref.html#workspace"));
      return ExitCode.COMMAND_LINE_ERROR;
    }

    Path workspacePath = workspace.getWorkspace();
    // TODO(kchodorow): Remove this once spaces are supported.
    if (workspacePath.getPathString().contains(" ")) {
      eventHandler.handle(
          Event.error(
              runtime.getProductName()
                  + " does not currently work properly from paths "
                  + "containing spaces ("
                  + workspace
                  + ")."));
      return ExitCode.LOCAL_ENVIRONMENTAL_ERROR;
    }

    if (workspacePath.getParentDirectory() != null) {
      Path doNotBuild =
          workspacePath.getParentDirectory().getRelative(BlazeWorkspace.DO_NOT_BUILD_FILE_NAME);

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
    }
    return ExitCode.SUCCESS;
  }

  /**
   * Parses the unconditional options from .rc files for the current command.
   *
   * <p>This is not as trivial as simply taking the list of options for the specified command
   * because commands can inherit arguments from each other, and we have to respect that (e.g. if an
   * option is specified for 'build', it needs to take effect for the 'test' command, too). More
   * specific commands should have priority over the broader commands (say a "build" option that
   * conflicts with a "common" option should override the common one regardless of order.)
   *
   * <p>For each command, the options are parsed in rc order. This uses the master rc file first,
   * and follows import statements. This is the order in which they were passed by the client.
   */
  void parseRcOptions(
      EventHandler eventHandler, ListMultimap<String, RcChunkOfArgs> commandToRcArgs)
      throws OptionsParsingException {
    for (String commandToParse : getCommandNamesToParse(commandAnnotation)) {
      // Get all args defined for this command (or "common"), grouped by rc chunk.
      for (RcChunkOfArgs rcArgs : commandToRcArgs.get(commandToParse)) {
        if (!rcArgs.getArgs().isEmpty()) {
          String inherited = commandToParse.equals(commandAnnotation.name()) ? "" : "Inherited ";
          String source =
              rcArgs.getRcFile().equals("client")
                  ? "Options provided by the client"
                  : String.format(
                      "Reading rc options for '%s' from %s",
                      commandAnnotation.name(), rcArgs.getRcFile());
          rcfileNotes.add(
              String.format(
                  "%s:\n  %s'%s' options: %s",
                  source, inherited, commandToParse, Joiner.on(' ').join(rcArgs.getArgs())));
        }
        optionsParser.parse(PriorityCategory.RC_FILE, rcArgs.getRcFile(), rcArgs.getArgs());
      }
    }
  }

  private void parseArgsAndConfigs(List<String> args, ExtendedEventHandler eventHandler)
      throws OptionsParsingException {
    Path workspaceDirectory = workspace.getWorkspace();
    // TODO(ulfjack): The working directory is passed by the client as part of CommonCommandOptions,
    // and we can't know it until after we've parsed the options, so use the workspace for now.
    Path workingDirectory = workspace.getWorkspace();

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
    optionsParser.parseWithSourceFunction(
        PriorityCategory.COMMAND_LINE, commandOptionSourceFunction, cmdLineAfterCommand);

    // Command-specific options from .blazerc passed in via --default_override and --rc_source.
    ClientOptions rcFileOptions = optionsParser.getOptions(ClientOptions.class);
    ListMultimap<String, RcChunkOfArgs> commandToRcArgs =
        structureRcOptionsAndConfigs(
            eventHandler,
            rcFileOptions.rcSource,
            rcFileOptions.optionsOverrides,
            runtime.getCommandMap().keySet());
    parseRcOptions(eventHandler, commandToRcArgs);

    if (commandAnnotation.builds()) {
      // splits project files from targets in the traditional sense
      ProjectFileSupport.handleProjectFiles(
          eventHandler,
          runtime.getProjectFileProvider(),
          workspaceDirectory,
          workingDirectory,
          optionsParser,
          commandAnnotation.name());
    }

    expandConfigOptions(eventHandler, commandToRcArgs);
  }

  /**
   * TODO(bazel-team): When we move CoreOptions options to be defined in starlark, make sure they're
   * not passed in here during {@link #getOptionsResult}.
   */
  ExitCode parseStarlarkOptions(CommandEnvironment env, ExtendedEventHandler eventHandler) {
    // For now, restrict starlark options to commands that already build to ensure that loading
    // will work. We may want to open this up to other commands in the future. The "info"
    // and "clean" commands have builds=true set in their annotation but don't actually do any
    // building (b/120041419).
    if (!commandAnnotation.builds()
        || commandAnnotation.name().equals("info")
        || commandAnnotation.name().equals("clean")) {
      return ExitCode.SUCCESS;
    }
    try {
      StarlarkOptionsParser.newStarlarkOptionsParser(env, optionsParser).parse(eventHandler);
    } catch (OptionsParsingException e) {
      env.getReporter().handle(Event.error(e.getMessage()));
      return ExitCode.PARSING_FAILURE;
    }
    return ExitCode.SUCCESS;
  }

  /**
   * Parses the options, taking care not to generate any output to outErr, return, or throw an
   * exception.
   *
   * @return ExitCode.SUCCESS if everything went well, or some other value if not
   */
  ExitCode parseOptions(List<String> args, ExtendedEventHandler eventHandler) {
    // The initialization code here was carefully written to parse the options early before we call
    // into the BlazeModule APIs, which means we must not generate any output to outErr, return, or
    // throw an exception. All the events happening here are instead stored in a temporary event
    // handler, and later replayed.
    ExitCode earlyExitCode = checkCwdInWorkspace(eventHandler);
    if (!earlyExitCode.equals(ExitCode.SUCCESS)) {
      return earlyExitCode;
    }

    try {
      parseArgsAndConfigs(args, eventHandler);
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
      optionsPolicyEnforcer.enforce(optionsParser, commandAnnotation.name());
      // Print warnings for odd options usage
      for (String warning : optionsParser.getWarnings()) {
        eventHandler.handle(Event.warn(warning));
      }
      CommonCommandOptions commonOptions = optionsParser.getOptions(CommonCommandOptions.class);
      for (String warning : commonOptions.deprecationWarnings) {
        eventHandler.handle(Event.warn(warning));
      }
    } catch (OptionsParsingException e) {
      eventHandler.handle(Event.error(e.getMessage()));
      return ExitCode.COMMAND_LINE_ERROR;
    }
    return ExitCode.SUCCESS;
  }

  /**
   * Expand the values of --config according to the definitions provided in the rc files and the
   * applicable command.
   */
  void expandConfigOptions(
      EventHandler eventHandler, ListMultimap<String, RcChunkOfArgs> commandToRcArgs)
      throws OptionsParsingException {
    ConfigExpander.expandConfigOptions(
        eventHandler,
        commandToRcArgs,
        getCommandNamesToParse(commandAnnotation),
        rcfileNotes::add,
        optionsParser);
  }

  private static List<String> getCommandNamesToParse(Command commandAnnotation) {
    List<String> result = new ArrayList<>();
    result.add("common");
    getCommandNamesToParseHelper(commandAnnotation, result);
    return result;
  }

  private static void getCommandNamesToParseHelper(
      Command commandAnnotation, List<String> accumulator) {
    for (Class<? extends BlazeCommand> base : commandAnnotation.inherits()) {
      getCommandNamesToParseHelper(base.getAnnotation(Command.class), accumulator);
    }
    accumulator.add(commandAnnotation.name());
  }

  private String getNotInRealWorkspaceError(Path doNotBuildFile) {
    String message =
        String.format(
            "%1$s should not be called from a %1$s output directory. ", runtime.getProductName());
    try {
      String realWorkspace = new String(FileSystemUtils.readContentAsLatin1(doNotBuildFile));
      message += String.format("The pertinent workspace directory is: '%s'", realWorkspace);
    } catch (IOException e) {
      // We are exiting anyway.
    }

    return message;
  }

  /**
   * The rc options are passed via {@link ClientOptions#optionsOverrides} and {@link
   * ClientOptions#rcSource}, which is basically a line-by-line transfer of the rc files read by the
   * client. This is not a particularly useful format for expanding the options, so this method
   * structures the list so that it is easier to find the arguments that apply to a command, or to
   * find the definitions of a config value.
   */
  @VisibleForTesting
  static ListMultimap<String, RcChunkOfArgs> structureRcOptionsAndConfigs(
      EventHandler eventHandler,
      List<String> rcFiles,
      List<ClientOptions.OptionOverride> rawOverrides,
      Set<String> validCommands) {
    ListMultimap<String, RcChunkOfArgs> commandToRcArgs = ArrayListMultimap.create();

    String lastRcFile = null;
    ListMultimap<String, String> commandToArgMapForLastRc = null;
    for (ClientOptions.OptionOverride override : rawOverrides) {
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
                "while reading option defaults file '"
                    + rcFile
                    + "':\n"
                    + "  invalid command name '"
                    + override.command
                    + "'."));
        continue;
      }

      // We've moved on to another rc file "chunk," store the accumulated args from the last one.
      if (!rcFile.equals(lastRcFile)) {
        if (lastRcFile != null) {
          // Go through the various commands identified in this rc file (or chunk of file) and
          // store them grouped first by command, then by rc chunk.
          for (String commandKey : commandToArgMapForLastRc.keySet()) {
            commandToRcArgs.put(
                commandKey,
                new RcChunkOfArgs(lastRcFile, commandToArgMapForLastRc.get(commandKey)));
          }
        }
        lastRcFile = rcFile;
        commandToArgMapForLastRc = ArrayListMultimap.create();
      }

      commandToArgMapForLastRc.put(override.command, override.option);
    }
    if (lastRcFile != null) {
      // Once again, for this last rc file chunk, store them grouped by command.
      for (String commandKey : commandToArgMapForLastRc.keySet()) {
        commandToRcArgs.put(
            commandKey, new RcChunkOfArgs(lastRcFile, commandToArgMapForLastRc.get(commandKey)));
      }
    }

    return commandToRcArgs;
  }
}
