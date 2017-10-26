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
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ListMultimap;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.runtime.commands.ProjectFileSupport;
import com.google.devtools.build.lib.runtime.proto.InvocationPolicyOuterClass.InvocationPolicy;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.common.options.InvocationPolicyEnforcer;
import com.google.devtools.common.options.OptionDefinition;
import com.google.devtools.common.options.OptionPriority.PriorityCategory;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;
import com.google.devtools.common.options.OptionsProvider;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.function.Function;
import java.util.logging.Level;
import javax.annotation.Nullable;

/**
 * Handles parsing the blaze command arguments.
 *
 * <p>This class manages rc options, default options, and invocation policy.
 */
public class BlazeOptionHandler {
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
  private final String commandName;
  private final InvocationPolicy invocationPolicy;
  private final List<String> rcfileNotes = new ArrayList<>();

  public BlazeOptionHandler(
      BlazeRuntime runtime,
      BlazeWorkspace workspace,
      BlazeCommand command,
      Command commandAnnotation,
      String commandName,
      OptionsParser optionsParser,
      InvocationPolicy invocationPolicy) {
    this.runtime = runtime;
    this.workspace = workspace;
    this.command = command;
    this.commandAnnotation = commandAnnotation;
    this.commandName = commandName;
    this.optionsParser = optionsParser;
    this.invocationPolicy = invocationPolicy;
  }

  // Return options as OptionsProvider so the options can't be easily modified after we've
  // applied the invocation policy.
  OptionsProvider getOptionsResult() {
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
              "The '" + commandName + "' command is only supported from within a workspace."));
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
    return ExitCode.SUCCESS;
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

    // Command-specific options from .blazerc passed in via --default_override
    // and --rc_source. A no-op if none are provided.
    ClientOptions rcFileOptions = optionsParser.getOptions(ClientOptions.class);
    List<Pair<String, ListMultimap<String, String>>> optionsMap =
        getOptionsMap(
            eventHandler,
            rcFileOptions.rcSource,
            rcFileOptions.optionsOverrides,
            runtime.getCommandMap().keySet());

    parseOptionsForCommand(rcfileNotes, commandAnnotation, optionsParser, optionsMap, null, null);
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

    // Fix-point iteration until all configs are loaded.
    List<String> configsLoaded = ImmutableList.of();
    Set<String> unknownConfigs = new LinkedHashSet<>();
    CommonCommandOptions commonOptions = optionsParser.getOptions(CommonCommandOptions.class);
    while (!commonOptions.configs.equals(configsLoaded)) {
      Set<String> missingConfigs = new LinkedHashSet<>(commonOptions.configs);
      missingConfigs.removeAll(configsLoaded);
      parseOptionsForCommand(
          rcfileNotes,
          commandAnnotation,
          optionsParser,
          optionsMap,
          missingConfigs,
          unknownConfigs);
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
  protected static void parseOptionsForCommand(
      List<String> rcfileNotes,
      Command commandAnnotation,
      OptionsParser optionsParser,
      List<Pair<String, ListMultimap<String, String>>> optionsMap,
      @Nullable Collection<String> configs,
      @Nullable Collection<String> unknownConfigs)
      throws OptionsParsingException {
    Set<String> knownConfigs = new HashSet<>();
    for (String commandToParse : getCommandNamesToParse(commandAnnotation)) {
      for (Pair<String, ListMultimap<String, String>> entry : optionsMap) {
        String rcFile = entry.first;
        List<String> allOptions = new ArrayList<>();
        if (configs == null) {
          Collection<String> values = entry.second.get(commandToParse);
          if (!values.isEmpty()) {
            allOptions.addAll(entry.second.get(commandToParse));
            String inherited = commandToParse.equals(commandAnnotation.name()) ? "" : "Inherited ";
            String source =
                rcFile.equals("client")
                    ? "Options provided by the client"
                    : String.format(
                        "Reading rc options for '%s' from %s", commandAnnotation.name(), rcFile);
            rcfileNotes.add(
                String.format(
                    "%s:\n  %s'%s' options: %s",
                    source, inherited, commandToParse, Joiner.on(' ').join(values)));
          }
        } else {
          for (String config : configs) {
            String configDef = commandToParse + ":" + config;
            Collection<String> values = entry.second.get(configDef);
            if (!values.isEmpty()) {
              allOptions.addAll(values);
              knownConfigs.add(config);
              rcfileNotes.add(
                  String.format(
                      "Found applicable config definition %s in file %s: %s",
                      configDef, rcFile, String.join(" ", values)));
            }
          }
        }
        processOptionList(optionsParser, rcFile, allOptions);
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
  private static void processOptionList(
      OptionsParser optionsParser, String rcfile, List<String> rcfileOptions)
      throws OptionsParsingException {
    if (!rcfileOptions.isEmpty()) {
      optionsParser.parse(PriorityCategory.RC_FILE, rcfile, rcfileOptions);
    }
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
   * Convert a list of option override specifications to a more easily digestible form.
   *
   * @param overrides list of option override specifications
   */
  @VisibleForTesting
  static List<Pair<String, ListMultimap<String, String>>> getOptionsMap(
      EventHandler eventHandler,
      List<String> rcFiles,
      List<ClientOptions.OptionOverride> overrides,
      Set<String> validCommands) {
    List<Pair<String, ListMultimap<String, String>>> result = new ArrayList<>();

    String lastRcFile = null;
    ListMultimap<String, String> lastMap = null;
    for (ClientOptions.OptionOverride override : overrides) {
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
}
