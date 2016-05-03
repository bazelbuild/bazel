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
package com.google.devtools.build.lib.runtime.commands;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.docgen.BlazeRuleHelpPrinter;
import com.google.devtools.build.lib.Constants;
import com.google.devtools.build.lib.analysis.BlazeVersionInfo;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.NoBuildEvent;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.runtime.BlazeCommand;
import com.google.devtools.build.lib.runtime.BlazeCommandUtils;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.util.io.OutErr;
import com.google.devtools.common.options.Converters;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsProvider;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * The 'blaze help' command, which prints all available commands as well as
 * specific help pages.
 */
@Command(name = "help",
         options = { HelpCommand.Options.class },
         allowResidue = true,
         mustRunInWorkspace = false,
         shortDescription = "Prints help for commands, or the index.",
         completion = "command|{startup_options,target-syntax,info-keys}",
         help = "resource:help.txt")
public final class HelpCommand implements BlazeCommand {
  private static final Joiner SPACE_JOINER = Joiner.on(" ");

  public static class Options extends OptionsBase {

    @Option(name = "help_verbosity",
            category = "help",
            defaultValue = "medium",
            converter = Converters.HelpVerbosityConverter.class,
            help = "Select the verbosity of the help command.")
    public OptionsParser.HelpVerbosity helpVerbosity;

    @Option(name = "long",
            abbrev = 'l',
            defaultValue = "null",
            category = "help",
            expansion = {"--help_verbosity", "long"},
            help = "Show full description of each option, instead of just its name.")
    public Void showLongFormOptions;

    @Option(name = "short",
            defaultValue = "null",
            category = "help",
            expansion = {"--help_verbosity", "short"},
            help = "Show only the names of the options, not their types or meanings.")
    public Void showShortFormOptions;
  }

  /**
   * Returns a map that maps option categories to descriptive help strings for categories that
   * are not part of the Bazel core.
   */
  private ImmutableMap<String, String> getOptionCategories(BlazeRuntime runtime) {
    ImmutableMap.Builder<String, String> optionCategoriesBuilder = ImmutableMap.builder();
    String name = Constants.PRODUCT_NAME;
    optionCategoriesBuilder
        .put("checking", String.format(
             "Checking options, which control %s's error checking and/or warnings", name))
        .put("coverage", String.format(
             "Options that affect how %s generates code coverage information", name))
        .put("experimental",
             "Experimental options, which control experimental (and potentially risky) features")
        .put("flags",
             "Flags options, for passing options to other tools")
        .put("help",
             "Help options")
        .put("host jvm startup", String.format(
            "Options that affect the startup of the %s server's JVM", name))
        .put("misc",
             "Miscellaneous options")
        .put("package loading",
             "Options that specify how to locate packages")
        .put("query", String.format(
            "Options affecting the '%s query' dependency query command", name))
        .put("run", String.format(
            "Options specific to '%s run'", name))
        .put("semantics",
             "Semantics options, which affect the build commands and/or output file contents")
        .put("server startup", String.format(
            "Startup options, which affect the startup of the %s server", name))
        .put("strategy", String.format(
            "Strategy options, which affect how %s will execute the build", name))
        .put("testing", String.format(
            "Options that affect how %s runs tests", name))
        .put("verbosity", String.format(
            "Verbosity options, which control what %s prints", name))
        .put("version",
             "Version options, for selecting which version of other tools will be used")
        .put("what",
             "Output selection options, for determining what to build/test");
    for (BlazeModule module : runtime.getBlazeModules()) {
      optionCategoriesBuilder.putAll(module.getOptionCategories());
    }
    return optionCategoriesBuilder.build();
  }

  @Override
  public void editOptions(CommandEnvironment env, OptionsParser optionsParser) {}

  @Override
  public ExitCode exec(CommandEnvironment env, OptionsProvider options) {
    env.getEventBus().post(new NoBuildEvent());

    BlazeRuntime runtime = env.getRuntime();
    OutErr outErr = env.getReporter().getOutErr();
    Options helpOptions = options.getOptions(Options.class);
    if (options.getResidue().isEmpty()) {
      emitBlazeVersionInfo(outErr);
      emitGenericHelp(runtime, outErr);
      return ExitCode.SUCCESS;
    }
    if (options.getResidue().size() != 1) {
      env.getReporter().handle(Event.error("You must specify exactly one command"));
      return ExitCode.COMMAND_LINE_ERROR;
    }
    String helpSubject = options.getResidue().get(0);
    if (helpSubject.equals("startup_options")) {
      emitBlazeVersionInfo(outErr);
      emitStartupOptions(outErr, helpOptions.helpVerbosity, runtime, getOptionCategories(runtime));
      return ExitCode.SUCCESS;
    } else if (helpSubject.equals("target-syntax")) {
      emitBlazeVersionInfo(outErr);
      emitTargetSyntaxHelp(outErr, getOptionCategories(runtime));
      return ExitCode.SUCCESS;
    } else if (helpSubject.equals("info-keys")) {
      emitInfoKeysHelp(env, outErr);
      return ExitCode.SUCCESS;
    } else if (helpSubject.equals("completion")) {
      emitCompletionHelp(runtime, outErr);
      return ExitCode.SUCCESS;
    }

    BlazeCommand command = runtime.getCommandMap().get(helpSubject);
    if (command == null) {
      ConfiguredRuleClassProvider provider = runtime.getRuleClassProvider();
      RuleClass ruleClass = provider.getRuleClassMap().get(helpSubject);
      if (ruleClass != null && ruleClass.isDocumented()) {
        // There is a rule with a corresponding name
        outErr.printOut(BlazeRuleHelpPrinter.getRuleDoc(helpSubject, provider));
        return ExitCode.SUCCESS;
      } else {
        env.getReporter().handle(Event.error(
            null, "'" + helpSubject + "' is neither a command nor a build rule"));
        return ExitCode.COMMAND_LINE_ERROR;
      }
    }
    emitBlazeVersionInfo(outErr);
    outErr.printOut(BlazeCommandUtils.getUsage(
        command.getClass(),
        getOptionCategories(runtime),
        helpOptions.helpVerbosity,
        runtime.getBlazeModules(),
        runtime.getRuleClassProvider()));
    return ExitCode.SUCCESS;
  }

  private void emitBlazeVersionInfo(OutErr outErr) {
    String releaseInfo = BlazeVersionInfo.instance().getReleaseName();
    String line = String.format("[%s %s]", Constants.PRODUCT_NAME, releaseInfo);
    outErr.printOut(String.format("%80s\n", line));
  }

  @SuppressWarnings("unchecked") // varargs generic array creation
  private void emitStartupOptions(OutErr outErr, OptionsParser.HelpVerbosity helpVerbosity,
      BlazeRuntime runtime, ImmutableMap<String, String> optionCategories) {
    outErr.printOut(
        BlazeCommandUtils.expandHelpTopic("startup_options",
            "resource:startup_options.txt",
            getClass(),
            BlazeCommandUtils.getStartupOptions(runtime.getBlazeModules()),
            optionCategories,
        helpVerbosity));
  }

  private void emitCompletionHelp(BlazeRuntime runtime, OutErr outErr) {
    // First startup_options
    Iterable<BlazeModule> blazeModules = runtime.getBlazeModules();
    ConfiguredRuleClassProvider ruleClassProvider = runtime.getRuleClassProvider();
    Map<String, BlazeCommand> commandsByName = runtime.getCommandMap();
    Set<String> commands = commandsByName.keySet();

    outErr.printOutLn("BAZEL_COMMAND_LIST=\"" + SPACE_JOINER.join(commands) + "\"");

    outErr.printOutLn("BAZEL_INFO_KEYS=\"");
    for (String name : InfoCommand.getHardwiredInfoItemNames(Constants.PRODUCT_NAME)) {
        outErr.printOutLn(name);
    }
    outErr.printOutLn("\"");

    outErr.printOutLn("BAZEL_STARTUP_OPTIONS=\"");
    Iterable<Class<? extends OptionsBase>> options =
        BlazeCommandUtils.getStartupOptions(blazeModules);
    outErr.printOut(OptionsParser.newOptionsParser(options).getOptionsCompletion());
    outErr.printOutLn("\"");

    for (String name : commands) {
      BlazeCommand command = commandsByName.get(name);
      String varName = name.toUpperCase().replace('-', '_');
      Command annotation = command.getClass().getAnnotation(Command.class);
      if (!annotation.completion().isEmpty()) {
        outErr.printOutLn("BAZEL_COMMAND_" + varName + "_ARGUMENT=\""
            + annotation.completion() + "\"");
      }
      options = BlazeCommandUtils.getOptions(command.getClass(), blazeModules, ruleClassProvider);
      outErr.printOutLn("BAZEL_COMMAND_" + varName + "_FLAGS=\"");
      outErr.printOut(OptionsParser.newOptionsParser(options).getOptionsCompletion());
      outErr.printOutLn("\"");
    }
  }

  private void emitTargetSyntaxHelp(OutErr outErr, ImmutableMap<String, String> optionCategories) {
    outErr.printOut(BlazeCommandUtils.expandHelpTopic("target-syntax",
                                    "resource:target-syntax.txt",
                                    getClass(),
                                    ImmutableList.<Class<? extends OptionsBase>>of(),
                                    optionCategories,
                                    OptionsParser.HelpVerbosity.MEDIUM));
  }

  private void emitInfoKeysHelp(CommandEnvironment env, OutErr outErr) {
    for (InfoItem item : InfoCommand.getInfoItemMap(env,
        OptionsParser.newOptionsParser(
            ImmutableList.<Class<? extends OptionsBase>>of())).values()) {
      outErr.printOut(String.format("%-23s %s\n", item.getName(), item.getDescription()));
    }
  }

  private void emitGenericHelp(BlazeRuntime runtime, OutErr outErr) {
    outErr.printOut(String.format("Usage: %s <command> <options> ...\n\n",
            Constants.PRODUCT_NAME));

    outErr.printOut("Available commands:\n");

    Map<String, BlazeCommand> commandsByName = runtime.getCommandMap();
    List<String> namesInOrder = new ArrayList<>(commandsByName.keySet());
    Collections.sort(namesInOrder);

    for (String name : namesInOrder) {
      BlazeCommand command = commandsByName.get(name);
      Command annotation = command.getClass().getAnnotation(Command.class);
      if (annotation.hidden()) {
        continue;
      }

      String shortDescription = annotation.shortDescription().
          replace("%{product}", Constants.PRODUCT_NAME);
      outErr.printOut(String.format("  %-19s %s\n", name, shortDescription));
    }

    outErr.printOut("\n");
    outErr.printOut("Getting more help:\n");
    outErr.printOut(String.format("  %s help <command>\n", Constants.PRODUCT_NAME));
    outErr.printOut("                   Prints help and options for <command>.\n");
    outErr.printOut(String.format("  %s help startup_options\n", Constants.PRODUCT_NAME));
    outErr.printOut(String.format("                   Options for the JVM hosting %s.\n",
            Constants.PRODUCT_NAME));
    outErr.printOut(String.format("  %s help target-syntax\n", Constants.PRODUCT_NAME));
    outErr.printOut("                   Explains the syntax for specifying targets.\n");
    outErr.printOut(String.format("  %s help info-keys\n", Constants.PRODUCT_NAME));
    outErr.printOut("                   Displays a list of keys used by the info command.\n");
  }
}
