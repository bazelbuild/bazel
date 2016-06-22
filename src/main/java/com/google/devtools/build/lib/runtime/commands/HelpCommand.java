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
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.collect.Iterables;
import com.google.common.escape.Escaper;
import com.google.common.html.HtmlEscapers;
import com.google.devtools.build.docgen.BlazeRuleHelpPrinter;
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
import com.google.devtools.build.lib.util.StringUtil;
import com.google.devtools.build.lib.util.io.OutErr;
import com.google.devtools.common.options.Converters;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsProvider;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
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

  /**
   * Only to be used to escape the internal hard-coded help texts when outputting HTML from help,
   * which don't pose a security risk.
   */
  private static final Escaper HTML_ESCAPER = HtmlEscapers.htmlEscaper();

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
  private static ImmutableMap<String, String> getOptionCategories(BlazeRuntime runtime) {
    ImmutableMap.Builder<String, String> optionCategoriesBuilder = ImmutableMap.builder();
    String name = runtime.getProductName();
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
      emitBlazeVersionInfo(outErr, runtime.getProductName());
      emitGenericHelp(outErr, runtime);
      return ExitCode.SUCCESS;
    }
    if (options.getResidue().size() != 1) {
      env.getReporter().handle(Event.error("You must specify exactly one command"));
      return ExitCode.COMMAND_LINE_ERROR;
    }
    String helpSubject = options.getResidue().get(0);
    if (helpSubject.equals("startup_options")) {
      emitBlazeVersionInfo(outErr, runtime.getProductName());
      emitStartupOptions(
          outErr, helpOptions.helpVerbosity, runtime, getOptionCategories(runtime));
      return ExitCode.SUCCESS;
    } else if (helpSubject.equals("target-syntax")) {
      emitBlazeVersionInfo(outErr, runtime.getProductName());
      emitTargetSyntaxHelp(outErr, getOptionCategories(runtime), runtime.getProductName());
      return ExitCode.SUCCESS;
    } else if (helpSubject.equals("info-keys")) {
      emitInfoKeysHelp(env, outErr);
      return ExitCode.SUCCESS;
    } else if (helpSubject.equals("completion")) {
      emitCompletionHelp(runtime, outErr);
      return ExitCode.SUCCESS;
    } else if (helpSubject.equals("everything-as-html")) {
      new HtmlEmitter(runtime).emit(outErr);
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
    emitBlazeVersionInfo(outErr, runtime.getProductName());
    outErr.printOut(BlazeCommandUtils.getUsage(
        command.getClass(),
        getOptionCategories(runtime),
        helpOptions.helpVerbosity,
        runtime.getBlazeModules(),
        runtime.getRuleClassProvider(),
        runtime.getProductName()));
    return ExitCode.SUCCESS;
  }

  private void emitBlazeVersionInfo(OutErr outErr, String productName) {
    String releaseInfo = BlazeVersionInfo.instance().getReleaseName();
    String line = String.format("[%s %s]", productName, releaseInfo);
    outErr.printOut(String.format("%80s\n", line));
  }

  private void emitStartupOptions(OutErr outErr, OptionsParser.HelpVerbosity helpVerbosity,
      BlazeRuntime runtime, ImmutableMap<String, String> optionCategories) {
    outErr.printOut(
        BlazeCommandUtils.expandHelpTopic("startup_options",
            "resource:startup_options.txt",
            getClass(),
            BlazeCommandUtils.getStartupOptions(runtime.getBlazeModules()),
            optionCategories,
            helpVerbosity,
            runtime.getProductName()));
  }

  private void emitCompletionHelp(BlazeRuntime runtime, OutErr outErr) {
    // First startup_options
    Iterable<BlazeModule> blazeModules = runtime.getBlazeModules();
    ConfiguredRuleClassProvider ruleClassProvider = runtime.getRuleClassProvider();
    Map<String, BlazeCommand> commandsByName = getSortedCommands(runtime);

    outErr.printOutLn("BAZEL_COMMAND_LIST=\"" + SPACE_JOINER.join(commandsByName.keySet()) + "\"");

    outErr.printOutLn("BAZEL_INFO_KEYS=\"");
    for (String name : InfoCommand.getHardwiredInfoItemNames(runtime.getProductName())) {
        outErr.printOutLn(name);
    }
    outErr.printOutLn("\"");

    outErr.printOutLn("BAZEL_STARTUP_OPTIONS=\"");
    Iterable<Class<? extends OptionsBase>> options =
        BlazeCommandUtils.getStartupOptions(blazeModules);
    outErr.printOut(OptionsParser.newOptionsParser(options).getOptionsCompletion());
    outErr.printOutLn("\"");

    for (Map.Entry<String, BlazeCommand> e : commandsByName.entrySet()) {
      BlazeCommand command = e.getValue();
      String varName = e.getKey().toUpperCase(Locale.US).replace('-', '_');
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

  private static Map<String, BlazeCommand> getSortedCommands(BlazeRuntime runtime) {
    return ImmutableSortedMap.copyOf(runtime.getCommandMap());
  }

  private void emitTargetSyntaxHelp(OutErr outErr, ImmutableMap<String, String> optionCategories,
      String productName) {
    outErr.printOut(BlazeCommandUtils.expandHelpTopic("target-syntax",
                                    "resource:target-syntax.txt",
                                    getClass(),
                                    ImmutableList.<Class<? extends OptionsBase>>of(),
                                    optionCategories,
                                    OptionsParser.HelpVerbosity.MEDIUM,
                                    productName));
  }

  private void emitInfoKeysHelp(CommandEnvironment env, OutErr outErr) {
    for (InfoItem item : InfoCommand.getInfoItemMap(env,
        OptionsParser.newOptionsParser(
            ImmutableList.<Class<? extends OptionsBase>>of())).values()) {
      outErr.printOut(String.format("%-23s %s\n", item.getName(), item.getDescription()));
    }
  }

  private void emitGenericHelp(OutErr outErr, BlazeRuntime runtime) {
    outErr.printOut(String.format("Usage: %s <command> <options> ...\n\n",
            runtime.getProductName()));
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
          replace("%{product}", runtime.getProductName());
      outErr.printOut(String.format("  %-19s %s\n", name, shortDescription));
    }

    outErr.printOut("\n");
    outErr.printOut("Getting more help:\n");
    outErr.printOut(String.format("  %s help <command>\n", runtime.getProductName()));
    outErr.printOut("                   Prints help and options for <command>.\n");
    outErr.printOut(String.format("  %s help startup_options\n", runtime.getProductName()));
    outErr.printOut(String.format("                   Options for the JVM hosting %s.\n",
        runtime.getProductName()));
    outErr.printOut(String.format("  %s help target-syntax\n", runtime.getProductName()));
    outErr.printOut("                   Explains the syntax for specifying targets.\n");
    outErr.printOut(String.format("  %s help info-keys\n", runtime.getProductName()));
    outErr.printOut("                   Displays a list of keys used by the info command.\n");
  }

  private static final class HtmlEmitter {
    private final BlazeRuntime runtime;
    private final ImmutableMap<String, String> optionCategories;

    private HtmlEmitter(BlazeRuntime runtime) {
      this.runtime = runtime;
      this.optionCategories = getOptionCategories(runtime);
    }

    private void emit(OutErr outErr) {
      Map<String, BlazeCommand> commandsByName = getSortedCommands(runtime);
      StringBuilder result = new StringBuilder();
      result.append("<h2>Commands</h2>\n");
      result.append("<table>\n");
      for (Map.Entry<String, BlazeCommand> e : commandsByName.entrySet()) {
        BlazeCommand command = e.getValue();
        Command annotation = command.getClass().getAnnotation(Command.class);
        if (annotation.hidden()) {
          continue;
        }
        String shortDescription = annotation.shortDescription().
            replace("%{product}", runtime.getProductName());

        result.append("<tr>\n");
        result.append(
            String.format(
                "  <td><a href=\"#%s\"><code>%s</code></a></td>\n", e.getKey(), e.getKey()));
        result.append("  <td>").append(HTML_ESCAPER.escape(shortDescription)).append("</td>\n");
        result.append("</tr>\n");
      }
      result.append("</table>\n");
      result.append("\n");

      result.append("<h2>Startup Options</h2>\n");
      appendOptionsHtml(result, BlazeCommandUtils.getStartupOptions(runtime.getBlazeModules()));
      result.append("\n");

      result.append("<h2><a name=\"common_options\">Options Common to all Commands</a></h2>\n");
      appendOptionsHtml(result, BlazeCommandUtils.getCommonOptions(runtime.getBlazeModules()));
      result.append("\n");

      for (Map.Entry<String, BlazeCommand> e : commandsByName.entrySet()) {
        result.append(
            String.format(
                "<h2><a name=\"%s\">%s Options</a></h2>\n", e.getKey(), capitalize(e.getKey())));
        BlazeCommand command = e.getValue();
        Command annotation = command.getClass().getAnnotation(Command.class);
        if (annotation.hidden()) {
          continue;
        }
        List<String> inheritedCmdNames = new ArrayList<>();
        for (Class<? extends BlazeCommand> base : annotation.inherits()) {
          String name = base.getAnnotation(Command.class).name();
          inheritedCmdNames.add(String.format("<a href=\"#%s\">%s</a>", name, name));
        }
        if (!inheritedCmdNames.isEmpty()) {
          result.append("<p>Inherits all options from ");
          result.append(StringUtil.joinEnglishList(inheritedCmdNames, "and"));
          result.append(".</p>\n\n");
        }
        Set<Class<? extends OptionsBase>> options = new HashSet<>();
        Collections.addAll(options, annotation.options());
        for (BlazeModule blazeModule : runtime.getBlazeModules()) {
          Iterables.addAll(options, blazeModule.getCommandOptions(annotation));
        }
        appendOptionsHtml(result, options);
        result.append("\n");
      }
      outErr.printOut(result.toString());
    }

    private void appendOptionsHtml(
        StringBuilder result, Iterable<Class<? extends OptionsBase>> optionsClasses) {
      OptionsParser parser = OptionsParser.newOptionsParser(optionsClasses);
      result.append(parser.describeOptionsHtml(optionCategories, HTML_ESCAPER)
          .replace("%{product}", runtime.getProductName()));
    }

    private static String capitalize(String s) {
      return s.substring(0, 1).toUpperCase(Locale.US) + s.substring(1);
    }
  }
}
