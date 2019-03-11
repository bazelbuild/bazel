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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.packages.StarlarkSemanticsOptions;
import com.google.devtools.build.lib.util.ResourceFileLoader;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import java.io.IOException;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.Set;

/**
 * Utility class for functionality related to Blaze commands.
 */
public class BlazeCommandUtils {
  /**
   * Options classes used as startup options in Blaze core.
   */
  private static final ImmutableList<Class<? extends OptionsBase>> DEFAULT_STARTUP_OPTIONS =
      ImmutableList.of(
          BlazeServerStartupOptions.class,
          HostJvmStartupOptions.class);

  /** The set of option-classes that are common to all Blaze commands. */
  private static final ImmutableList<Class<? extends OptionsBase>> COMMON_COMMAND_OPTIONS =
      ImmutableList.of(
          BlazeCommandEventHandler.Options.class,
          CommonCommandOptions.class,
          ClientOptions.class,
          // Skylark options aren't applicable to all commands, but making them a common option
          // allows users to put them in the common section of the bazelrc. See issue #3538.
          StarlarkSemanticsOptions.class);

  private BlazeCommandUtils() {}

  public static ImmutableList<Class<? extends OptionsBase>> getStartupOptions(
      Iterable<BlazeModule> modules) {
    Set<Class<? extends OptionsBase>> options = new HashSet<>();
       options.addAll(DEFAULT_STARTUP_OPTIONS);
    for (BlazeModule blazeModule : modules) {
      Iterables.addAll(options, blazeModule.getStartupOptions());
    }

    return ImmutableList.copyOf(options);
  }

  public static ImmutableSet<Class<? extends OptionsBase>> getCommonOptions(
      Iterable<BlazeModule> modules) {
    ImmutableSet.Builder<Class<? extends OptionsBase>> builder = ImmutableSet.builder();
    builder.addAll(COMMON_COMMAND_OPTIONS);
    for (BlazeModule blazeModule : modules) {
      builder.addAll(blazeModule.getCommonCommandOptions());
    }
    return builder.build();
  }

  /**
   * Returns the set of all options (including those inherited directly and
   * transitively) for this AbstractCommand's @Command annotation.
   *
   * <p>Why does metaprogramming always seem like such a bright idea in the
   * beginning?
   */
  public static ImmutableList<Class<? extends OptionsBase>> getOptions(
      Class<? extends BlazeCommand> clazz,
      Iterable<BlazeModule> modules,
      ConfiguredRuleClassProvider ruleClassProvider) {
    Command commandAnnotation = clazz.getAnnotation(Command.class);
    if (commandAnnotation == null) {
      throw new IllegalStateException("@Command missing for " + clazz.getName());
    }

    Set<Class<? extends OptionsBase>> options = new HashSet<>();
    options.addAll(getCommonOptions(modules));
    Collections.addAll(options, commandAnnotation.options());

    if (commandAnnotation.usesConfigurationOptions()) {
      options.addAll(ruleClassProvider.getConfigurationOptions());
    }

    for (BlazeModule blazeModule : modules) {
      Iterables.addAll(options, blazeModule.getCommandOptions(commandAnnotation));
    }

    for (Class<? extends BlazeCommand> base : commandAnnotation.inherits()) {
      options.addAll(getOptions(base, modules, ruleClassProvider));
    }
    return ImmutableList.copyOf(options);
  }

  /**
   * Returns the expansion of the specified help topic.
   *
   * @param topic the name of the help topic; used in %{command} expansion.
   * @param help the text template of the help message. Certain %{x} variables will be expanded. A
   *     prefix of "resource:" means use the .jar resource of that name.
   * @param helpVerbosity a tri-state verbosity option selecting between just names, names and
   *     syntax, and full description.
   * @param productName the product name
   */
  public static final String expandHelpTopic(
      String topic,
      String help,
      Class<? extends BlazeCommand> commandClass,
      Collection<Class<? extends OptionsBase>> options,
      OptionsParser.HelpVerbosity helpVerbosity,
      String productName) {
    OptionsParser parser = OptionsParser.newOptionsParser(options);

    String template;
    if (help.startsWith("resource:")) {
      String resourceName = help.substring("resource:".length());
      try {
        template = ResourceFileLoader.loadResource(commandClass, resourceName);
      } catch (IOException e) {
        throw new IllegalStateException(
            "failed to load help resource '"
                + resourceName
                + "' due to I/O error: "
                + e.getMessage(),
            e);
      }
    } else {
      template = help;
    }

    if (!template.contains("%{options}")) {
      throw new IllegalStateException("Help template for '" + topic + "' omits %{options}!");
    }

    String optionStr;
      optionStr =
          parser.describeOptions(productName, helpVerbosity).replace("%{product}", productName);

    return template
            .replace("%{product}", productName)
            .replace("%{command}", topic)
            .replace("%{options}", optionStr)
            .trim()
        + "\n\n"
        + (helpVerbosity == OptionsParser.HelpVerbosity.MEDIUM
            ? "(Use 'help --long' for full details or --short to just enumerate options.)\n"
            : "");
  }

  /**
   * The help page for this command.
   *
   * @param verbosity a tri-state verbosity option selecting between just names, names and syntax,
   *     and full description.
   */
  public static String getUsage(
      Class<? extends BlazeCommand> commandClass,
      OptionsParser.HelpVerbosity verbosity,
      Iterable<BlazeModule> blazeModules,
      ConfiguredRuleClassProvider ruleClassProvider,
      String productName) {
    Command commandAnnotation = commandClass.getAnnotation(Command.class);
    return BlazeCommandUtils.expandHelpTopic(
        commandAnnotation.name(),
        commandAnnotation.help(),
        commandClass,
        BlazeCommandUtils.getOptions(commandClass, blazeModules, ruleClassProvider),
        verbosity,
        productName);
  }
}
