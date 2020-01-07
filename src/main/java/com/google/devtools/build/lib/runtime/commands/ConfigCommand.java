// Copyright 2019 The Bazel Authors. All rights reserved.
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

import static com.google.common.collect.ImmutableMap.toImmutableMap;
import static java.nio.charset.StandardCharsets.UTF_8;
import static java.util.Comparator.comparing;
import static java.util.Map.Entry.comparingByKey;

import com.google.common.base.Functions;
import com.google.common.collect.HashBasedTable;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Sets;
import com.google.common.collect.Table;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.runtime.BlazeCommand;
import com.google.devtools.build.lib.runtime.BlazeCommandResult;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.commands.ConfigCommand.ConfigOptions;
import com.google.devtools.build.lib.skyframe.BuildConfigurationValue;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.skyframe.InMemoryMemoizingEvaluator;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDefinition;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingResult;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Collectors;
import javax.annotation.Nullable;

/** Handles the 'config' command on the Blaze command line. */
@Command(
    name = "config",
    builds = true,
    inherits = {BuildCommand.class},
    options = {ConfigOptions.class},
    usesConfigurationOptions = true,
    shortDescription = "Displays details of configurations.",
    allowResidue = true,
    completion = "string",
    hidden = true,
    help = "resource:config.txt")
public class ConfigCommand implements BlazeCommand {

  /** Options for the "config" command. */
  public static class ConfigOptions extends OptionsBase {
    @Option(
        name = "dump_all",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
        effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
        help = "If set, dump all known configurations instead of just the ids.")
    public boolean dumpAll;
  }

  @Override
  public void editOptions(OptionsParser optionsParser) {}

  @Override
  public BlazeCommandResult exec(CommandEnvironment env, OptionsParsingResult options) {
    ImmutableMap<String, BuildConfiguration> configurations = findConfigurations(env);

    try (PrintWriter writer =
        new PrintWriter(
            new OutputStreamWriter(env.getReporter().getOutErr().getOutputStream(), UTF_8))) {

      if (options.getResidue().isEmpty()) {
        if (options.getOptions(ConfigOptions.class).dumpAll) {
          return reportAllConfigurations(writer, env);
        } else {
          return reportConfigurationIds(writer, configurations.keySet());
        }
      }

      if (options.getResidue().size() == 1) {
        String configHash = options.getResidue().get(0);
        env.getReporter()
            .handle(Event.info(String.format("Displaying config with id %s", configHash)));

        BuildConfiguration config = configurations.get(configHash);
        if (config == null) {
          env.getReporter()
              .handle(Event.error(String.format("No configuration found with id: %s", configHash)));
          return BlazeCommandResult.exitCode(ExitCode.COMMAND_LINE_ERROR);
        }

        StringBuilder sb = new StringBuilder();
        config.describe(sb);
        writer.print(sb.toString());

        return BlazeCommandResult.exitCode(ExitCode.SUCCESS);
      } else if (options.getResidue().size() == 2) {
        String configHash1 = options.getResidue().get(0);
        String configHash2 = options.getResidue().get(1);
        env.getReporter()
            .handle(
                Event.info(
                    String.format(
                        "Displaying diff between configs" + " %s and" + " %s",
                        configHash1, configHash2)));

        BuildConfiguration config1 = configurations.get(configHash1);
        if (config1 == null) {
          env.getReporter()
              .handle(
                  Event.error(String.format("No configuration found with id: %s", configHash1)));
          return BlazeCommandResult.exitCode(ExitCode.COMMAND_LINE_ERROR);
        }
        BuildConfiguration config2 = configurations.get(configHash2);
        if (config2 == null) {
          env.getReporter()
              .handle(
                  Event.error(String.format("No configuration found with id: %s", configHash2)));
          return BlazeCommandResult.exitCode(ExitCode.COMMAND_LINE_ERROR);
        }

        writer.printf(
            "Displaying diff between configs" + " %s and" + " %s\n", configHash1, configHash2);
        Table<Class<? extends FragmentOptions>, String, Pair<Object, Object>> diffs =
            diffConfigurations(config1, config2);
        writer.print(describeConfigDiff(diffs));
        return BlazeCommandResult.exitCode(ExitCode.SUCCESS);
      } else {
        env.getReporter().handle(Event.error("Too many config ids."));
        return BlazeCommandResult.exitCode(ExitCode.COMMAND_LINE_ERROR);
      }
    }
  }

  private ImmutableMap<String, BuildConfiguration> findConfigurations(CommandEnvironment env) {
    InMemoryMemoizingEvaluator evaluator =
        (InMemoryMemoizingEvaluator)
            env.getRuntime().getWorkspace().getSkyframeExecutor().getEvaluatorForTesting();
    return evaluator.getDoneValues().entrySet().stream()
        .filter(e -> SkyFunctions.BUILD_CONFIGURATION.equals(e.getKey().functionName()))
        .map(Map.Entry::getValue)
        .map(v -> (BuildConfigurationValue) v)
        .map(BuildConfigurationValue::getConfiguration)
        .collect(
            toImmutableMap(
                BuildConfiguration::checksum, Functions.identity(), (config1, config2) -> config1));
  }

  private BlazeCommandResult reportAllConfigurations(PrintWriter writer, CommandEnvironment env) {
    InMemoryMemoizingEvaluator evaluator =
        (InMemoryMemoizingEvaluator)
            env.getRuntime().getWorkspace().getSkyframeExecutor().getEvaluatorForTesting();
    ImmutableMap<BuildConfigurationValue.Key, BuildConfigurationValue> configs =
        evaluator.getDoneValues().entrySet().stream()
            .filter(e -> SkyFunctions.BUILD_CONFIGURATION.equals(e.getKey().functionName()))
            .collect(
                toImmutableMap(
                    e -> (BuildConfigurationValue.Key) e.getKey(),
                    e -> (BuildConfigurationValue) e.getValue()));

    for (Map.Entry<BuildConfigurationValue.Key, BuildConfigurationValue> entry :
        configs.entrySet()) {
      writer.print("BuildConfigurationValue.Key: ");
      writer.println(entry.getKey().toString());

      writer.print("BuildConfigurationValue:\n");
      StringBuilder sb = new StringBuilder();
      entry.getValue().getConfiguration().describe(sb);
      writer.print(sb.toString());
    }
    return BlazeCommandResult.exitCode(ExitCode.SUCCESS);
  }

  private BlazeCommandResult reportConfigurationIds(
      PrintWriter writer, ImmutableSet<String> configurationIds) {
    writer.println("Available configurations:");
    writer.println(configurationIds.stream().collect(Collectors.joining("\n")));

    return BlazeCommandResult.exitCode(ExitCode.SUCCESS);
  }

  /**
   * Starlark options don't have configuration fragments. This is just to keep their output
   * consistent with native options, i.e. to include "user-defined" section in the output list.
   */
  private static class UserDefinedFragment extends FragmentOptions {
    // Intentionally empty: we read the actual options directly from BuildOptions.
  }

  private Table<Class<? extends FragmentOptions>, String, Pair<Object, Object>> diffConfigurations(
      BuildConfiguration config1, BuildConfiguration config2) {
    Table<Class<? extends FragmentOptions>, String, Pair<Object, Object>> diffs =
        HashBasedTable.create();

    for (Class<? extends FragmentOptions> fragment :
        Sets.union(
            config1.getOptions().getFragmentClasses(), config2.getOptions().getFragmentClasses())) {
      FragmentOptions options1 = config1.getOptions().get(fragment);
      FragmentOptions options2 = config2.getOptions().get(fragment);
      diffs.row(fragment).putAll(diffOptions(fragment, options1, options2));
    }

    diffs.row(UserDefinedFragment.class).putAll(diffStarlarkOptions(config1, config2));
    return diffs;
  }

  private static Map<String, Pair<Object, Object>> diffOptions(
      Class<? extends FragmentOptions> fragment,
      @Nullable FragmentOptions options1,
      @Nullable FragmentOptions options2) {
    Map<String, Pair<Object, Object>> diffs = new HashMap<>();

    for (OptionDefinition option : OptionsParser.getOptionDefinitions(fragment)) {
      Object value1 = options1 == null ? null : options1.getValueFromDefinition(option);
      Object value2 = options2 == null ? null : options2.getValueFromDefinition(option);

      if (!Objects.equals(value1, value2)) {
        diffs.put(option.getOptionName(), Pair.of(value1, value2));
      }
    }

    return diffs;
  }

  private static Map<String, Pair<Object, Object>> diffStarlarkOptions(
      BuildConfiguration config1, BuildConfiguration config2) {
    Map<Label, Object> starlarkOptions1 = config1.getOptions().getStarlarkOptions();
    Map<Label, Object> starlarkOptions2 = config2.getOptions().getStarlarkOptions();
    Map<String, Pair<Object, Object>> diffs = new HashMap<>();
    for (Label option : Sets.union(starlarkOptions1.keySet(), starlarkOptions2.keySet())) {
      Object value1 = starlarkOptions1.get(option);
      Object value2 = starlarkOptions2.get(option);
      if (!Objects.equals(value1, value2)) {
        diffs.put(option.toString(), Pair.of(value1, value2));
      }
    }
    return diffs;
  }

  private static String describeConfigDiff(
      Table<Class<? extends FragmentOptions>, String, Pair<Object, Object>> diff) {
    StringBuilder sb = new StringBuilder();

    diff.rowKeySet().stream()
        .sorted(comparing(Class::getName))
        .forEach(fragmentClass -> displayFragmentDiff(fragmentClass, diff.row(fragmentClass), sb));

    return sb.toString();
  }

  private static void displayFragmentDiff(
      Class<? extends FragmentOptions> fragmentClass,
      Map<String, Pair<Object, Object>> diff,
      StringBuilder sb) {
    sb.append("Fragment ").append(fragmentClass.getName()).append(" {\n");
    diff.entrySet().stream()
        .sorted(comparingByKey())
        .forEach(
            e ->
                sb.append("  ")
                    .append(e.getKey())
                    .append(": ")
                    .append(e.getValue().getFirst())
                    .append(", ")
                    .append(e.getValue().getSecond())
                    .append("\n"));
    sb.append("}\n");
  }
}
