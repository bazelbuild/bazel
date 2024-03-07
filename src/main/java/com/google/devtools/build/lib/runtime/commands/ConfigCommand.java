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

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.collect.ImmutableSortedMap.toImmutableSortedMap;
import static java.nio.charset.StandardCharsets.UTF_8;
import static java.util.Comparator.comparing;
import static java.util.stream.Collectors.joining;

import com.google.common.collect.HashBasedTable;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Ordering;
import com.google.common.collect.Sets;
import com.google.common.collect.Table;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.analysis.config.Fragment;
import com.google.devtools.build.lib.analysis.config.FragmentClassSet;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.FragmentRegistry;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.analysis.config.output.ConfigurationForOutput;
import com.google.devtools.build.lib.analysis.config.output.FragmentOptionsForOutput;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.runtime.BlazeCommand;
import com.google.devtools.build.lib.runtime.BlazeCommandResult;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.commands.ConfigCommand.ConfigOptions;
import com.google.devtools.build.lib.runtime.commands.ConfigCommandOutputFormatter.JsonOutputFormatter;
import com.google.devtools.build.lib.runtime.commands.ConfigCommandOutputFormatter.TextOutputFormatter;
import com.google.devtools.build.lib.server.FailureDetails;
import com.google.devtools.build.lib.server.FailureDetails.ConfigCommand.Code;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.skyframe.config.BuildConfigurationKey;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.skyframe.InMemoryMemoizingEvaluator;
import com.google.devtools.common.options.EnumConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParsingResult;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.util.Collection;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
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
  /** Defines the types of output this command can produce. */
  public enum OutputType {
    TEXT,
    JSON
  }

  /** Options for the "config" command. */
  public static class ConfigOptions extends OptionsBase {
    @Option(
        name = "dump_all",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
        effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
        help = "If set, dump all known configurations instead of just the ids.")
    public boolean dumpAll;

    /** Converter for --output. */
    public static class OutputTypeConverter extends EnumConverter<OutputType> {
      public OutputTypeConverter() {
        super(OutputType.class, "output type");
      }
    }

    @Option(
        name = "output",
        converter = OutputTypeConverter.class,
        defaultValue = "text",
        documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
        effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
        help = "Formats the output of displayed results. Can be one of: 'text', 'json'. ")
    public OutputType outputType;
  }

  /**
   * Data structure defining the difference between two {@link BuildConfigurationValue}s from the
   * point of this command's output.
   *
   * <p>See {@link ConfigurationForOutput} for further details.
   */
  protected static class ConfigurationDiffForOutput {
    final String configHash1;
    final String configHash2;
    final List<FragmentDiffForOutput> fragmentsDiff;

    ConfigurationDiffForOutput(
        String configHash1, String configHash2, List<FragmentDiffForOutput> fragmentsDiff) {
      this.configHash1 = configHash1;
      this.configHash2 = configHash2;
      this.fragmentsDiff = fragmentsDiff;
    }

    @Override
    public boolean equals(Object o) {
      if (o instanceof ConfigurationDiffForOutput) {
        ConfigurationDiffForOutput other = (ConfigurationDiffForOutput) o;
        return other.configHash1.equals(configHash1)
            && other.configHash2.equals(configHash2)
            && other.fragmentsDiff.equals(fragmentsDiff);
      }
      return false;
    }

    @Override
    public int hashCode() {
      return Objects.hash(configHash1, configHash2, fragmentsDiff);
    }
  }

  /**
   * Data structure defining the difference between two {@link BuildConfigurationValue}s for a given
   * {@link FragmentOptions}from the point of this command's output.
   *
   * <p>See {@link ConfigurationForOutput} for further details.
   */
  protected static class FragmentDiffForOutput {
    final String name;
    final Map<String, Pair<String, String>> optionsDiff;

    FragmentDiffForOutput(String name, Map<String, Pair<String, String>> optionsDiff) {
      this.name = name;
      this.optionsDiff = optionsDiff;
    }

    @Override
    public boolean equals(Object o) {
      if (o instanceof FragmentDiffForOutput) {
        FragmentDiffForOutput other = (FragmentDiffForOutput) o;
        return other.name.equals(name) && other.optionsDiff.equals(optionsDiff);
      }
      return false;
    }

    @Override
    public int hashCode() {
      return Objects.hash(name, optionsDiff);
    }
  }

  /**
   * Main entry point into the <code>blaze config</code> command.
   *
   * <p>Its purpose is to parse all options, figure out what variation of the command that implies,
   * run the right logic, and return the right exit code.
   */
  @Override
  public BlazeCommandResult exec(CommandEnvironment env, OptionsParsingResult options) {
    ImmutableSortedMap<BuildConfigurationKey, BuildConfigurationValue> configurations =
        findConfigurations(env);
    if (configurations.isEmpty()) {
      String message =
          "No configurations found. This can happen if the 'config' subcommand is used after "
              + "files, including their metadata, have changed since the last invocation of "
              + "another subcommand. Try running a 'build' or 'cquery' directly followed by "
              + "'config'.";
      env.getReporter().handle(Event.error(message));
      return createFailureResult(message, Code.CONFIGURATION_NOT_FOUND);
    }

    try (PrintWriter writer =
        new PrintWriter(
            new OutputStreamWriter(env.getReporter().getOutErr().getOutputStream(), UTF_8))) {

      ConfigOptions configCommandOptions = options.getOptions(ConfigOptions.class);
      ConfigCommandOutputFormatter outputFormatter =
          configCommandOptions.outputType == OutputType.TEXT
              ? new TextOutputFormatter(writer)
              : new JsonOutputFormatter(writer);
      ImmutableSortedMap<
              Class<? extends Fragment>, ImmutableSortedSet<Class<? extends FragmentOptions>>>
          fragmentDefs =
              getFragmentDefs(env.getRuntime().getRuleClassProvider().getFragmentRegistry());

      if (options.getResidue().isEmpty()) {
        if (configCommandOptions.dumpAll) {
          return reportAllConfigurations(outputFormatter, forOutput(configurations, fragmentDefs));
        } else {
          return reportConfigurationIds(outputFormatter, forOutput(configurations, fragmentDefs));
        }
      } else if (options.getResidue().size() == 1) {
        String configHash = options.getResidue().get(0);
        return reportSingleConfiguration(
            outputFormatter, env, forOutput(configurations, fragmentDefs), configHash);
      } else if (options.getResidue().size() == 2) {
        String configHash1 = options.getResidue().get(0);
        String configHash2 = options.getResidue().get(1);
        return reportConfigurationDiff(
            forOutput(configurations, fragmentDefs),
            configHash1,
            configHash2,
            outputFormatter,
            env);
      } else {
        String message = "Too many config ids.";
        env.getReporter().handle(Event.error(message));
        return createFailureResult(message, Code.TOO_MANY_CONFIG_IDS);
      }
    }
  }

  /**
   * Returns all {@link BuildConfigurationValue}s in Skyframe as a map from their {@link
   * BuildConfigurationKey} to instance.
   */
  private static ImmutableSortedMap<BuildConfigurationKey, BuildConfigurationValue>
      findConfigurations(CommandEnvironment env) {
    InMemoryMemoizingEvaluator evaluator =
        (InMemoryMemoizingEvaluator)
            env.getRuntime().getWorkspace().getSkyframeExecutor().getEvaluator();
    return evaluator.getDoneValues().entrySet().stream()
        .filter(e -> SkyFunctions.BUILD_CONFIGURATION.equals(e.getKey().functionName()))
        .collect(
            toImmutableSortedMap(
                comparing(e -> e.getOptions().checksum()),
                e -> (BuildConfigurationKey) e.getKey(),
                e -> (BuildConfigurationValue) e.getValue()));
  }

  /**
   * Returns the {@link Fragment}s and the {@link FragmentOptions} they require from Blaze's
   * runtime.
   *
   * <p>These are the fragments that Blaze "knows about", not necessarily the fragments in a {@link
   * BuildConfigurationValue}. Trimming, in particular, strips fragments out of actual
   * configurations. It's safe to assume untrimmed configuration have all fragments listed here.
   */
  private static ImmutableSortedMap<
          Class<? extends Fragment>, ImmutableSortedSet<Class<? extends FragmentOptions>>>
      getFragmentDefs(FragmentRegistry fragmentRegistry) {
    return fragmentRegistry.getAllFragments().stream()
        .collect(
            toImmutableSortedMap(
                FragmentClassSet.LEXICAL_FRAGMENT_SORTER,
                fragment -> fragment,
                fragment ->
                    ImmutableSortedSet.copyOf(
                        Comparator.comparing(Class::getName), Fragment.requiredOptions(fragment))));
  }

  /**
   * Converts {@link #findConfigurations}'s output into a list of {@link ConfigurationForOutput}
   * instances.
   */
  private static ImmutableSortedSet<ConfigurationForOutput> forOutput(
      ImmutableSortedMap<BuildConfigurationKey, BuildConfigurationValue> asSkyKeyMap,
      ImmutableSortedMap<
              Class<? extends Fragment>, ImmutableSortedSet<Class<? extends FragmentOptions>>>
          fragmentDefs) {
    ImmutableSortedSet.Builder<ConfigurationForOutput> ans =
        ImmutableSortedSet.orderedBy(comparing(e -> e.getConfigHash()));
    for (Map.Entry<BuildConfigurationKey, BuildConfigurationValue> entry : asSkyKeyMap.entrySet()) {
      BuildConfigurationKey key = entry.getKey();
      BuildConfigurationValue config = entry.getValue();
      ans.add(
          ConfigurationForOutput.getConfigurationForOutput(
              key, config.checksum(), config, fragmentDefs));
    }
    return ans.build();
  }

  /**
   * Returns the configuration matching a hash prefix.
   *
   * @param configurations collection of configurations to search
   * @param configPrefix prefix or exact value of the matching configuration's hash
   * @throws InvalidConfigurationException if not exactly one configuration matches
   */
  private static ConfigurationForOutput getConfiguration(
      Collection<ConfigurationForOutput> configurations, String configPrefix)
      throws InvalidConfigurationException {
    ImmutableList<ConfigurationForOutput> matches =
        configurations.stream()
            .filter(config -> doesConfigMatch(config, configPrefix))
            .collect(toImmutableList());
    if (matches.isEmpty()) {
      throw new InvalidConfigurationException(
          String.format("No configuration found with ID prefix %s", configPrefix));
    } else if (matches.size() > 1) {
      throw new InvalidConfigurationException(
          String.format(
              "Configuration identifier '%s' is ambiguous.\n"
                  + "'%s' is a prefix of multiple configurations:\n "
                  + matches.stream()
                      .map(ConfigurationForOutput::getConfigHash)
                      .collect(joining("\n "))
                  + "\n\n"
                  + "Use a sufficient prefix to uniquely identify one configuration.",
              configPrefix,
              configPrefix));
    }
    return Iterables.getOnlyElement(matches);
  }

  private static boolean doesConfigMatch(ConfigurationForOutput config, String configPrefix) {
    return config.getConfigHash().startsWith(configPrefix);
  }

  /**
   * Reports the result of <code>blaze config --dump_all</code> and returns the appropriate command
   * exit code.
   */
  private static BlazeCommandResult reportAllConfigurations(
      ConfigCommandOutputFormatter writer,
      ImmutableSortedSet<ConfigurationForOutput> configurations) {
    writer.writeConfigurations(configurations);
    return BlazeCommandResult.success();
  }

  /**
   * Reports the result of <code>blaze config</code> and returns the appropriate command exit code.
   */
  private static BlazeCommandResult reportConfigurationIds(
      ConfigCommandOutputFormatter writer,
      ImmutableSortedSet<ConfigurationForOutput> configurations) {
    writer.writeConfigurationIDs(configurations);
    return BlazeCommandResult.success();
  }

  /**
   * Reports the result of <code>blaze config <configHash></code> and returns the appropriate
   * command exit code.
   */
  private static BlazeCommandResult reportSingleConfiguration(
      ConfigCommandOutputFormatter writer,
      CommandEnvironment env,
      ImmutableSortedSet<ConfigurationForOutput> allConfigurations,
      String configHash) {
    env.getReporter().handle(Event.info(String.format("Displaying config with id %s", configHash)));
    try {
      writer.writeConfiguration(getConfiguration(allConfigurations, configHash));
      return BlazeCommandResult.success();
    } catch (InvalidConfigurationException e) {
      env.getReporter().handle(Event.error(e.getMessage()));
      return createFailureResult(e.getMessage(), Code.CONFIGURATION_NOT_FOUND);
    }
  }

  /**
   * Reports the result of <code>blaze config <configHash1> <configHash2></code> and returns the
   * appropriate command exit code.
   */
  private static BlazeCommandResult reportConfigurationDiff(
      ImmutableSortedSet<ConfigurationForOutput> allConfigs,
      String configHash1,
      String configHash2,
      ConfigCommandOutputFormatter writer,
      CommandEnvironment env) {
    env.getReporter()
        .handle(
            Event.info(
                String.format(
                    "Displaying diff between configs" + " %s and" + " %s",
                    configHash1, configHash2)));
    try {
      ConfigurationForOutput config1 = getConfiguration(allConfigs, configHash1);
      ConfigurationForOutput config2 = getConfiguration(allConfigs, configHash2);
      Table<String, String, Pair<Object, Object>> diffs = diffConfigurations(config1, config2);
      writer.writeConfigurationDiff(getConfigurationDiffForOutput(configHash1, configHash2, diffs));
      return BlazeCommandResult.success();
    } catch (InvalidConfigurationException e) {
      env.getReporter().handle(Event.error(e.getMessage()));
      return createFailureResult(e.getMessage(), Code.CONFIGURATION_NOT_FOUND);
    }
  }

  private static Table<String, String, Pair<Object, Object>> diffConfigurations(
      ConfigurationForOutput config1, ConfigurationForOutput config2) {
    Table<String, String, Pair<Object, Object>> diffs = HashBasedTable.create();

    for (String fragmentName :
        Sets.union(config1.fragmentOptionNames(), config2.fragmentOptionNames())) {
      FragmentOptionsForOutput options1 = config1.fragment(fragmentName);
      FragmentOptionsForOutput options2 = config2.fragment(fragmentName);
      diffs.row(fragmentName).putAll(diffOptions(options1, options2));
    }
    return diffs;
  }

  private static Map<String, Pair<Object, Object>> diffOptions(
      @Nullable FragmentOptionsForOutput options1, @Nullable FragmentOptionsForOutput options2) {
    Set<String> optionNames1 = options1 == null ? ImmutableSet.of() : options1.optionNames();
    Set<String> optionNames2 = options2 == null ? ImmutableSet.of() : options2.optionNames();
    Map<String, Pair<Object, Object>> diffs = new HashMap<>();

    for (String optionName : Sets.union(optionNames1, optionNames2)) {
      String value1 = options1 == null ? null : options1.getOption(optionName);
      String value2 = options2 == null ? null : options2.getOption(optionName);

      if (!Objects.equals(value1, value2)) {
        diffs.put(optionName, Pair.of(value1, value2));
      }
    }

    return diffs;
  }

  private static ConfigurationDiffForOutput getConfigurationDiffForOutput(
      String configHash1, String configHash2, Table<String, String, Pair<Object, Object>> diffs) {
    ImmutableSortedSet.Builder<FragmentDiffForOutput> fragmentDiffs =
        ImmutableSortedSet.orderedBy(comparing(e -> e.name));
    diffs
        .rowKeySet()
        .forEach(
            fragmentName -> {
              ImmutableSortedMap<String, Pair<String, String>> sortedOptionDiffs =
                  diffs.row(fragmentName).entrySet().stream()
                      .collect(
                          toImmutableSortedMap(
                              Ordering.natural(),
                              Map.Entry::getKey,
                              e -> toNullableStringPair(e.getValue())));
              fragmentDiffs.add(new FragmentDiffForOutput(fragmentName, sortedOptionDiffs));
            });
    return new ConfigurationDiffForOutput(configHash1, configHash2, fragmentDiffs.build().asList());
  }

  private static Pair<String, String> toNullableStringPair(Pair<Object, Object> pair) {
    return Pair.of(String.valueOf(pair.first), String.valueOf(pair.second));
  }

  private static BlazeCommandResult createFailureResult(String message, Code detailedCode) {
    return BlazeCommandResult.failureDetail(
        FailureDetail.newBuilder()
            .setMessage(message)
            .setConfigCommand(FailureDetails.ConfigCommand.newBuilder().setCode(detailedCode))
            .build());
  }
}
