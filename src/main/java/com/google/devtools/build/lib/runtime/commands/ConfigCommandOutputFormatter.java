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

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Streams;
import com.google.devtools.build.lib.analysis.config.output.ConfigurationForOutput;
import com.google.devtools.build.lib.analysis.config.output.FragmentForOutput;
import com.google.devtools.build.lib.analysis.config.output.FragmentOptionsForOutput;
import com.google.devtools.build.lib.runtime.commands.ConfigCommand.ConfigurationDiffForOutput;
import com.google.devtools.build.lib.runtime.commands.ConfigCommand.FragmentDiffForOutput;
import com.google.devtools.build.lib.util.Pair;
import com.google.gson.Gson;
import java.io.PrintWriter;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Formats output for {@link ConfigCommand}.
 *
 * <p>The basic contract is @link ConfigCommand} makes all important structural decisions: what data
 * gets reported, how different pieces of data relate to each other, and how data is ordered. A
 * {@link ConfigCommandOutputFormatter} then outputs this in a format-appropriate way.
 */
abstract class ConfigCommandOutputFormatter {
  protected final PrintWriter writer;

  /** Constructs a formatter that writes output to the given {@link PrintWriter}. */
  ConfigCommandOutputFormatter(PrintWriter writer) {
    this.writer = writer;
  }

  /** Outputs a list of configuration hash IDs. */
  public abstract void writeConfigurationIDs(Iterable<ConfigurationForOutput> configurations);

  /** Outputs a single configuration. */
  public abstract void writeConfiguration(ConfigurationForOutput configuration);

  /** Outputs a series of configurations. */
  public abstract void writeConfigurations(Iterable<ConfigurationForOutput> configurations);

  /** Outputs the diff between two configurations */
  public abstract void writeConfigurationDiff(ConfigurationDiffForOutput diff);

  /** A {@link ConfigCommandOutputFormatter} that outputs plan user-readable text. */
  static class TextOutputFormatter extends ConfigCommandOutputFormatter {
    TextOutputFormatter(PrintWriter writer) {
      super(writer);
    }

    @Override
    public void writeConfigurationIDs(Iterable<ConfigurationForOutput> configurations) {
      writer.println("Available configurations:");
      configurations.forEach(
          config ->
              writer.printf(
                  "%s %s%s%n",
                  config.getConfigHash(),
                  config.getMnemonic(),
                  (config.isExec() ? " (exec)" : "")));
    }

    @Override
    public void writeConfiguration(ConfigurationForOutput configuration) {
      writer.println("BuildConfigurationValue " + configuration.getConfigHash() + ":");
      writer.println("Skyframe Key: " + configuration.getSkyKey());

      StringBuilder fragments = new StringBuilder();
      for (FragmentForOutput fragment : configuration.getFragments()) {
        fragments
            .append(fragment.getName())
            .append(": [")
            .append(String.join(",", fragment.getFragmentOptions()))
            .append("], ");
      }

      writer.println("Fragments: " + fragments);
      for (FragmentOptionsForOutput fragment : configuration.getFragmentOptions()) {
        writer.println("FragmentOptions " + fragment.getName() + " {");
        for (Map.Entry<String, String> optionSetting : fragment.getOptions().entrySet()) {
          writer.printf("  %s: %s\n", optionSetting.getKey(), optionSetting.getValue());
        }
        writer.println("}");
      }
    }

    @Override
    public void writeConfigurations(Iterable<ConfigurationForOutput> configurations) {
      for (ConfigurationForOutput config : configurations) {
        writeConfiguration(config);
      }
    }

    @Override
    public void writeConfigurationDiff(ConfigurationDiffForOutput diff) {
      writer.printf(
          "Displaying diff between configs %s and %s\n", diff.configHash1, diff.configHash2);
      for (FragmentDiffForOutput fragmentDiff : diff.fragmentsDiff) {
        writer.println("FragmentOptions " + fragmentDiff.name + " {");
        for (Map.Entry<String, Pair<String, String>> optionDiff :
            fragmentDiff.optionsDiff.entrySet()) {
          writer.printf(
              "  %s: %s, %s\n",
              optionDiff.getKey(), optionDiff.getValue().first, optionDiff.getValue().second);
        }
        writer.println("}");
      }
    }
  }

  /** A {@link ConfigCommandOutputFormatter} that outputs structured JSON. */
  static class JsonOutputFormatter extends ConfigCommandOutputFormatter {
    private final Gson gson;

    JsonOutputFormatter(PrintWriter writer) {
      super(writer);
      this.gson = new Gson();
    }

    @Override
    public void writeConfigurationIDs(Iterable<ConfigurationForOutput> configurations) {
      Iterable<String> configurationIDs =
          Streams.stream(configurations)
              .map(config -> config.getConfigHash())
              .collect(Collectors.toList());
      writer.println(gson.toJson(ImmutableMap.of("configuration-IDs", configurationIDs)));
    }

    @Override
    public void writeConfiguration(ConfigurationForOutput configuration) {
      writer.println(gson.toJson(configuration));
    }

    @Override
    public void writeConfigurations(Iterable<ConfigurationForOutput> configurations) {
      writer.println(gson.toJson(configurations));
    }

    @Override
    public void writeConfigurationDiff(ConfigurationDiffForOutput diff) {
      writer.println(gson.toJson(diff));
    }
  }
}
