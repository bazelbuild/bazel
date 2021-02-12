// Copyright 2020 The Bazel Authors. All rights reserved.
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

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth8.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import com.google.devtools.build.lib.runtime.BlazeCommandDispatcher;
import com.google.devtools.build.lib.runtime.commands.ConfigCommand.ConfigurationDiffForOutput;
import com.google.devtools.build.lib.runtime.commands.ConfigCommand.ConfigurationForOutput;
import com.google.devtools.build.lib.runtime.commands.ConfigCommand.FragmentDiffForOutput;
import com.google.devtools.build.lib.testutil.Suite;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.testutil.TestSpec;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.io.RecordingOutErr;
import com.google.gson.Gson;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Optional;
import java.util.stream.Collectors;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link ConfigCommand} ("<code>$ blaze config</code>").
 *
 * <p>These tests assume all important testable properties are determined in {@link ConfigCommand},
 * so the output formatter used doesn't affect those properties. We test with <code>--output=json
 * </code> for easy parsing.
 */
@TestSpec(size = Suite.MEDIUM_TESTS)
@RunWith(JUnit4.class)
public class ConfigCommandTest extends BuildIntegrationTestCase {
  private BlazeCommandDispatcher dispatcher;

  @Before
  public final void init() throws Exception {
    getRuntime().overrideCommands(ImmutableList.of(new BuildCommand(), new ConfigCommand()));
    dispatcher = new BlazeCommandDispatcher(getRuntime());
    write(
        "test/defs.bzl",
        "def _simple_rule_impl(ctx):",
        "  pass",
        "simple_rule = rule(",
        "  implementation = _simple_rule_impl,",
        "  attrs = {},",
        ")");
    write("test/BUILD", "load('//test:defs.bzl', 'simple_rule')", "simple_rule(name='buildme')");
  }

  /**
   * Performs loading and analysis on the fixed rule <code>//test:buildme</code> with the given
   * build options (as they'd appear on the command line).
   */
  private void analyzeTarget(String... args) throws Exception {
    List<String> params = Lists.newArrayList("build");
    // Basic flags required to make any build work. Ideally we'd go through BlazeRuntimeWrapper,
    // which does the same setup. But that's explicitly documented as not supported command
    // invocations, which is exactly what we we need here.
    params.addAll(TestConstants.PRODUCT_SPECIFIC_FLAGS);
    params.add("//test:buildme");
    params.add("--nobuild"); // Execution phase isn't necessary to collect configurations.
    Collections.addAll(params, args);
    dispatcher.exec(params, "my client", outErr);
  }

  /**
   * Calls <cod>blaze config --output=json</code> with the given flags and returns the raw output.
   *
   * <p>Should be called after {@link #analyzeTarget} so there are actual configs to read.
   */
  private RecordingOutErr callConfigCommand(String... args) throws Exception {
    List<String> params = Lists.newArrayList("config");
    params.add("--output=json");
    Collections.addAll(params, args);
    RecordingOutErr recordingOutErr = new RecordingOutErr();
    dispatcher.exec(params, "my client", recordingOutErr);
    return recordingOutErr;
  }

  /**
   * Returns the value of an option under a configuration's {@link FragmentOptions}.
   *
   * <p>Throws {@link NoSuchElementException} if it can't be found.
   */
  private static String getOptionValue(
      ConfigurationForOutput config, String fragmentOptions, String optionName) {
    List<String> ans =
        config.fragmentOptions.stream()
            .filter(fragment -> fragment.name.endsWith(fragmentOptions))
            .flatMap(fragment -> fragment.options.entrySet().stream())
            .filter(setting -> setting.getKey().equals(optionName))
            .map(entry -> entry.getValue())
            .collect(Collectors.toList());
    if (ans.size() > 1) {
      throw new NoSuchElementException(
          String.format("Multple matches for fragment=%s, option=%s", fragmentOptions, optionName));
    } else if (ans.isEmpty()) {
      throw new NoSuchElementException(
          String.format("No matches for fragment=%s, option=%s", fragmentOptions, optionName));
    }
    return ans.get(0);
  }

  private static boolean isTargetConfig(ConfigurationForOutput config) {
    return !Boolean.parseBoolean(getOptionValue(config, "CoreOptions", "is host configuration"))
        && !Boolean.parseBoolean(getOptionValue(config, "CoreOptions", "is exec configuration"));
  }

  /** Converts {@code a.b.d} to {@code d}. * */
  private static String getBaseName(String str) {
    return str.substring(str.lastIndexOf(".") + 1);
  }

  /** Converts a list of {@code a.b.d} strings to {@code d} form. * */
  private static List<String> getBaseNames(List<String> list) {
    return list.stream().map(entry -> getBaseName(entry)).collect(Collectors.toList());
  }

  @Test
  public void showConfigIds() throws Exception {
    analyzeTarget();
    JsonObject fullJson =
        JsonParser.parseString(callConfigCommand().outAsLatin1()).getAsJsonObject();
    // Should be one ID for the target configuration and one for the host.
    assertThat(fullJson).isNotNull();
    assertThat(fullJson.has("configuration-IDs")).isTrue();
    assertThat(fullJson.get("configuration-IDs").getAsJsonArray().size()).isEqualTo(2);
  }

  @Test
  public void showSingleConfig() throws Exception {
    analyzeTarget();
    String configHash1 =
        JsonParser.parseString(callConfigCommand().outAsLatin1())
            .getAsJsonObject()
            .get("configuration-IDs")
            .getAsJsonArray()
            .get(0)
            .getAsString();
    ConfigurationForOutput config =
        new Gson()
            .fromJson(callConfigCommand(configHash1).outAsLatin1(), ConfigurationForOutput.class);
    assertThat(config).isNotNull();
    // Verify config metadata:
    assertThat(config.configHash).isEqualTo(configHash1);
    assertThat(config.skyKey)
        .isEqualTo(String.format("BuildConfigurationValue.Key[%s]", configHash1));
    // Verify the existence of a couple of expected fragments:
    assertThat(
            config.fragments.stream()
                .map(
                    fragment ->
                        Pair.of(getBaseName(fragment.name), getBaseNames(fragment.fragmentOptions)))
                .collect(Collectors.toList()))
        .containsAtLeast(
            Pair.of("PlatformConfiguration", ImmutableList.of("PlatformOptions")),
            Pair.of("TestConfiguration", ImmutableList.of("TestConfiguration$TestOptions")));
    // Verify the existence of a couple of expected fragment options:
    assertThat(
            config.fragmentOptions.stream()
                .map(fragment -> getBaseName(fragment.name))
                .collect(Collectors.toList()))
        .containsAtLeast("PlatformOptions", "CoreOptions", "user-defined");
    // Verify the existence of a couple of expected option names:
    assertThat(
            config.fragmentOptions.stream()
                .filter(fragment -> fragment.name.endsWith("CoreOptions"))
                .flatMap(fragment -> fragment.options.keySet().stream())
                .collect(Collectors.toList()))
        .containsAtLeast("run_under", "check_visibility", "stamp");
  }

  @Test
  public void showSingleConfigHashPrefix() throws Exception {
    analyzeTarget();
    String configHash =
        JsonParser.parseString(callConfigCommand().outAsLatin1())
            .getAsJsonObject()
            .get("configuration-IDs")
            .getAsJsonArray()
            .get(0)
            .getAsString();
    String hashPrefix = configHash.substring(0, configHash.length() / 2);
    ConfigurationForOutput config =
        new Gson()
            .fromJson(callConfigCommand(hashPrefix).outAsLatin1(), ConfigurationForOutput.class);
    assertThat(config).isNotNull();
    assertThat(config.configHash).startsWith(hashPrefix);
  }

  @Test
  public void showSingleConfig_hostConfig() throws Exception {
    analyzeTarget();
    ConfigurationForOutput config =
        new Gson().fromJson(callConfigCommand("host").outAsLatin1(), ConfigurationForOutput.class);
    assertThat(config).isNotNull();
    assertThat(config.isHost).isTrue();
  }

  @Test
  public void unknownHashPrefix() throws Exception {
    analyzeTarget();
    String configHash =
        JsonParser.parseString(callConfigCommand().outAsLatin1())
            .getAsJsonObject()
            .get("configuration-IDs")
            .getAsJsonArray()
            .get(0)
            .getAsString();
    // No valid hash has spaces.
    String hashPrefix = configHash.substring(0, configHash.length() / 2) + " ";
    assertThat(callConfigCommand(hashPrefix).errAsLatin1())
        .contains("No configuration found with ID prefix " + hashPrefix);
  }

  @Test
  public void showAllConfigs() throws Exception {
    analyzeTarget();

    int numConfigs = 0;
    for (JsonElement configJson :
        JsonParser.parseString(callConfigCommand("--dump_all").outAsLatin1()).getAsJsonArray()) {
      ConfigurationForOutput config = new Gson().fromJson(configJson, ConfigurationForOutput.class);
      assertThat(config).isNotNull();
      numConfigs++;
    }
    assertThat(numConfigs).isEqualTo(2); // Host + target.
  }

  @Test
  public void compareConfigs() throws Exception {
    analyzeTarget("--action_env=a=1");
    analyzeTarget("--action_env=b=2");
    String targetConfig1Hash = null;
    String targetConfig2Hash = null;

    // Find the two target configuration hashes.
    for (JsonElement element :
        JsonParser.parseString(callConfigCommand().outAsLatin1())
            .getAsJsonObject()
            .get("configuration-IDs")
            .getAsJsonArray()) {
      String configHash = element.getAsString();
      ConfigurationForOutput config =
          new Gson()
              .fromJson(callConfigCommand(configHash).outAsLatin1(), ConfigurationForOutput.class);
      if (isTargetConfig(config)) {
        if (targetConfig1Hash == null) {
          targetConfig1Hash = config.configHash;
        } else {
          assertThat(targetConfig2Hash).isNull();
          targetConfig2Hash = config.configHash;
        }
      }
    }
    assertThat(targetConfig1Hash).isNotNull();
    assertThat(targetConfig2Hash).isNotNull();

    // Get their diff.
    String result = callConfigCommand(targetConfig1Hash, targetConfig2Hash).outAsLatin1();
    ConfigurationDiffForOutput diff = new Gson().fromJson(result, ConfigurationDiffForOutput.class);
    assertThat(diff).isNotNull();
    assertThat(diff.configHash1).isEqualTo(targetConfig1Hash);
    assertThat(diff.configHash2).isEqualTo(targetConfig2Hash);
    FragmentDiffForOutput fragmentDiff = Iterables.getOnlyElement(diff.fragmentsDiff);
    assertThat(fragmentDiff.name).endsWith("CoreOptions");
    Map.Entry<String, Pair<String, String>> optionDiff =
        Iterables.getOnlyElement(fragmentDiff.optionsDiff.entrySet());
    assertThat(optionDiff.getKey()).isEqualTo("action_env");
    // Convert from Pair<firstVal, secondVal> to an ImmutableList because the ordering of the
    // difference depends on which configuration comes first, which depends on the configuration
    // hash name, which we can't predict statically.
    assertThat(ImmutableList.of(optionDiff.getValue().first, optionDiff.getValue().second))
        .containsExactly("[a=1]", "[b=2]");
  }

  @Test
  public void compareConfigsHashPrefix() throws Exception {
    analyzeTarget("--action_env=a=1");
    analyzeTarget("--action_env=b=2");
    String targetConfig1Hash = null;
    String targetConfig2Hash = null;

    // Find the two target configuration hashes.
    for (JsonElement element :
        JsonParser.parseString(callConfigCommand().outAsLatin1())
            .getAsJsonObject()
            .get("configuration-IDs")
            .getAsJsonArray()) {
      String configHash = element.getAsString();
      ConfigurationForOutput config =
          new Gson()
              .fromJson(callConfigCommand(configHash).outAsLatin1(), ConfigurationForOutput.class);
      if (isTargetConfig(config)) {
        if (targetConfig1Hash == null) {
          targetConfig1Hash = config.configHash;
        } else {
          assertThat(targetConfig2Hash).isNull();
          targetConfig2Hash = config.configHash;
        }
      }
    }

    String hashPrefix1 = targetConfig1Hash.substring(0, targetConfig1Hash.length() / 2);
    String hashPrefix2 = targetConfig2Hash.substring(0, targetConfig2Hash.length() / 2);

    ConfigurationDiffForOutput diff =
        new Gson()
            .fromJson(
                callConfigCommand(hashPrefix1, hashPrefix2).outAsLatin1(),
                ConfigurationDiffForOutput.class);
    assertThat(diff).isNotNull();
    assertThat(diff.configHash1).startsWith(hashPrefix1);
    assertThat(diff.configHash2).startsWith(hashPrefix2);
  }

  @Test
  public void compareConfigs_hostConfig() throws Exception {
    analyzeTarget("--action_env=a=1");
    analyzeTarget("--action_env=b=2");
    String targetConfigHash = null;

    // Find a target configuration hash.
    for (JsonElement element :
        JsonParser.parseString(callConfigCommand().outAsLatin1())
            .getAsJsonObject()
            .get("configuration-IDs")
            .getAsJsonArray()) {
      String configHash = element.getAsString();
      ConfigurationForOutput config =
          new Gson()
              .fromJson(callConfigCommand(configHash).outAsLatin1(), ConfigurationForOutput.class);
      if (isTargetConfig(config)) {
        if (targetConfigHash == null) {
          targetConfigHash = config.configHash;
          break;
        }
      }
    }

    ConfigurationDiffForOutput diff =
        new Gson()
            .fromJson(
                callConfigCommand(targetConfigHash, "host").outAsLatin1(),
                ConfigurationDiffForOutput.class);
    assertThat(diff).isNotNull();
    assertThat(diff.configHash1).isEqualTo(targetConfigHash);
    assertThat(diff.fragmentsDiff).isNotEmpty();

    // Find the "is host config" option, check that it is different.
    Optional<Pair<String, String>> isHostDiff =
        diff.fragmentsDiff.stream()
            .flatMap(fragmentDiff -> fragmentDiff.optionsDiff.entrySet().stream())
            .filter(od -> od.getKey().equals("is host configuration"))
            .map(Map.Entry::getValue)
            .findAny();
    assertThat(isHostDiff).isPresent();
    assertThat(isHostDiff.get().getFirst()).isEqualTo("false");
    assertThat(isHostDiff.get().getSecond()).isEqualTo("true");
  }

  @Test
  public void starlarkFlagsInUserDefinedFragment() throws Exception {
    write(
        "test/flagdef.bzl",
        "def _rule_impl(ctx):",
        "    return []",
        "string_flag = rule(",
        "    implementation = _rule_impl,",
        "    build_setting = config.string(flag = True)",
        ")",
        "simple_rule = rule(",
        "    implementation = _rule_impl,",
        "    attrs = {}",
        ")");
    write(
        "custom_flags/BUILD",
        "load('//test:flagdef.bzl', 'string_flag')",
        "string_flag(",
        "    name = 'my_flag',",
        "    build_setting_default = '')");

    analyzeTarget("--//custom_flags:my_flag=hello");

    ConfigurationForOutput targetConfig = null;
    String result = callConfigCommand("--dump_all").outAsLatin1();
    for (JsonElement configJson : JsonParser.parseString(result).getAsJsonArray()) {
      ConfigurationForOutput config = new Gson().fromJson(configJson, ConfigurationForOutput.class);
      if (isTargetConfig(config)) {
        targetConfig = config;
        break;
      }
    }

    assertThat(targetConfig).isNotNull();
    assertThat(getOptionValue(targetConfig, "user-defined", "//custom_flags:my_flag"))
        .isEqualTo("hello");
  }

  @Test
  public void defineFlagsIndividuallyListedInUserDefinedFragment() throws Exception {
    analyzeTarget("--define", "a=1", "--define", "b=2");

    ConfigurationForOutput targetConfig = null;
    for (JsonElement configJson :
        JsonParser.parseString(callConfigCommand("--dump_all").outAsLatin1()).getAsJsonArray()) {
      ConfigurationForOutput config = new Gson().fromJson(configJson, ConfigurationForOutput.class);
      if (isTargetConfig(config)) {
        targetConfig = config;
        break;
      }
    }

    assertThat(targetConfig).isNotNull();
    assertThat(getOptionValue(targetConfig, "user-defined", "--define:a")).isEqualTo("1");
    assertThat(getOptionValue(targetConfig, "user-defined", "--define:b")).isEqualTo("2");
    assertThat(
            targetConfig.fragmentOptions.stream()
                .filter(fragment -> fragment.name.endsWith("CoreOptions"))
                .flatMap(fragment -> fragment.options.keySet().stream())
                .filter(name -> name.equals("define"))
                .collect(Collectors.toList()))
        .isEmpty();
  }
}
