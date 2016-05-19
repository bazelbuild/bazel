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

package com.google.devtools.build.lib.rules.cpp;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.fail;

import com.google.common.base.Joiner;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Multimap;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.ExpansionException;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.Variables;
import com.google.devtools.build.lib.testutil.Suite;
import com.google.devtools.build.lib.testutil.TestSpec;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CToolchain;
import com.google.protobuf.TextFormat;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Tests for toolchain features.
 */
@RunWith(JUnit4.class)
@TestSpec(size = Suite.MEDIUM_TESTS)
public class CcToolchainFeaturesTest {
  
  /**
   * Creates a {@code Variables} configuration from a list of key/value pairs.
   * 
   * <p>If there are multiple entries with the same key, the variable will be treated as sequence
   * type.    
   */
  private Variables createVariables(String... entries) {
    if (entries.length % 2 != 0) {
      throw new IllegalArgumentException(
          "createVariables takes an even number of arguments (key/value pairs)");
    }
    Multimap<String, String> entryMap = ArrayListMultimap.create();
    for (int i = 0; i < entries.length; i += 2) {
      entryMap.put(entries[i], entries[i + 1]);
    }
    Variables.Builder variables = new Variables.Builder();
    for (String name : entryMap.keySet()) {
      Collection<String> value = entryMap.get(name);
      if (value.size() > 1) {
        variables.addSequenceVariable(name, value);
      } else {
        variables.addVariable(name, value.iterator().next());
      }
    }
    return variables.build();
  }
  
  /**
   * Creates a CcToolchainFeatures from features described in the given toolchain fragment.
   */
  public static CcToolchainFeatures buildFeatures(String... toolchain) throws Exception {
    CToolchain.Builder toolchainBuilder = CToolchain.newBuilder();
    TextFormat.merge(Joiner.on("").join(toolchain), toolchainBuilder);      
    return new CcToolchainFeatures(toolchainBuilder.buildPartial());    
  }

  private Set<String> getEnabledFeatures(CcToolchainFeatures features,
      String... requestedFeatures) throws Exception {
    FeatureConfiguration configuration =
        features.getFeatureConfiguration(Arrays.asList(requestedFeatures));
    ImmutableSet.Builder<String> enabledFeatures = ImmutableSet.builder();
    for (String feature : features.getActivatableNames()) {
      if (configuration.isEnabled(feature)) {
        enabledFeatures.add(feature);
      }
    }
    return enabledFeatures.build();
  }

  @Test
  public void testUnconditionalFeature() throws Exception {
    assertThat(buildFeatures("").getFeatureConfiguration("a")
        .isEnabled("a")).isFalse();
    assertThat(buildFeatures("feature { name: 'a' }").getFeatureConfiguration("b")
        .isEnabled("a")).isFalse();
    assertThat(buildFeatures("feature { name: 'a' }").getFeatureConfiguration("a")
        .isEnabled("a")).isTrue();
  }

  @Test
  public void testUnsupportedAction() throws Exception {
    FeatureConfiguration configuration = buildFeatures("").getFeatureConfiguration();
    assertThat(configuration.getCommandLine("invalid-action", createVariables())).isEmpty();
  }

  @Test
  public void testFlagOrderEqualsSpecOrder() throws Exception {
    FeatureConfiguration configuration = buildFeatures(
        "feature {",
        "  name: 'a'",
        "  flag_set {",
        "     action: 'c++-compile'",
        "     flag_group { flag: '-a-c++-compile' }",
        "  }",
        "  flag_set {",
        "     action: 'link'",
        "     flag_group { flag: '-a-c++-compile' }",
        "  }",
        "}",
        "feature {",
        "  name: 'b'",
        "  flag_set {",
        "     action: 'c++-compile'",
        "     flag_group { flag: '-b-c++-compile' }",
        "  }",
        "  flag_set {",
        "     action: 'link'",
        "     flag_group { flag: '-b-link' }",
        "  }",
        "}").getFeatureConfiguration("a", "b");
    List<String> commandLine = configuration.getCommandLine(
        CppCompileAction.CPP_COMPILE, createVariables());
    assertThat(commandLine).containsExactly("-a-c++-compile", "-b-c++-compile").inOrder();
  }

  @Test
  public void testEnvVars() throws Exception {
    FeatureConfiguration configuration = buildFeatures(
        "feature {",
        "  name: 'a'",
        "  env_set {",
        "     action: 'c++-compile'",
        "     env_entry { key: 'foo', value: 'bar' }",
        "     env_entry { key: 'cat', value: 'meow' }",
        "  }",
        "  flag_set {",
        "     action: 'c++-compile'",
        "     flag_group { flag: '-a-c++-compile' }",
        "  }",
        "}",
        "feature {",
        "  name: 'b'",
        "  env_set {",
        "     action: 'c++-compile'",
        "     env_entry { key: 'dog', value: 'woof' }",
        "  }",
        "}",
        "feature {",
        "  name: 'c'",
        "  env_set {",
        "     action: 'c++-compile'",
        "     env_entry { key: 'doNotInclude', value: 'doNotIncludePlease' }",
        "  }",
        "}").getFeatureConfiguration("a", "b");
    Map<String, String> env = configuration.getEnvironmentVariables(
        CppCompileAction.CPP_COMPILE, createVariables());
    assertThat(env).containsExactly("foo", "bar", "cat", "meow", "dog", "woof").inOrder();
    assertThat(env).doesNotContainEntry("doNotInclude", "doNotIncludePlease");
  }

  private String getExpansionOfFlag(String value) throws Exception {
    return getExpansionOfFlag(value, createVariables());
  }
  
  private List<String> getCommandLineForFlagGroups(String groups, Variables variables)
      throws Exception {
    FeatureConfiguration configuration = buildFeatures(
        "feature {",
        "  name: 'a'",
        "  flag_set {",
        "    action: 'c++-compile'",
        "    " + groups, 
        "  }",
        "}").getFeatureConfiguration("a");
    return configuration.getCommandLine(CppCompileAction.CPP_COMPILE, variables);    
  }
  
  private List<String> getCommandLineForFlag(String value, Variables variables) throws Exception {
    return getCommandLineForFlagGroups("flag_group { flag: '" + value + "' }", variables);
  }
  
  private String getExpansionOfFlag(String value, Variables variables) throws Exception {
    return getCommandLineForFlag(value, variables).get(0);
  }
  
  private String getFlagParsingError(String value) throws Exception {
    try {
      getExpansionOfFlag(value);
      fail("Expected InvalidConfigurationException");
      return "";
    } catch (InvalidConfigurationException e) {
      return e.getMessage();
    }
  }
  
  private String getFlagExpansionError(String value, Variables variables) throws Exception {
    try {
      getExpansionOfFlag(value, variables);
      fail("Expected ExpansionException");
      return "";
    } catch (ExpansionException e) {
      return e.getMessage();
    }
  }

  @Test
  public void testVariableExpansion() throws Exception {
    assertThat(getExpansionOfFlag("%%")).isEqualTo("%");
    assertThat(getExpansionOfFlag("%% a %% b %%")).isEqualTo("% a % b %");
    assertThat(getExpansionOfFlag("%%{var}")).isEqualTo("%{var}");
    assertThat(getExpansionOfFlag("%{v}", createVariables("v", "<flag>"))).isEqualTo("<flag>");
    assertThat(getExpansionOfFlag(" %{v1} %{v2} ", createVariables("v1", "1", "v2", "2")))
        .isEqualTo(" 1 2 ");
    assertThat(getFlagParsingError("%")).contains("expected '{'");
    assertThat(getFlagParsingError("% ")).contains("expected '{'");
    assertThat(getFlagParsingError("%{")).contains("expected variable name");
    assertThat(getFlagParsingError("%{}")).contains("expected variable name");
    assertThat(getCommandLineForFlag("%{v}",
        new Variables.Builder().addSequenceVariable("v", ImmutableList.<String>of()).build()))
        .isEmpty();
    assertThat(getFlagExpansionError("%{v}", createVariables())).contains("unknown variable 'v'");
    assertThat(getFlagExpansionError("%{v}", new Variables.Builder()
        .addSequenceVariable("v", ImmutableList.<String>of("1"))
        .addVariable("v", "2").build()))
        .contains("variable 'v'");
  }

  @Test
  public void testListVariableExpansion() throws Exception {
    assertThat(getCommandLineForFlag("%{v}", createVariables("v", "1", "v", "2")))
        .containsExactly("1", "2");
    assertThat(getCommandLineForFlag("%{v1} %{v2}",
        createVariables("v1", "a1", "v1", "a2", "v2", "b")))
        .containsExactly("a1 b", "a2 b");
    assertThat(getFlagExpansionError("%{v1} %{v2}",
        createVariables("v1", "a1", "v1", "a2", "v2", "b1", "v2", "b2")))
        .contains("'v1' and 'v2'");
  }

  @Test
  public void testFlagGroupVariableExpansion() throws Exception {
    assertThat(getCommandLineForFlagGroups(
        "flag_group { flag: '-f' flag: '%{v}' } flag_group { flag: '-end' }",
        createVariables("v", "1", "v", "2")))
        .containsExactly("-f", "1", "-f", "2", "-end");
    assertThat(getCommandLineForFlagGroups(
        "flag_group { flag: '-f' flag: '%{v}' } flag_group { flag: '%{v}' }",
        createVariables("v", "1", "v", "2")))
        .containsExactly("-f", "1", "-f", "2", "1", "2");
    assertThat(getCommandLineForFlagGroups(
        "flag_group { flag: '-f' flag: '%{v}' } flag_group { flag: '%{v}' }",
        createVariables("v", "1", "v", "2")))
        .containsExactly("-f", "1", "-f", "2", "1", "2");
    try {
      getCommandLineForFlagGroups(
          "flag_group { flag: '%{v1}' flag: '%{v2}' }",
          createVariables("v1", "1", "v1", "2", "v2", "1", "v2", "2"));
      fail("Expected ExpansionException");
    } catch (ExpansionException e) {
      assertThat(e.getMessage()).contains("'v1' and 'v2'");
    }
  }
  
  private Variables.Sequence createNestedSequence(int depth, int count, String prefix) {
    if (depth == 0) {
      Variables.ValueSequence.Builder builder = new Variables.ValueSequence.Builder();
      for (int i = 0; i < count; ++i) {
        String value = prefix + i;
        builder.addValue(value);
      }
      return builder.build();

    } else {
      Variables.NestedSequence.Builder builder = new Variables.NestedSequence.Builder();
      for (int i = 0; i < count; ++i) {
        String value = prefix + i;
        builder.addSequence(createNestedSequence(depth - 1, count, value));
      }
      return builder.build();
    }
  }

  private Variables createNestedVariables(String name, int depth, int count) {
    return new Variables.Builder()
        .addSequence(name, createNestedSequence(depth, count, "")).build();
  }

  @Test
  public void testFlagTreeVariableExpansion() throws Exception {
    String nestedGroup =
        ""
            + "flag_group {"
            + "  flag_group { flag: '-a' }"
            + "  flag_group {"
            + "    flag: '%{v}'"
            + "  }"
            + "  flag_group { flag: '-b' }"
            + "}";
    assertThat(getCommandLineForFlagGroups(nestedGroup, createNestedVariables("v", 1, 3)))
        .containsExactly(
            "-a", "00", "01", "02", "-b", "-a", "10", "11", "12", "-b", "-a", "20", "21", "22",
            "-b");

    assertThat(getCommandLineForFlagGroups(nestedGroup, createNestedVariables("v", 0, 3)))
        .containsExactly("-a", "0", "-b", "-a", "1", "-b", "-a", "2", "-b");

    try {
      getCommandLineForFlagGroups(nestedGroup, createNestedVariables("v", 2, 3));
      fail("Expected ExpansionException");
    } catch (ExpansionException e) {
      assertThat(e.getMessage()).contains("'v'");
    }

    try {
      buildFeatures(
          "feature {",
          "  name: 'a'",
          "  flag_set {",
          "    action: 'c++-compile'",
          "    flag_group {",
          "      flag_group { flag: '-f' }",
          "      flag: '-f'",
          "    }",
          "  }",
          "}");
      fail("Expected ExpansionException");
    } catch (ExpansionException e) {
      assertThat(e.getMessage()).contains("Invalid toolchain configuration");
    }
  }

  @Test
  public void testImplies() throws Exception {
    CcToolchainFeatures features = buildFeatures(
        "feature { name: 'a' implies: 'b' implies: 'c' }",
        "feature { name: 'b' }",
        "feature { name: 'c' implies: 'd' }",
        "feature { name: 'd' }",
        "feature { name: 'e' }");
    assertThat(getEnabledFeatures(features, "a")).containsExactly("a", "b", "c", "d");
  }

  @Test
  public void testRequires() throws Exception {
    CcToolchainFeatures features = buildFeatures(
        "feature { name: 'a' requires: { feature: 'b' } }",
        "feature { name: 'b' requires: { feature: 'c' } }",
        "feature { name: 'c' }"); 
    assertThat(getEnabledFeatures(features, "a")).isEmpty();
    assertThat(getEnabledFeatures(features, "a", "b")).isEmpty();
    assertThat(getEnabledFeatures(features, "a", "c")).containsExactly("c");
    assertThat(getEnabledFeatures(features, "a", "b", "c")).containsExactly("a", "b", "c");
  }

  @Test
  public void testDisabledRequirementChain() throws Exception {
    CcToolchainFeatures features = buildFeatures(
        "feature { name: 'a' }",
        "feature { name: 'b' requires: { feature: 'c' } implies: 'a' }",
        "feature { name: 'c' }");
    assertThat(getEnabledFeatures(features, "b")).isEmpty();
    features = buildFeatures(
        "feature { name: 'a' }",
        "feature { name: 'b' requires: { feature: 'a' } implies: 'c' }",
        "feature { name: 'c' }",
        "feature { name: 'd' requires: { feature: 'c' } implies: 'e' }",
        "feature { name: 'e' }"); 
    assertThat(getEnabledFeatures(features, "b", "d")).isEmpty();
  }

  @Test
  public void testEnabledRequirementChain() throws Exception {
    CcToolchainFeatures features = buildFeatures(
        "feature { name: '0' implies: 'a' }",
        "feature { name: 'a' }",
        "feature { name: 'b' requires: { feature: 'a' } implies: 'c' }",
        "feature { name: 'c' }",
        "feature { name: 'd' requires: { feature: 'c' } implies: 'e' }",
        "feature { name: 'e' }"); 
    assertThat(getEnabledFeatures(features, "0", "b", "d")).containsExactly(
        "0", "a", "b", "c", "d", "e");
  }

  @Test
  public void testLogicInRequirements() throws Exception {
    CcToolchainFeatures features = buildFeatures(
        "feature { name: 'a' requires: { feature: 'b' feature: 'c' } requires: { feature: 'd' } }",
        "feature { name: 'b' }",
        "feature { name: 'c' }",
        "feature { name: 'd' }");
    assertThat(getEnabledFeatures(features, "a", "b", "c")).containsExactly("a", "b", "c");
    assertThat(getEnabledFeatures(features, "a", "b")).containsExactly("b");
    assertThat(getEnabledFeatures(features, "a", "c")).containsExactly("c");
    assertThat(getEnabledFeatures(features, "a", "d")).containsExactly("a", "d");
  }

  @Test
  public void testImpliesImpliesRequires() throws Exception {
    CcToolchainFeatures features = buildFeatures(
        "feature { name: 'a' implies: 'b' }",
        "feature { name: 'b' requires: { feature: 'c' } }",
        "feature { name: 'c' }");
    assertThat(getEnabledFeatures(features, "a")).isEmpty();
  }

  @Test
  public void testMultipleImplies() throws Exception {
    CcToolchainFeatures features = buildFeatures(
        "feature { name: 'a' implies: 'b' implies: 'c' implies: 'd' }",
        "feature { name: 'b' }",
        "feature { name: 'c' requires: { feature: 'e' } }",
        "feature { name: 'd' }",
        "feature { name: 'e' }");
    assertThat(getEnabledFeatures(features, "a")).isEmpty();
    assertThat(getEnabledFeatures(features, "a", "e")).containsExactly("a", "b", "c", "d", "e");
  }

  @Test
  public void testDisabledFeaturesDoNotEnableImplications() throws Exception {
    CcToolchainFeatures features = buildFeatures(
        "feature { name: 'a' implies: 'b' requires: { feature: 'c' } }",
        "feature { name: 'b' }",
        "feature { name: 'c' }");
    assertThat(getEnabledFeatures(features, "a")).isEmpty();    
  }

  @Test
  public void testFeatureNameCollision() throws Exception {
    try {
      buildFeatures(
          "feature { name: '<<<collision>>>' }",
          "feature { name: '<<<collision>>>' }");
      fail("Expected InvalidConfigurationException");
    } catch (InvalidConfigurationException e) {
      assertThat(e.getMessage()).contains("<<<collision>>>");
    }
  }

  @Test
  public void testReferenceToUndefinedFeature() throws Exception {
    try {
      buildFeatures("feature { name: 'a' implies: '<<<undefined>>>' }");
      fail("Expected InvalidConfigurationException");
    } catch (InvalidConfigurationException e) {
      assertThat(e.getMessage()).contains("<<<undefined>>>");
    }
  }

  @Test
  public void testImpliesWithCycle() throws Exception {
    CcToolchainFeatures features = buildFeatures(
        "feature { name: 'a' implies: 'b' }",
        "feature { name: 'b' implies: 'a' }");
    assertThat(getEnabledFeatures(features, "a")).containsExactly("a", "b");
    assertThat(getEnabledFeatures(features, "b")).containsExactly("a", "b");
 }

  @Test
  public void testMultipleImpliesCycle() throws Exception {
    CcToolchainFeatures features = buildFeatures(
        "feature { name: 'a' implies: 'b' implies: 'c' implies: 'd' }",
        "feature { name: 'b' }",
        "feature { name: 'c' requires: { feature: 'e' } }",
        "feature { name: 'd' requires: { feature: 'f' } }",
        "feature { name: 'e' requires: { feature: 'c' } }",
        "feature { name: 'f' }");
    assertThat(getEnabledFeatures(features, "a", "e")).isEmpty();
    assertThat(getEnabledFeatures(features, "a", "e", "f")).containsExactly(
        "a", "b", "c", "d", "e", "f");
  }

  @Test
  public void testRequiresWithCycle() throws Exception {
    CcToolchainFeatures features = buildFeatures(
        "feature { name: 'a' requires: { feature: 'b' } }",
        "feature { name: 'b' requires: { feature: 'a' } }",
        "feature { name: 'c' implies: 'a' }",
        "feature { name: 'd' implies: 'b' }");
    assertThat(getEnabledFeatures(features, "c")).isEmpty();
    assertThat(getEnabledFeatures(features, "d")).isEmpty();
    assertThat(getEnabledFeatures(features, "c", "d")).containsExactly("a", "b", "c", "d");
  }

  @Test
  public void testImpliedByOneEnabledAndOneDisabledFeature() throws Exception {
    CcToolchainFeatures features = buildFeatures(
        "feature { name: 'a' }",
        "feature { name: 'b' requires: { feature: 'a' } implies: 'd' }",
        "feature { name: 'c' implies: 'd' }",
        "feature { name: 'd' }");
    assertThat(getEnabledFeatures(features, "b", "c")).containsExactly("c", "d");    
  }

  @Test
  public void testRequiresOneEnabledAndOneUnsupportedFeature() throws Exception {
    CcToolchainFeatures features = buildFeatures(
        "feature { name: 'a' requires: { feature: 'b' } requires: { feature: 'c' } }",
        "feature { name: 'b' }",
        "feature { name: 'c' requires: { feature: 'd' } }",
        "feature { name: 'd' }");
    assertThat(getEnabledFeatures(features, "a", "b", "c")).containsExactly("a", "b");        
  }

  @Test
  public void testSuppressionViaMissingBuildVariable() throws Exception {
    FeatureConfiguration configuration =
        buildFeatures(
                "feature {",
                "  name: 'a'",
                "  flag_set {",
                "     action: 'c++-compile'",
                "     expand_if_all_available: 'v'",
                "     flag_group { flag: '%{v}' }",
                "  }",
                "  flag_set {",
                "     action: 'c++-compile'",
                "     expand_if_all_available: 'v'",
                "     expand_if_all_available: 'w'",
                "     flag_group { flag: '%{v}%{w}' }",
                "  }",
                "  flag_set {",
                "     action: 'c++-compile'",
                "     flag_group { flag: 'unconditional' }",
                "  }",
                "}")
            .getFeatureConfiguration("a");

    assertThat(configuration.getCommandLine(CppCompileAction.CPP_COMPILE, createVariables()))
        .containsExactly("unconditional");
    assertThat(
            configuration.getCommandLine(CppCompileAction.CPP_COMPILE, createVariables("v", "1")))
        .containsExactly("1", "unconditional");
    assertThat(
            configuration.getCommandLine(
                CppCompileAction.CPP_COMPILE, createVariables("v", "1", "v", "2")))
        .containsExactly("1", "2", "unconditional")
        .inOrder();
    assertThat(
            configuration.getCommandLine(
                CppCompileAction.CPP_COMPILE, createVariables("v", "1", "v", "2", "w", "3")))
        .containsExactly("1", "2", "13", "23", "unconditional")
        .inOrder();
  }

  @Test
  public void testConfiguration() throws Exception {
    CcToolchainFeatures features = buildFeatures(
        "feature {",
        "  name: 'a'",
        "  flag_set {",
        "    action: 'c++-compile'",
        "    flag_group {",
        "      flag: '-f'",
        "      flag: '%{v}'",
        "    }",
        "  }",
        "}",
        "feature { name: 'b' implies: 'a' }");
    assertThat(getEnabledFeatures(features, "b")).containsExactly("a", "b");
    assertThat(features.getFeatureConfiguration("b").getCommandLine(CppCompileAction.CPP_COMPILE,
        createVariables("v", "1"))).containsExactly("-f", "1");
    byte[] serialized = TestUtils.serializeObject(features);
    CcToolchainFeatures deserialized =
        (CcToolchainFeatures) TestUtils.deserializeObject(serialized);
    assertThat(getEnabledFeatures(deserialized, "b")).containsExactly("a", "b");    
    assertThat(features.getFeatureConfiguration("b").getCommandLine(CppCompileAction.CPP_COMPILE,
        createVariables("v", "1"))).containsExactly("-f", "1");
  }

  @Test
  public void testActivateActionConfigFromFeature() throws Exception {
    CcToolchainFeatures toolchainFeatures =
        buildFeatures(
            "action_config {",
            "  config_name: 'action-a'",
            "  action_name: 'action-a'",
            "  tool {",
            "    tool_path: 'toolchain/feature-a'",
            "    with_feature: { feature: 'feature-a' }",
            "  }",
            "}",
            "feature {",
            "   name: 'activates-action-a'",
            "   implies: 'action-a'",
            "}");

    FeatureConfiguration featureConfiguration =
        toolchainFeatures.getFeatureConfiguration("activates-action-a");

    assertThat(featureConfiguration.actionIsConfigured("action-a")).isTrue();
  }

  @Test
  public void testFeatureCanRequireActionConfig() throws Exception {
    CcToolchainFeatures toolchainFeatures =
        buildFeatures(
            "action_config {",
            "  config_name: 'action-a'",
            "  action_name: 'action-a'",
            "  tool {",
            "    tool_path: 'toolchain/feature-a'",
            "    with_feature: { feature: 'feature-a' }",
            "  }",
            "}",
            "feature {",
            "   name: 'requires-action-a'",
            "   requires: { feature: 'action-a' }",
            "}");

    FeatureConfiguration featureConfigurationWithoutAction =
        toolchainFeatures.getFeatureConfiguration("requires-action-a");
    assertThat(featureConfigurationWithoutAction.isEnabled("requires-action-a")).isFalse();

    FeatureConfiguration featureConfigurationWithAction =
        toolchainFeatures.getFeatureConfiguration("action-a", "requires-action-a");
    assertThat(featureConfigurationWithAction.isEnabled("requires-action-a")).isTrue();
  }

  @Test
  public void testSimpleActionTool() throws Exception {
    FeatureConfiguration configuration =
        buildFeatures(
                "action_config {",
                "  config_name: 'action-a'",
                "  action_name: 'action-a'",
                "  tool {",
                "    tool_path: 'toolchain/a'",
                "  }",
                "}",
                "feature {",
                "   name: 'activates-action-a'",
                "   implies: 'action-a'",
                "}")
            .getFeatureConfiguration("activates-action-a");
    PathFragment crosstoolPath = new PathFragment("crosstool/");
    PathFragment toolPath = configuration.getToolForAction("action-a").getToolPath(crosstoolPath);
    assertThat(toolPath.toString()).isEqualTo("crosstool/toolchain/a");
  }

  @Test
  public void testActionToolFromFeatureSet() throws Exception {
    CcToolchainFeatures toolchainFeatures =
        buildFeatures(
            "action_config {",
            "  config_name: 'action-a'",
            "  action_name: 'action-a'",
            "  tool {",
            "    tool_path: 'toolchain/features-a-and-b'",
            "    with_feature: {",
            "      feature: 'feature-a'",
            "      feature: 'feature-b'",
            "     }",
            "  }",
            "  tool {",
            "    tool_path: 'toolchain/feature-a'",
            "    with_feature: { feature: 'feature-a' }",
            "  }",
            "  tool {",
            "    tool_path: 'toolchain/feature-b'",
            "    with_feature: { feature: 'feature-b' }",
            "  }",
            "  tool {",
            "    tool_path: 'toolchain/default'",
            "  }",
            "}",
            "feature {",
            "  name: 'feature-a'",
            "}",
            "feature {",
            "  name: 'feature-b'",
            "}",
            "feature {",
            "  name: 'activates-action-a'",
            "  implies: 'action-a'",
            "}");

    PathFragment crosstoolPath = new PathFragment("crosstool/");

    FeatureConfiguration featureAConfiguration =
        toolchainFeatures.getFeatureConfiguration("feature-a", "activates-action-a");
    assertThat(
            featureAConfiguration
                .getToolForAction("action-a")
                .getToolPath(crosstoolPath)
                .toString())
        .isEqualTo("crosstool/toolchain/feature-a");

    FeatureConfiguration featureBConfiguration =
        toolchainFeatures.getFeatureConfiguration("feature-b", "activates-action-a");
    assertThat(
            featureBConfiguration
                .getToolForAction("action-a")
                .getToolPath(crosstoolPath)
                .toString())
        .isEqualTo("crosstool/toolchain/feature-b");

    FeatureConfiguration featureAAndBConfiguration =
        toolchainFeatures.getFeatureConfiguration("feature-a", "feature-b", "activates-action-a");
    assertThat(
            featureAAndBConfiguration
                .getToolForAction("action-a")
                .getToolPath(crosstoolPath)
                .toString())
        .isEqualTo("crosstool/toolchain/features-a-and-b");

    FeatureConfiguration noFeaturesConfiguration =
        toolchainFeatures.getFeatureConfiguration("activates-action-a");
    assertThat(
            noFeaturesConfiguration
                .getToolForAction("action-a")
                .getToolPath(crosstoolPath)
                .toString())
        .isEqualTo("crosstool/toolchain/default");
  }

  @Test
  public void testErrorForNoMatchingTool() throws Exception {
    CcToolchainFeatures toolchainFeatures =
        buildFeatures(
            "action_config {",
            "  config_name: 'action-a'",
            "  action_name: 'action-a'",
            "  tool {",
            "    tool_path: 'toolchain/feature-a'",
            "    with_feature: { feature: 'feature-a' }",
            "  }",
            "}",
            "feature {",
            "  name: 'feature-a'",
            "}",
            "feature {",
            "  name: 'activates-action-a'",
            "  implies: 'action-a'",
            "}");

    PathFragment crosstoolPath = new PathFragment("crosstool/");

    FeatureConfiguration noFeaturesConfiguration =
        toolchainFeatures.getFeatureConfiguration("activates-action-a");

    try {
      noFeaturesConfiguration.getToolForAction("action-a").getToolPath(crosstoolPath);
      fail("Expected IllegalArgumentException");
    } catch (IllegalArgumentException e) {
      assertThat(e.getMessage())
          .contains("Matching tool for action action-a not found for given feature configuration");
    }
  }

  @Test
  public void testActivateActionConfigDirectly() throws Exception {
    CcToolchainFeatures toolchainFeatures =
        buildFeatures(
            "action_config {",
            "  config_name: 'action-a'",
            "  action_name: 'action-a'",
            "  tool {",
            "    tool_path: 'toolchain/feature-a'",
            "    with_feature: { feature: 'feature-a' }",
            "  }",
            "}");

    FeatureConfiguration featureConfiguration =
        toolchainFeatures.getFeatureConfiguration("action-a");

    assertThat(featureConfiguration.actionIsConfigured("action-a")).isTrue();
  }

  @Test
  public void testActionConfigCanActivateFeature() throws Exception {
    CcToolchainFeatures toolchainFeatures =
        buildFeatures(
            "action_config {",
            "  config_name: 'action-a'",
            "  action_name: 'action-a'",
            "  tool {",
            "    tool_path: 'toolchain/feature-a'",
            "    with_feature: { feature: 'feature-a' }",
            "  }",
            "  implies: 'activated-feature'",
            "}",
            "feature {",
            "   name: 'activated-feature'",
            "}");

    FeatureConfiguration featureConfiguration =
        toolchainFeatures.getFeatureConfiguration("action-a");

    assertThat(featureConfiguration.isEnabled("activated-feature")).isTrue();
  }

  @Test
  public void testInvalidActionConfigurationDuplicateActionConfigs() throws Exception {
    try {
      buildFeatures(
          "action_config {",
          "  config_name: 'action-a'",
          "  action_name: 'action-1'",
          "}",
          "action_config {",
          "  config_name: 'action-a'",
          "  action_name: 'action-2'",
          "}");
      fail("Expected InvalidConfigurationException");
    } catch (InvalidConfigurationException e) {
      assertThat(e.getMessage())
          .contains("feature or action config 'action-a' was specified multiple times.");
    }
  }

  @Test
  public void testInvalidActionConfigurationMultipleActionConfigsForAction() throws Exception {
    try {
      buildFeatures(
          "action_config {",
          "  config_name: 'name-a'",
          "  action_name: 'action-a'",
          "}",
          "action_config {",
          "  config_name: 'name-b'",
          "  action_name: 'action-a'",
          "}");
      fail("Expected InvalidConfigurationException");
    } catch (InvalidConfigurationException e) {
      assertThat(e.getMessage()).contains("multiple action configs for action 'action-a'");
    }
  }
}
