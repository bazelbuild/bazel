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
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.ActionConfig;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.ExpansionException;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.Variables;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.Variables.IntegerValue;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.Variables.LibraryToLinkValue;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.Variables.SequenceBuilder;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.Variables.StringSequenceBuilder;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.Variables.StructureBuilder;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.Variables.VariableValue;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.Variables.VariableValueBuilder;
import com.google.devtools.build.lib.testutil.Suite;
import com.google.devtools.build.lib.testutil.TestSpec;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CToolchain;
import com.google.protobuf.TextFormat;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

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
      if (value.size() == 1) {
        variables.addStringVariable(name, value.iterator().next());
      } else {
        variables.addStringSequenceVariable(name, ImmutableList.copyOf(value));
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

  private String getFlagGroupsExpansionError(String flagGroups, Variables variables)
      throws Exception {
    try {
      getCommandLineForFlagGroups(flagGroups, variables).get(0);
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
    assertThat(
            getCommandLineForFlag(
                "%{v}",
                new Variables.Builder()
                    .addStringSequenceVariable("v", ImmutableList.<String>of())
                    .build()))
        .isEmpty();
    assertThat(getFlagExpansionError("%{v}", createVariables()))
        .contains("Invalid toolchain configuration: Cannot find variable named 'v'");
  }

  private Variables createStructureSequenceVariables(String name, StructureBuilder... values) {
    SequenceBuilder builder = new SequenceBuilder();
    for (StructureBuilder value : values) {
      builder.addValue(value.build());
    }
    return new Variables.Builder().addCustomBuiltVariable(name, builder).build();
  }

  private Variables createStructureVariables(String name, StructureBuilder value) {
    return new Variables.Builder().addCustomBuiltVariable(name, value).build();
  }

  @Test
  public void testSimpleStructureVariableExpansion() throws Exception {
    assertThat(
            getCommandLineForFlagGroups(
                "flag_group { flag: '-A%{struct.foo}' flag: '-B%{struct.bar}' }",
                createStructureVariables(
                    "struct",
                    new StructureBuilder()
                        .addField("foo", "fooValue")
                        .addField("bar", "barValue"))))
        .containsExactly("-AfooValue", "-BbarValue");
  }

  @Test
  public void testNestedStructureVariableExpansion() throws Exception {
    assertThat(
            getCommandLineForFlagGroups(
                "flag_group { flag: '-A%{struct.foo.bar}' }",
                createStructureVariables(
                    "struct",
                    new StructureBuilder()
                        .addField("foo", new StructureBuilder().addField("bar", "fooBarValue")))))
        .containsExactly("-AfooBarValue");
  }

  @Test
  public void testAccessingStructureAsStringFails() throws Exception {
    assertThat(
            getFlagGroupsExpansionError(
                "flag_group { flag: '-A%{struct}' }",
                createStructureVariables(
                    "struct",
                    new StructureBuilder()
                        .addField("foo", "fooValue")
                        .addField("bar", "barValue"))))
        .isEqualTo(
            "Invalid toolchain configuration: Cannot expand variable 'struct': expected string, "
                + "found structure");
  }

  @Test
  public void testAccessingStringValueAsStructureFails() throws Exception {
    assertThat(
            getFlagGroupsExpansionError(
                "flag_group { flag: '-A%{stringVar.foo}' }",
                createVariables("stringVar", "stringVarValue")))
        .isEqualTo(
            "Invalid toolchain configuration: Cannot expand variable 'stringVar.foo': variable "
                + "'stringVar' is string, expected structure");
  }

  @Test
  public void testAccessingSequenceAsStructureFails() throws Exception {
    assertThat(
            getFlagGroupsExpansionError(
                "flag_group { flag: '-A%{sequence.foo}' }",
                createVariables("sequence", "foo1", "sequence", "foo2")))
        .isEqualTo(
            "Invalid toolchain configuration: Cannot expand variable 'sequence.foo': variable "
                + "'sequence' is sequence, expected structure");
  }

  @Test
  public void testAccessingMissingStructureFieldFails() throws Exception {
    assertThat(
            getFlagGroupsExpansionError(
                "flag_group { flag: '-A%{struct.missing}' }",
                createStructureVariables(
                    "struct", new StructureBuilder().addField("bar", "barValue"))))
        .isEqualTo(
            "Invalid toolchain configuration: Cannot expand variable 'struct.missing': structure "
                + "struct doesn't have a field named 'missing'");
  }

  @Test
  public void testSequenceOfStructuresExpansion() throws Exception {
    assertThat(
            getCommandLineForFlagGroups(
                "flag_group { iterate_over: 'structs' flag: '-A%{structs.foo}' }",
                createStructureSequenceVariables(
                    "structs",
                    new StructureBuilder().addField("foo", "foo1Value"),
                    new StructureBuilder().addField("foo", "foo2Value"))))
        .containsExactly("-Afoo1Value", "-Afoo2Value");
  }

  @Test
  public void testStructureOfSequencesExpansion() throws Exception {
    assertThat(
            getCommandLineForFlagGroups(
                "flag_group {"
                    + "  iterate_over: 'struct.sequences'"
                    + "  flag: '-A%{struct.sequences.foo}'"
                    + "}",
                createStructureVariables(
                    "struct",
                    new StructureBuilder()
                        .addField(
                            "sequences",
                            new SequenceBuilder()
                                .addValue(new StructureBuilder().addField("foo", "foo1Value"))
                                .addValue(new StructureBuilder().addField("foo", "foo2Value"))))))
        .containsExactly("-Afoo1Value", "-Afoo2Value");
  }

  @Test
  public void testDottedNamesNotAlwaysMeanStructures() throws Exception {
    assertThat(
            getCommandLineForFlagGroups(
                "flag_group {"
                    + "  iterate_over: 'struct.sequence'"
                    + "  flag_group {"
                    + "    iterate_over: 'other_sequence'"
                    + "    flag_group {"
                    + "      flag: '-A%{struct.sequence} -B%{other_sequence}'"
                    + "    }"
                    + "  }"
                    + "}",
                new Variables.Builder()
                    .addCustomBuiltVariable(
                        "struct",
                        new StructureBuilder()
                            .addField("sequence", ImmutableList.of("first", "second")))
                    .addStringSequenceVariable("other_sequence", ImmutableList.of("foo", "bar"))
                    .build()))
        .containsExactly("-Afirst -Bfoo", "-Afirst -Bbar", "-Asecond -Bfoo", "-Asecond -Bbar");
  }

  // <p>TODO(b/32655571): Get rid of this test once implicit iteration is not supported.
  // It's there only to document a known limitation of the system.
  @Test
  public void testVariableLookupIsBrokenForImplicitStructFieldIteration() throws Exception {
    assertThat(
            getCommandLineForFlagGroups(
                "flag_group { flag: '-A%{struct.sequence}' }",
                createStructureVariables(
                    "struct",
                    new StructureBuilder().addField("sequence", ImmutableList.of("foo", "bar")))))
        .containsExactly("-Afoo", "-Abar");
  }

  @Test
  public void testExpandIfAllAvailableWithStructsExpandsIfPresent() throws Exception {
    assertThat(
            getCommandLineForFlagGroups(
                "flag_group {"
                    + "  expand_if_all_available: 'struct'"
                    + "  flag: '-A%{struct.foo}'"
                    + "  flag: '-B%{struct.bar}'"
                    + "}",
                createStructureVariables(
                    "struct",
                    new Variables.StructureBuilder()
                        .addField("foo", "fooValue")
                        .addField("bar", "barValue"))))
        .containsExactly("-AfooValue", "-BbarValue");
  }

  @Test
  public void testExpandIfAllAvailableWithStructsDoesntExpandIfMissing() throws Exception {
    assertThat(
            getCommandLineForFlagGroups(
                "flag_group {"
                    + "  expand_if_all_available: 'nonexistent'"
                    + "  flag: '-A%{struct.foo}'"
                    + "  flag: '-B%{struct.bar}'"
                    + "}",
                createStructureVariables(
                    "struct",
                    new Variables.StructureBuilder()
                        .addField("foo", "fooValue")
                        .addField("bar", "barValue"))))
        .isEmpty();
  }

  @Test
  public void testExpandIfAllAvailableWithStructsDoesntCrashIfMissing() throws Exception {
    assertThat(
            getCommandLineForFlagGroups(
                "flag_group {"
                    + "  expand_if_all_available: 'nonexistent'"
                    + "  flag: '-A%{nonexistent.foo}'"
                    + "  flag: '-B%{nonexistent.bar}'"
                    + "}",
                createVariables()))
        .isEmpty();
  }

  @Test
  public void testExpandIfAllAvailableWithStructFieldDoesntCrashIfMissing() throws Exception {
    assertThat(
            getCommandLineForFlagGroups(
                "flag_group {"
                    + "  expand_if_all_available: 'nonexistent.nonexistant_field'"
                    + "  flag: '-A%{nonexistent.foo}'"
                    + "  flag: '-B%{nonexistent.bar}'"
                    + "}",
                createVariables()))
        .isEmpty();
  }

  @Test
  public void testExpandIfAllAvailableWithStructFieldExpandsIfPresent() throws Exception {
    assertThat(
            getCommandLineForFlagGroups(
                "flag_group {"
                    + "  expand_if_all_available: 'struct.foo'"
                    + "  flag: '-A%{struct.foo}'"
                    + "  flag: '-B%{struct.bar}'"
                    + "}",
                createStructureVariables(
                    "struct",
                    new Variables.StructureBuilder()
                        .addField("foo", "fooValue")
                        .addField("bar", "barValue"))))
        .containsExactly("-AfooValue", "-BbarValue");
  }

  @Test
  public void testExpandIfAllAvailableWithStructFieldDoesntExpandIfMissing() throws Exception {
    assertThat(
            getCommandLineForFlagGroups(
                "flag_group {"
                    + "  expand_if_all_available: 'struct.foo'"
                    + "  flag: '-A%{struct.foo}'"
                    + "  flag: '-B%{struct.bar}'"
                    + "}",
                createStructureVariables(
                    "struct", new Variables.StructureBuilder().addField("bar", "barValue"))))
        .isEmpty();
  }

  @Test
  public void testExpandIfAllAvailableWithStructFieldScopesRight() throws Exception {
    assertThat(
            getCommandLineForFlagGroups(
                "flag_group {"
                    + "  flag_group {"
                    + "    expand_if_all_available: 'struct.foo'"
                    + "    flag: '-A%{struct.foo}'"
                    + "  }"
                    + "  flag_group { "
                    + "    flag: '-B%{struct.bar}'"
                    + "  }"
                    + "}",
                createStructureVariables(
                    "struct", new Variables.StructureBuilder().addField("bar", "barValue"))))
        .containsExactly("-BbarValue");
  }

  @Test
  public void testExpandIfNoneAvailableExpandsIfNotAvailable() throws Exception {
    assertThat(
        getCommandLineForFlagGroups(
            "flag_group {"
                + "  flag_group {"
                + "    expand_if_none_available: 'not_available'"
                + "    flag: '-foo'"
                + "  }"
                + "  flag_group { "
                + "    expand_if_none_available: 'available'"
                + "    flag: '-bar'"
                + "  }"
                + "}",
            createVariables("available", "available")))
        .containsExactly("-foo");
  }

  @Test
  public void testExpandIfNoneAvailableDoesntExpandIfThereIsOneOfManyAvailable() throws Exception {
    assertThat(
        getCommandLineForFlagGroups(
            "flag_group {"
                + "  flag_group {"
                + "    expand_if_none_available: 'not_available'"
                + "    expand_if_none_available: 'available'"
                + "    flag: '-foo'"
                + "  }"
                + "}",
            createVariables("available", "available")))
        .isEmpty();
  }

  @Test
  public void testExpandIfTrueDoesntExpandIfMissing() throws Exception {
    assertThat(
        getCommandLineForFlagGroups(
            "flag_group {"
                + "  expand_if_true: 'missing'"
                + "  flag: '-A%{missing}'"
                + "}"
                + "flag_group {"
                + "  expand_if_false: 'missing'"
                + "  flag: '-B%{missing}'"
                + "}",
            createVariables()))
        .isEmpty();
  }

  @Test
  public void testExpandIfTrueExpandsIfOne() throws Exception {
    assertThat(
            getCommandLineForFlagGroups(
                "flag_group {"
                    + "  expand_if_true: 'struct.bool'"
                    + "  flag: '-A%{struct.foo}'"
                    + "  flag: '-B%{struct.bar}'"
                    + "}"
                    + "flag_group {"
                    + "  expand_if_false: 'struct.bool'"
                    + "  flag: '-X%{struct.foo}'"
                    + "  flag: '-Y%{struct.bar}'"
                    + "}",
                createStructureVariables(
                    "struct",
                    new Variables.StructureBuilder()
                        .addField("bool", new IntegerValue(1))
                        .addField("foo", "fooValue")
                        .addField("bar", "barValue"))))
        .containsExactly("-AfooValue", "-BbarValue");
  }

  @Test
  public void testExpandIfTrueExpandsIfZero() throws Exception {
    assertThat(
            getCommandLineForFlagGroups(
                "flag_group {"
                    + "  expand_if_true: 'struct.bool'"
                    + "  flag: '-A%{struct.foo}'"
                    + "  flag: '-B%{struct.bar}'"
                    + "}"
                    + "flag_group {"
                    + "  expand_if_false: 'struct.bool'"
                    + "  flag: '-X%{struct.foo}'"
                    + "  flag: '-Y%{struct.bar}'"
                    + "}",
                createStructureVariables(
                    "struct",
                    new Variables.StructureBuilder()
                        .addField("bool", new IntegerValue(0))
                        .addField("foo", "fooValue")
                        .addField("bar", "barValue"))))
        .containsExactly("-XfooValue", "-YbarValue");
  }

  @Test
  public void testExpandIfEqual() throws Exception {
    assertThat(
        getCommandLineForFlagGroups(
            "flag_group {"
                + "  expand_if_equal: { variable: 'var' value: 'equal_value' }"
                + "  flag: '-foo_%{var}'"
                + "}"
                + "flag_group {"
                + "  expand_if_equal: { variable: 'var' value: 'non_equal_value' }"
                + "  flag: '-bar_%{var}'"
                + "}"
                + "flag_group {"
                + "  expand_if_equal: { variable: 'non_existing_var' value: 'non_existing' }"
                + "  flag: '-baz_%{non_existing_var}'"
                + "}",
            createVariables("var", "equal_value")))
        .containsExactly("-foo_equal_value");
  }

  @Test
  public void testLegacyListVariableExpansion() throws Exception {
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
  public void testListVariableExpansion() throws Exception {
    assertThat(
            getCommandLineForFlagGroups(
                "flag_group { iterate_over: 'v' flag: '%{v}' }",
                createVariables("v", "1", "v", "2")))
        .containsExactly("1", "2");
  }

  @Test
  public void testListVariableExpansionMixedWithNonListVariable() throws Exception {
    assertThat(
            getCommandLineForFlagGroups(
                "flag_group { iterate_over: 'v1' flag: '%{v1} %{v2}' }",
                createVariables("v1", "a1", "v1", "a2", "v2", "b")))
        .containsExactly("a1 b", "a2 b");
  }

  @Test
  public void testNestedListVariableExpansion() throws Exception {
    assertThat(
            getCommandLineForFlagGroups(
                "flag_group {"
                    + "  iterate_over: 'v1'"
                    + "  flag_group {"
                    + "    iterate_over: 'v2'"
                    + "    flag: '%{v1} %{v2}'"
                    + "  }"
                    + "}",
                createVariables("v1", "a1", "v1", "a2", "v2", "b1", "v2", "b2")))
        .containsExactly("a1 b1", "a1 b2", "a2 b1", "a2 b2");
  }

  @Test
  public void testListVariableExpansionMixedWithImplicitlyAccessedListVariableFails()
      throws Exception {
    assertThat(
            getFlagGroupsExpansionError(
                "flag_group { iterate_over: 'v1' flag: '%{v1} %{v2}' }",
                createVariables("v1", "a1", "v1", "a2", "v2", "b1", "v2", "b2")))
        .contains("Cannot expand variable 'v2': expected string, found sequence");
  }

  @Test
  public void testListVariableExpansionMixedWithImplicitlyAccessedListVariableWithinFlagGroupWorks()
      throws Exception {
    assertThat(
            getCommandLineForFlagGroups(
                "flag_group {"
                    + "  iterate_over: 'v1'"
                    + "  flag_group {"
                    + "    flag: '-A%{v1} -B%{v2}'"
                    + "  }"
                    + "}",
                createVariables("v1", "a1", "v1", "a2", "v2", "b1", "v2", "b2")))
        .containsExactly("-Aa1 -Bb1", "-Aa1 -Bb2", "-Aa2 -Bb1", "-Aa2 -Bb2");
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
  
  private VariableValueBuilder createNestedSequence(int depth, int count, String prefix) {
    if (depth == 0) {
      StringSequenceBuilder builder = new StringSequenceBuilder();
      for (int i = 0; i < count; ++i) {
        String value = prefix + i;
        builder.addValue(value);
      }
      return builder;
    } else {
      SequenceBuilder builder = new SequenceBuilder();
      for (int i = 0; i < count; ++i) {
        String value = prefix + i;
        builder.addValue(createNestedSequence(depth - 1, count, value));
      }
      return builder;
    }
  }

  private Variables createNestedVariables(String name, int depth, int count) {
    return new Variables.Builder()
        .addCustomBuiltVariable(name, createNestedSequence(depth, count, ""))
        .build();
  }

  @Test
  public void testFlagTreeVariableExpansion() throws Exception {
    String nestedGroup =
        "flag_group {"
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
  public void testDefaultFeatures() throws Exception {
    CcToolchainFeatures features =
        buildFeatures("feature { name: 'a' }", "feature { name: 'b' enabled: true }");
    assertThat(features.getDefaultFeatures()).containsExactly("b");
  }

  @Test
  public void testWithFeature_OneSetOneFeature() throws Exception {
    CcToolchainFeatures features =
        buildFeatures(
            "feature {",
            "  name: 'a'",
            "  flag_set {",
            "    with_feature {feature: 'b'}",
            "    action: 'c++-compile'",
            "    flag_group {",
            "      flag: 'dummy_flag'",
            "    }",
            "  }",
            "}",
            "feature {name: 'b'}");
    assertThat(
            features
                .getFeatureConfiguration("a", "b")
                .getCommandLine(CppCompileAction.CPP_COMPILE, createVariables()))
        .containsExactly("dummy_flag");
    assertThat(
            features
                .getFeatureConfiguration("a")
                .getCommandLine(CppCompileAction.CPP_COMPILE, createVariables()))
        .doesNotContain("dummy_flag");
  }

  @Test
  public void testWithFeature_OneSetMultipleFeatures() throws Exception {
    CcToolchainFeatures features =
        buildFeatures(
            "feature {",
            "  name: 'a'",
            "  flag_set {",
            "    with_feature {feature: 'b', feature: 'c'}",
            "    action: 'c++-compile'",
            "    flag_group {",
            "      flag: 'dummy_flag'",
            "    }",
            "  }",
            "}",
            "feature {name: 'b'}",
            "feature {name: 'c'}");
    assertThat(
            features
                .getFeatureConfiguration("a", "b", "c")
                .getCommandLine(CppCompileAction.CPP_COMPILE, createVariables()))
        .containsExactly("dummy_flag");
    assertThat(
            features
                .getFeatureConfiguration("a", "b")
                .getCommandLine(CppCompileAction.CPP_COMPILE, createVariables()))
        .doesNotContain("dummy_flag");
    assertThat(
            features
                .getFeatureConfiguration("a")
                .getCommandLine(CppCompileAction.CPP_COMPILE, createVariables()))
        .doesNotContain("dummy_flag");
  }

  @Test
  public void testWithFeature_MulipleSetsMultipleFeatures() throws Exception {
    CcToolchainFeatures features =
        buildFeatures(
            "feature {",
            "  name: 'a'",
            "  flag_set {",
            "    with_feature {feature: 'b1', feature: 'c1'}",
            "    with_feature {feature: 'b2', feature: 'c2'}",
            "    action: 'c++-compile'",
            "    flag_group {",
            "      flag: 'dummy_flag'",
            "    }",
            "  }",
            "}",
            "feature {name: 'b1'}",
            "feature {name: 'c1'}",
            "feature {name: 'b2'}",
            "feature {name: 'c2'}");
    assertThat(
            features
                .getFeatureConfiguration("a", "b1", "c1", "b2", "c2")
                .getCommandLine(CppCompileAction.CPP_COMPILE, createVariables()))
        .containsExactly("dummy_flag");
    assertThat(
            features
                .getFeatureConfiguration("a", "b1", "c1")
                .getCommandLine(CppCompileAction.CPP_COMPILE, createVariables()))
        .containsExactly("dummy_flag");
    assertThat(
            features
                .getFeatureConfiguration("a", "b1", "b2")
                .getCommandLine(CppCompileAction.CPP_COMPILE, createVariables()))
        .doesNotContain("dummy_flag");
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
    PathFragment crosstoolPath = PathFragment.create("crosstool/");
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

    PathFragment crosstoolPath = PathFragment.create("crosstool/");

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

    PathFragment crosstoolPath = PathFragment.create("crosstool/");

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

  @Test
  public void testFlagsFromActionConfig() throws Exception {
    FeatureConfiguration featureConfiguration =
        buildFeatures(
                "action_config {",
                "  config_name: 'c++-compile'",
                "  action_name: 'c++-compile'",
                "  flag_set {",
                "    flag_group {flag: 'foo'}",
                "  }",
                "}")
            .getFeatureConfiguration("c++-compile");
    List<String> commandLine =
        featureConfiguration.getCommandLine("c++-compile", createVariables());
    assertThat(commandLine).contains("foo");
    ;
  }

  @Test
  public void testErrorForFlagFromActionConfigWithSpecifiedAction() throws Exception {
    try {
      buildFeatures(
              "action_config {",
              "  config_name: 'c++-compile'",
              "  action_name: 'c++-compile'",
              "  flag_set {",
              "    action: 'c++-compile'",
              "    flag_group {flag: 'foo'}",
              "  }",
              "}")
          .getFeatureConfiguration("c++-compile");
      fail("Should throw InvalidConfigurationException");
    } catch (InvalidConfigurationException e) {
      assertThat(e.getMessage())
          .contains(String.format(ActionConfig.FLAG_SET_WITH_ACTION_ERROR, "c++-compile"));
    }
  }

  @Test
  public void testLibraryToLinkValue() {
    assertThat(
            LibraryToLinkValue.forDynamicLibrary("foo")
                .getFieldValue("LibraryToLinkValue", LibraryToLinkValue.NAME_FIELD_NAME)
                .getStringValue(LibraryToLinkValue.NAME_FIELD_NAME))
        .isEqualTo("foo");
    assertThat(
            LibraryToLinkValue.forDynamicLibrary("foo")
                .getFieldValue("LibraryToLinkValue", LibraryToLinkValue.OBJECT_FILES_FIELD_NAME))
        .isNull();

    assertThat(
            LibraryToLinkValue.forObjectFileGroup(ImmutableList.of("foo", "bar"), false)
                .getFieldValue("LibraryToLinkValue", LibraryToLinkValue.NAME_FIELD_NAME))
        .isNull();
    Iterable<? extends VariableValue> objects =
        LibraryToLinkValue.forObjectFileGroup(ImmutableList.of("foo", "bar"), false)
            .getFieldValue("LibraryToLinkValue", LibraryToLinkValue.OBJECT_FILES_FIELD_NAME)
            .getSequenceValue(LibraryToLinkValue.OBJECT_FILES_FIELD_NAME);
    ImmutableList.Builder<String> objectNames = ImmutableList.builder();
    for (VariableValue object : objects) {
      objectNames.add(object.getStringValue("name"));
    }
    assertThat(objectNames.build()).containsExactly("foo", "bar");
  }

  @Test
  public void testProvidesCollision() throws Exception {
    try {
      buildFeatures(
          "feature {",
          " name: 'a'",
          " provides: 'provides_string'",
          "}",
          "feature {",
          " name: 'b'",
          " provides: 'provides_string'",
          "}").getFeatureConfiguration("a", "b");
      fail("Should throw CollidingProvidesException on collision, instead did not throw.");
    } catch (Exception e) {
      assertThat(e).hasMessageThat().contains("a b");
    }
  }
}
