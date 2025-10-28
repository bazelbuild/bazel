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
import static org.junit.Assert.assertThrows;

import com.google.common.base.Preconditions;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ListMultimap;
import com.google.devtools.build.lib.actions.InputMetadataProvider;
import com.google.devtools.build.lib.actions.PathMapper;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.util.ResourceLoader;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.ActionConfig;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.ExpansionException;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.CcToolchainVariables.Sequence;
import com.google.devtools.build.lib.rules.cpp.CcToolchainVariables.StringValue;
import com.google.devtools.build.lib.rules.cpp.CcToolchainVariables.VariableValue;
import com.google.devtools.build.lib.rules.cpp.CcToolchainVariables.VariableValueAdapter;
import com.google.devtools.build.lib.skyframe.serialization.testutils.RoundTripping;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicInteger;
import javax.annotation.Nullable;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for toolchain features. */
@RunWith(JUnit4.class)
public final class CcToolchainFeaturesTest extends BuildViewTestCase {

  private final AtomicInteger starlarkConfigCounter = new AtomicInteger(0);

  private void loadCcToolchainConfigLib() throws Exception {
    scratch.appendFile("tools/cpp/BUILD", "");
    scratch.overwriteFile(
        "tools/cpp/cc_toolchain_config_lib.bzl",
        ResourceLoader.readFromResources(
            TestConstants.RULES_CC_REPOSITORY_EXECROOT + "cc/cc_toolchain_config_lib.bzl"));
  }

  private CcToolchainFeatures buildFeatures(String... content) throws Exception {
    loadCcToolchainConfigLib();
    String packageName = "crosstool" + starlarkConfigCounter.getAndIncrement();
    scratch.overwriteFile(
        packageName + "/crosstool.bzl",
        "load(",
        "    '//tools/cpp:cc_toolchain_config_lib.bzl',",
        "    'action_config',",
        "    'artifact_name_pattern',",
        "    'env_entry',",
        "    'env_set',",
        "    'feature',",
        "    'feature_set',",
        "    'flag_group',",
        "    'flag_set',",
        "    'tool',",
        "    'variable_with_value',",
        "    'with_feature_set',",
        ")",
        "load('@rules_cc//cc/toolchains:cc_toolchain_config_info.bzl',"
            + " 'CcToolchainConfigInfo')",
        "load('@rules_cc//cc/common:cc_common.bzl', 'cc_common')",
        "",
        "def _impl(ctx):",
        "    return cc_common.create_cc_toolchain_config_info(",
        "        ctx = ctx,",
        String.join("\n", content) + ",",
        "        toolchain_identifier = 'toolchain',",
        "        host_system_name = 'host',",
        "        target_system_name = 'target',",
        "        target_cpu = 'cpu',",
        "        target_libc = 'libc',",
        "        compiler = 'compiler',",
        "    )",
        "",
        "cc_toolchain_config_rule = rule(implementation = _impl, provides ="
            + " [CcToolchainConfigInfo])");
    scratch.overwriteFile("bazel_internal/test_rules/cc/BUILD");
    scratch.overwriteFile(
        "bazel_internal/test_rules/cc/ctf_rule.bzl",
        """
        load('@rules_cc//cc/toolchains:cc_toolchain_config_info.bzl', 'CcToolchainConfigInfo')
        load('@rules_cc//cc/common:cc_common.bzl', 'cc_common')
        MyInfo = provider()
        def _impl(ctx):
          return [MyInfo(f = cc_common.cc_toolchain_features(
                    toolchain_config_info = ctx.attr.config[CcToolchainConfigInfo],
                    tools_directory = "crosstool",
                  ))]
        cc_toolchain_features = rule(_impl, attrs = {"config":attr.label()})
        """);
    scratch.overwriteFile(
        packageName + "/BUILD",
        "load(':crosstool.bzl', 'cc_toolchain_config_rule')",
        "load('//bazel_internal/test_rules/cc:ctf_rule.bzl', 'cc_toolchain_features')",
        "cc_toolchain_features(name = 'f', config = ':r')",
        "cc_toolchain_config_rule(name = 'r')");

    ConfiguredTarget target = getConfiguredTarget("//" + packageName + ":f");
    assertThat(target).isNotNull();
    return (CcToolchainFeatures) getStarlarkProvider(target, "MyInfo").getValue("f");
  }

  /**
   * Creates a {@code Variables} configuration from a list of key/value pairs.
   *
   * <p>If there are multiple entries with the same key, the variable will be treated as sequence
   * type.
   */
  private static CcToolchainVariables createVariables(String... entries) {
    if (entries.length % 2 != 0) {
      throw new IllegalArgumentException(
          "createVariables takes an even number of arguments (key/value pairs)");
    }
    ListMultimap<String, String> entryMap = ArrayListMultimap.create();
    for (int i = 0; i < entries.length; i += 2) {
      entryMap.put(entries[i], entries[i + 1]);
    }
    CcToolchainVariables.Builder variables = CcToolchainVariables.builder();
    for (String name : entryMap.keySet()) {
      List<String> value = entryMap.get(name);
      if (value.size() == 1) {
        variables.addVariable(name, value.get(0));
      } else {
        variables.addStringSequenceVariable(name, ImmutableList.copyOf(value));
      }
    }
    return variables.build();
  }

  private static ImmutableSet<String> getEnabledFeatures(
      CcToolchainFeatures features, String... requestedFeatures) throws Exception {
    FeatureConfiguration configuration =
        features.getFeatureConfiguration(ImmutableSet.copyOf(requestedFeatures));
    ImmutableSet.Builder<String> enabledFeatures = ImmutableSet.builder();
    for (String feature : features.getActivatableNames()) {
      if (configuration.isEnabled(feature)) {
        enabledFeatures.add(feature);
      }
    }
    return enabledFeatures.build();
  }

  @Test
  public void testFeatureConfigurationCodec() throws Exception {
    FeatureConfiguration emptyConfiguration =
        FeatureConfiguration.intern(
            buildFeatures("features=[feature(name = 'no_legacy_features')]")
                .getFeatureConfiguration(ImmutableSet.of()));
    FeatureConfiguration emptyFeatures =
        buildFeatures(
                "features=[feature(name = 'no_legacy_features'),feature(name='a'),"
                    + " feature(name='b')]")
            .getFeatureConfiguration(ImmutableSet.of("a", "b"));
    FeatureConfiguration featuresWithFlags =
        buildFeatures(
                "features = [",
                "    feature(name = 'no_legacy_features'),",
                "    feature(",
                "        name = 'a',",
                "        flag_sets = [",
                "            flag_set(",
                "                actions = ['action-a'],",
                "                flag_groups = [flag_group(flags = ['flag-a'])],",
                "            ),",
                "            flag_set(",
                "                actions = ['action-b'],",
                "                flag_groups = [flag_group(flags = ['flag-b'])],",
                "            ),",
                "        ],",
                "    ),",
                "    feature(",
                "        name = 'b',",
                "        flag_sets = [",
                "            flag_set(",
                "                actions = ['action-c'],",
                "                flag_groups = [flag_group(flags = ['flag-c'])],",
                "            ),",
                "        ],",
                "    ),",
                "]")
            .getFeatureConfiguration(ImmutableSet.of("a", "b"));
    FeatureConfiguration featureWithEnvSet =
        buildFeatures(
                "features = [",
                "    feature(name = 'no_legacy_features'),",
                "    feature(",
                "        name = 'a',",
                "        env_sets = [",
                "            env_set(",
                "                actions = ['action-a'],",
                "                env_entries = [",
                "                    env_entry(key = 'foo', value = 'bar'),",
                "                    env_entry(key = 'baz', value = 'zee'),",
                "                ],",
                "            ),",
                "        ],",
                "    ),",
                "]")
            .getFeatureConfiguration(ImmutableSet.of("a"));

    new SerializationTester(emptyConfiguration, emptyFeatures, featuresWithFlags, featureWithEnvSet)
        .runTests();
  }

  @Test
  public void testUnconditionalFeature() throws Exception {
    assertThat(
            buildFeatures("features = []")
                .getFeatureConfiguration(ImmutableSet.of("a"))
                .isEnabled("a"))
        .isFalse();
    assertThat(
            buildFeatures("features = [feature(name = 'a')]")
                .getFeatureConfiguration(ImmutableSet.of("b"))
                .isEnabled("a"))
        .isFalse();
    assertThat(
            buildFeatures("features = [feature(name = 'a')]")
                .getFeatureConfiguration(ImmutableSet.of("a"))
                .isEnabled("a"))
        .isTrue();
  }

  @Test
  public void testUnsupportedAction() throws Exception {
    FeatureConfiguration configuration =
        buildFeatures("features = []").getFeatureConfiguration(ImmutableSet.of());
    assertThat(configuration.getCommandLine("invalid-action", createVariables())).isEmpty();
  }

  @Test
  public void testFlagOrderEqualsSpecOrder() throws Exception {
    FeatureConfiguration configuration =
        buildFeatures(
                "features = [",
                "    feature(",
                "        name = 'a',",
                "        flag_sets = [",
                "            flag_set(",
                "                actions = ['c++-compile'],",
                "                flag_groups = [flag_group(flags = ['-a-c++-compile'])],",
                "            ),",
                "            flag_set(",
                "                actions = ['link'],",
                "                flag_groups = [flag_group(flags = ['-a-c++-compile'])],",
                "            ),",
                "        ],",
                "    ),",
                "    feature(",
                "        name = 'b',",
                "        flag_sets = [",
                "            flag_set(",
                "                actions = ['c++-compile'],",
                "                flag_groups = [flag_group(flags = ['-b-c++-compile'])],",
                "            ),",
                "            flag_set(",
                "                actions = ['link'],",
                "                flag_groups = [flag_group(flags = ['-b-link'])],",
                "            ),",
                "        ],",
                "    ),",
                "]")
            .getFeatureConfiguration(ImmutableSet.of("a", "b"));
    List<String> commandLine =
        configuration.getCommandLine(CppActionNames.CPP_COMPILE, createVariables());
    assertThat(commandLine).containsExactly("-a-c++-compile", "-b-c++-compile").inOrder();
  }

  @Test
  public void testEnvVars() throws Exception {
    FeatureConfiguration configuration =
        buildFeatures(
                "features = [",
                "    feature(",
                "        name = 'a',",
                "        env_sets = [",
                "            env_set(",
                "                actions = ['c++-compile'],",
                "                env_entries = [",
                "                    env_entry(key = 'foo', value = 'bar'),",
                "                    env_entry(key = 'cat', value = 'meow'),",
                "                ],",
                "            ),",
                "        ],",
                "        flag_sets = [",
                "            flag_set(",
                "                actions = ['c++-compile'],",
                "                flag_groups = [flag_group(flags = ['-a-c++-compile'])],",
                "            ),",
                "        ],",
                "    ),",
                "    feature(",
                "        name = 'b',",
                "        env_sets = [",
                "            env_set(",
                "                actions = ['c++-compile'],",
                "                env_entries = [env_entry(key = 'dog', value = 'woof')],",
                "            ),",
                "            env_set(",
                "                actions = ['c++-compile'],",
                "                with_features = [with_feature_set(features = ['d'])],",
                "                env_entries = [env_entry(key = 'withFeature', value = 'value1')],",
                "            ),",
                "            env_set(",
                "                actions = ['c++-compile'],",
                "                with_features = [with_feature_set(features = ['e'])],",
                "                env_entries = [env_entry(key = 'withoutFeature', value ="
                    + " 'value2')],",
                "            ),",
                "            env_set(",
                "                actions = ['c++-compile'],",
                "                with_features = [with_feature_set(not_features = ['f'])],",
                "                env_entries = [env_entry(key = 'withNotFeature', value ="
                    + " 'value3')],",
                "            ),",
                "            env_set(",
                "                actions = ['c++-compile'],",
                "                with_features = [with_feature_set(not_features = ['g'])],",
                "                env_entries = [env_entry(key = 'withoutNotFeature', value ="
                    + " 'value4')],",
                "            ),",
                "        ],",
                "    ),",
                "    feature(",
                "        name = 'c',",
                "        env_sets = [",
                "            env_set(",
                "                actions = ['c++-compile'],",
                "                env_entries = [env_entry(key = 'doNotInclude', value ="
                    + " 'doNotIncludePlease')],",
                "            ),",
                "        ],",
                "    ),",
                "    feature(name = 'd'),",
                "    feature(name = 'e'),",
                "    feature(name = 'f'),",
                "    feature(name = 'g'),",
                "]")
            .getFeatureConfiguration(ImmutableSet.of("a", "b", "d", "f"));
    ImmutableMap<String, String> env =
        configuration.getEnvironmentVariables(
            CppActionNames.CPP_COMPILE, createVariables(), PathMapper.NOOP);
    assertThat(env)
        .containsExactly(
            "foo", "bar", "cat", "meow", "dog", "woof",
            "withFeature", "value1", "withoutNotFeature", "value4")
        .inOrder();
    assertThat(env).doesNotContainEntry("withoutFeature", "value2");
    assertThat(env).doesNotContainEntry("withNotFeature", "value3");
    assertThat(env).doesNotContainEntry("doNotInclude", "doNotIncludePlease");
  }

  @Test
  public void testEnvVarsWithMissingVariableIsNotExpanded() throws Exception {
    FeatureConfiguration configuration =
        buildFeatures(
                "features = [",
                "    feature(",
                "        name = 'a',",
                "        env_sets = [",
                "            env_set(",
                "                actions = ['c++-compile'],",
                "                env_entries = [",
                "                    env_entry(key = 'foo', value = 'bar', expand_if_available"
                    + " = 'v')",
                "                ],",
                "            ),",
                "        ],",
                "    ),",
                "]")
            .getFeatureConfiguration(ImmutableSet.of("a"));

    ImmutableMap<String, String> env =
        configuration.getEnvironmentVariables(
            CppActionNames.CPP_COMPILE, createVariables(), PathMapper.NOOP);

    assertThat(env).doesNotContainEntry("foo", "bar");
  }

  @Test
  public void testEnvVarsWithAllVariablesPresentAreExpanded() throws Exception {
    FeatureConfiguration configuration =
        buildFeatures(
                "features = [",
                "    feature(",
                "        name = 'a',",
                "        env_sets = [",
                "            env_set(",
                "                actions = ['c++-compile'],",
                "                env_entries = [",
                "                    env_entry(key = 'foo', value = 'bar', expand_if_available"
                    + " = 'v')",
                "                ],",
                "            ),",
                "        ],",
                "    ),",
                "]")
            .getFeatureConfiguration(ImmutableSet.of("a"));

    ImmutableMap<String, String> env =
        configuration.getEnvironmentVariables(
            CppActionNames.CPP_COMPILE, createVariables("v", "1"), PathMapper.NOOP);

    assertThat(env).containsExactly("foo", "bar").inOrder();
  }

  @Test
  public void testEnvVarsWithAllVariablesPresentAreExpandedWithVariableExpansion()
      throws Exception {
    FeatureConfiguration configuration =
        buildFeatures(
                "features = [",
                "    feature(",
                "        name = 'a',",
                "        env_sets = [",
                "            env_set(",
                "                actions = ['c++-compile'],",
                "                env_entries = [",
                "                    env_entry(key = 'foo', value = '%{v}', expand_if_available"
                    + " = 'v')",
                "                ],",
                "            ),",
                "        ],",
                "    ),",
                "]")
            .getFeatureConfiguration(ImmutableSet.of("a"));

    ImmutableMap<String, String> env =
        configuration.getEnvironmentVariables(
            CppActionNames.CPP_COMPILE, createVariables("v", "1"), PathMapper.NOOP);

    assertThat(env).containsExactly("foo", "1").inOrder();
  }

  private String getExpansionOfFlag(String value) throws Exception {
    return getExpansionOfFlag(value, createVariables());
  }

  private String getExpansionOfFlag(String value, CcToolchainVariables variables) throws Exception {
    return getCommandLineForFlag(value, variables).get(0);
  }

  private List<String> getCommandLineForFlagGroups(String groups, CcToolchainVariables variables)
      throws Exception {
    FeatureConfiguration configuration =
        buildFeatures(
                "features = [",
                "   feature(name = 'no_legacy_features'),",
                "   feature(",
                "    name = 'a',",
                "    flag_sets = [",
                "        flag_set(",
                "            actions = ['c++-compile'],",
                "            flag_groups = [" + groups + "],",
                "        ),",
                "    ],",
                ")]")
            .getFeatureConfiguration(ImmutableSet.of("a"));
    return configuration.getCommandLine(CppActionNames.CPP_COMPILE, variables);
  }

  private List<String> getCommandLineForFlag(String value, CcToolchainVariables variables)
      throws Exception {
    return getCommandLineForFlagGroups("flag_group(flags = ['" + value + "'])", variables);
  }

  private String getFlagParsingError(String value) {
    return assertThrows(AssertionError.class, () -> getExpansionOfFlag(value)).getMessage();
  }

  private String getFlagExpansionError(String value, CcToolchainVariables variables) {
    return assertThrows(ExpansionException.class, () -> getExpansionOfFlag(value, variables))
        .getMessage();
  }

  private String getFlagGroupsExpansionError(String flagGroups, CcToolchainVariables variables) {
    return assertThrows(
            ExpansionException.class, () -> getCommandLineForFlagGroups(flagGroups, variables))
        .getMessage();
  }

  @Test
  public void testVariableExpansion() throws Exception {
    assertThat(getExpansionOfFlag("%%")).isEqualTo("%");
    assertThat(getExpansionOfFlag("%% a %% b %%")).isEqualTo("% a % b %");
    assertThat(getExpansionOfFlag("%%{var}")).isEqualTo("%{var}");
    assertThat(getExpansionOfFlag("%{v}", createVariables("v", "<flag>"))).isEqualTo("<flag>");
    assertThat(getExpansionOfFlag(" %{v1} %{v2} ", createVariables("v1", "1", "v2", "2")))
        .isEqualTo(" 1 2 ");
    assertThat(getFlagParsingError("%"))
        .contains("expected '{' at position 1 while parsing a flag containing '%'");
    assertThat(getFlagParsingError("% "))
        .contains("expected '{' at position 1 while parsing a flag containing '% '");
    assertThat(getFlagParsingError("%{")).contains("expected variable name");
    assertThat(getFlagParsingError("%{}")).contains("expected variable name");
    assertThat(
            getCommandLineForFlagGroups(
                "flag_group(iterate_over = 'v', flags = ['%{v}'])",
                CcToolchainVariables.builder()
                    .addStringSequenceVariable("v", ImmutableList.of())
                    .build()))
        .isEmpty();
    assertThat(getFlagExpansionError("%{v}", createVariables()))
        .contains("Invalid toolchain configuration: Cannot find variable named 'v'");
  }

  private static CcToolchainVariables createStructureSequenceVariables(
      String name, VariableValue... values) {
    return CcToolchainVariables.builder().addVariable(name, ImmutableList.copyOf(values)).build();
  }

  /**
   * Single structure value. Be careful not to create sequences of single structures, as the memory
   * overhead is prohibitively big.
   */
  @Immutable
  private static final class StructureValue extends VariableValueAdapter {
    private static final String STRUCTURE_VARIABLE_TYPE_NAME = "structure";

    private final ImmutableMap<String, VariableValue> value;

    private StructureValue(ImmutableMap<String, VariableValue> value) {
      this.value = value;
    }

    @Nullable
    @Override
    public VariableValue getFieldValue(
        String variableName,
        String field,
        @Nullable InputMetadataProvider inputMetadataProvider,
        PathMapper pathMapper,
        boolean throwOnMissingVariable) {
      return value.getOrDefault(field, null);
    }

    @Override
    public String getVariableTypeName() {
      return STRUCTURE_VARIABLE_TYPE_NAME;
    }

    @Override
    public boolean isTruthy() {
      return !value.isEmpty();
    }

    @Override
    public boolean equals(Object other) {
      if (!(other instanceof StructureValue)) {
        return false;
      }
      if (this == other) {
        return true;
      }
      return Objects.equals(value, ((StructureValue) other).value);
    }

    @Override
    public int hashCode() {
      return value.hashCode();
    }
  }

  /** Builder for StructureValue. */
  public static class StructureBuilder {
    private final ImmutableMap.Builder<String, VariableValue> fields = ImmutableMap.builder();

    /** Adds a field to the structure. */
    @CanIgnoreReturnValue
    public StructureBuilder addField(String name, VariableValue value) {
      fields.put(name, value);
      return this;
    }

    /** Adds a field to the structure. */
    @CanIgnoreReturnValue
    public StructureBuilder addField(String name, StructureBuilder valueBuilder) {
      Preconditions.checkArgument(
          valueBuilder != null,
          "Cannot use null builder to get a field value for field '%s'",
          name);
      fields.put(name, valueBuilder.build());
      return this;
    }

    /** Adds a field to the structure. */
    @CanIgnoreReturnValue
    public StructureBuilder addField(String name, String value) {
      fields.put(name, new StringValue(value));
      return this;
    }

    /** Adds a field to the structure. */
    @CanIgnoreReturnValue
    public StructureBuilder addField(String name, ImmutableList<String> values) {
      fields.put(name, new Sequence(values));
      return this;
    }

    /** Returns an immutable structure. */
    public StructureValue build() {
      return new StructureValue(fields.buildOrThrow());
    }
  }

  private static CcToolchainVariables createStructureVariables(
      String name, StructureBuilder value) {
    return CcToolchainVariables.builder().addVariable(name, value.build()).build();
  }

  @Test
  public void testSimpleStructureVariableExpansion() throws Exception {
    assertThat(
            getCommandLineForFlagGroups(
                "flag_group(flags = ['-A%{struct.foo}', '-B%{struct.bar}'])",
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
                "flag_group(flags = ['-A%{struct.foo.bar}'])",
                createStructureVariables(
                    "struct",
                    new StructureBuilder()
                        .addField("foo", new StructureBuilder().addField("bar", "fooBarValue")))))
        .containsExactly("-AfooBarValue");
  }

  @Test
  public void testAccessingStructureAsStringFails() {
    assertThat(
            getFlagGroupsExpansionError(
                "flag_group(flags = ['-A%{struct}'])",
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
  public void testAccessingStringValueAsStructureFails() {
    assertThat(
            getFlagGroupsExpansionError(
                "flag_group(flags = ['-A%{stringVar.foo}'])",
                createVariables("stringVar", "stringVarValue")))
        .isEqualTo(
            "Invalid toolchain configuration: Cannot expand variable 'stringVar.foo': variable "
                + "'stringVar' is string, expected structure");
  }

  @Test
  public void testAccessingSequenceAsStructureFails() {
    assertThat(
            getFlagGroupsExpansionError(
                "flag_group(flags = ['-A%{sequence.foo}'])",
                createVariables("sequence", "foo1", "sequence", "foo2")))
        .isEqualTo(
            "Invalid toolchain configuration: Cannot expand variable 'sequence.foo': variable "
                + "'sequence' is sequence, expected structure");
  }

  @Test
  public void testAccessingMissingStructureFieldFails() {
    assertThat(
            getFlagGroupsExpansionError(
                "flag_group(flags = ['-A%{struct.missing}'])",
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
                "flag_group(iterate_over = 'structs', flags = ['-A%{structs.foo}'])",
                createStructureSequenceVariables(
                    "structs",
                    new StructureBuilder().addField("foo", "foo1Value").build(),
                    new StructureBuilder().addField("foo", "foo2Value").build())))
        .containsExactly("-Afoo1Value", "-Afoo2Value");
  }

  @Test
  public void testStructureOfSequencesExpansion() throws Exception {
    assertThat(
            getCommandLineForFlagGroups(
                "flag_group("
                    + "  iterate_over = 'struct.sequences',"
                    + "  flags = ['-A%{struct.sequences.foo}']"
                    + ")",
                createStructureVariables(
                    "struct",
                    new StructureBuilder()
                        .addField(
                            "sequences",
                            new Sequence(
                                ImmutableList.of(
                                    new StructureBuilder().addField("foo", "foo1Value").build(),
                                    new StructureBuilder()
                                        .addField("foo", "foo2Value")
                                        .build()))))))
        .containsExactly("-Afoo1Value", "-Afoo2Value");
  }

  @Test
  public void testDottedNamesNotAlwaysMeanStructures() throws Exception {
    assertThat(
            getCommandLineForFlagGroups(
                "flag_group("
                    + "  iterate_over = 'struct.sequence',"
                    + "  flag_groups = [flag_group("
                    + "    iterate_over = 'other_sequence',"
                    + "    flag_groups = [flag_group("
                    + "      flags = ['-A%{struct.sequence} -B%{other_sequence}']"
                    + "    )]"
                    + "  )]"
                    + ")",
                CcToolchainVariables.builder()
                    .addVariable(
                        "struct",
                        new StructureBuilder()
                            .addField("sequence", ImmutableList.of("first", "second"))
                            .build())
                    .addStringSequenceVariable("other_sequence", ImmutableList.of("foo", "bar"))
                    .build()))
        .containsExactly("-Afirst -Bfoo", "-Afirst -Bbar", "-Asecond -Bfoo", "-Asecond -Bbar");
  }

  @Test
  public void testExpandIfAllAvailableWithStructsExpandsIfPresent() throws Exception {
    assertThat(
            getCommandLineForFlagGroups(
                "flag_group("
                    + "  expand_if_available = 'struct',"
                    + "  flags = ['-A%{struct.foo}', '-B%{struct.bar}']"
                    + ")",
                createStructureVariables(
                    "struct",
                    new StructureBuilder()
                        .addField("foo", "fooValue")
                        .addField("bar", "barValue"))))
        .containsExactly("-AfooValue", "-BbarValue");
  }

  @Test
  public void testExpandIfAllAvailableWithStructsDoesntExpandIfMissing() throws Exception {
    assertThat(
            getCommandLineForFlagGroups(
                "flag_group("
                    + "  expand_if_available = 'nonexistent',"
                    + "  flags = ['-A%{struct.foo}','-B%{struct.bar}']"
                    + ")",
                createStructureVariables(
                    "struct",
                    new StructureBuilder()
                        .addField("foo", "fooValue")
                        .addField("bar", "barValue"))))
        .isEmpty();
  }

  @Test
  public void testExpandIfAllAvailableWithStructsDoesntCrashIfMissing() throws Exception {
    assertThat(
            getCommandLineForFlagGroups(
                "flag_group("
                    + "  expand_if_available = 'nonexistent',"
                    + "  flags = ['-A%{nonexistent.foo}','-B%{nonexistent.bar}']"
                    + ")",
                createVariables()))
        .isEmpty();
  }

  @Test
  public void testExpandIfAllAvailableWithStructFieldDoesntCrashIfMissing() throws Exception {
    assertThat(
            getCommandLineForFlagGroups(
                "flag_group("
                    + "  expand_if_available = 'nonexistent.nonexistant_field',"
                    + "  flags = ['-A%{nonexistent.foo}','-B%{nonexistent.bar}']"
                    + ")",
                createVariables()))
        .isEmpty();
  }

  @Test
  public void testExpandIfAllAvailableWithStructFieldExpandsIfPresent() throws Exception {
    assertThat(
            getCommandLineForFlagGroups(
                "flag_group("
                    + "  expand_if_available = 'struct.foo',"
                    + "  flags = ['-A%{struct.foo}','-B%{struct.bar}']"
                    + ")",
                createStructureVariables(
                    "struct",
                    new StructureBuilder()
                        .addField("foo", "fooValue")
                        .addField("bar", "barValue"))))
        .containsExactly("-AfooValue", "-BbarValue");
  }

  @Test
  public void testExpandIfAllAvailableWithStructFieldDoesntExpandIfMissing() throws Exception {
    assertThat(
            getCommandLineForFlagGroups(
                "flag_group("
                    + "  expand_if_available = 'struct.foo',"
                    + "  flags = ['-A%{struct.foo}','-B%{struct.bar}']"
                    + ")",
                createStructureVariables(
                    "struct", new StructureBuilder().addField("bar", "barValue"))))
        .isEmpty();
  }

  @Test
  public void testExpandIfAllAvailableWithStructFieldScopesRight() throws Exception {
    assertThat(
            getCommandLineForFlagGroups(
                "flag_group(flag_groups = [flag_group("
                    + "    expand_if_available = 'struct.foo',"
                    + "    flags = ['-A%{struct.foo}']"
                    + "  ),"
                    + "  flag_group("
                    + "    flags = ['-B%{struct.bar}']"
                    + "  )]"
                    + ")",
                createStructureVariables(
                    "struct", new StructureBuilder().addField("bar", "barValue"))))
        .containsExactly("-BbarValue");
  }

  @Test
  public void testExpandIfNoneAvailableExpandsIfNotAvailable() throws Exception {
    assertThat(
            getCommandLineForFlagGroups(
                "flag_group(flag_groups = [flag_group("
                    + "    expand_if_not_available = 'not_available',"
                    + "    flags = ['-foo']"
                    + "  ),"
                    + "  flag_group("
                    + "    expand_if_not_available = 'available',"
                    + "    flags = ['-bar']"
                    + "  )]"
                    + ")",
                createVariables("available", "available")))
        .containsExactly("-foo");
  }

  @Test
  public void testExpandIfNoneAvailableDoesntExpandIfThereIsOneOfManyAvailable() throws Exception {
    assertThat(
            getCommandLineForFlagGroups(
                "flag_group("
                    + "  expand_if_not_available = 'not_available',"
                    + "  flag_groups = [flag_group("
                    + "    expand_if_not_available = 'available',"
                    + "    flags = ['-foo']"
                    + "  )]"
                    + ")",
                createVariables("available", "available")))
        .isEmpty();
  }

  @Test
  public void testExpandIfTrueDoesntExpandIfMissing() throws Exception {
    assertThat(
            getCommandLineForFlagGroups(
                "flag_group("
                    + "  expand_if_true = 'missing',"
                    + "  flags = ['-A%{missing}']"
                    + "),"
                    + "flag_group("
                    + "  expand_if_false = 'missing',"
                    + "  flags = ['-B%{missing}']"
                    + ")",
                createVariables()))
        .isEmpty();
  }

  @Test
  public void testExpandIfTrueExpandsIfOne() throws Exception {
    assertThat(
            getCommandLineForFlagGroups(
                "flag_group("
                    + "  expand_if_true = 'struct.bool',"
                    + "  flags = ['-A%{struct.foo}','-B%{struct.bar}']"
                    + "),"
                    + "flag_group("
                    + "  expand_if_false = 'struct.bool',"
                    + "  flags = ['-X%{struct.foo}','-Y%{struct.bar}']"
                    + ")",
                createStructureVariables(
                    "struct",
                    new StructureBuilder()
                        .addField("bool", booleanValue(true))
                        .addField("foo", "fooValue")
                        .addField("bar", "barValue"))))
        .containsExactly("-AfooValue", "-BbarValue");
  }

  @Test
  public void testExpandIfTrueExpandsIfZero() throws Exception {
    assertThat(
            getCommandLineForFlagGroups(
                "flag_group("
                    + "  expand_if_true = 'struct.bool',"
                    + "  flags = ['-A%{struct.foo}','-B%{struct.bar}']"
                    + "),"
                    + "flag_group("
                    + "  expand_if_false = 'struct.bool',"
                    + "  flags = ['-X%{struct.foo}', '-Y%{struct.bar}']"
                    + ")",
                createStructureVariables(
                    "struct",
                    new StructureBuilder()
                        .addField("bool", booleanValue(false))
                        .addField("foo", "fooValue")
                        .addField("bar", "barValue"))))
        .containsExactly("-XfooValue", "-YbarValue");
  }

  private static VariableValue booleanValue(boolean val) throws ExpansionException {
    return CcToolchainVariables.builder()
        .addVariable("name", val)
        .build()
        .getVariable("name", PathMapper.NOOP);
  }

  @Test
  public void testExpandIfEqual() throws Exception {
    assertThat(
            getCommandLineForFlagGroups(
                "flag_group(  expand_if_equal = variable_with_value(name = 'var', value ="
                    + " 'equal_value'),  flags = ['-foo_%{var}']),flag_group(  expand_if_equal ="
                    + " variable_with_value(name = 'var', value = 'non_equal_value'),  flags ="
                    + " ['-bar_%{var}']),flag_group(  expand_if_equal = variable_with_value(name ="
                    + " 'non_existing_var', value = 'non_existing'),  flags ="
                    + " ['-baz_%{non_existing_var}'])",
                createVariables("var", "equal_value")))
        .containsExactly("-foo_equal_value");
  }

  @Test
  public void testListVariableExpansion() throws Exception {
    assertThat(
            getCommandLineForFlagGroups(
                "flag_group(iterate_over = 'v', flags = ['%{v}'])",
                createVariables("v", "1", "v", "2")))
        .containsExactly("1", "2");
  }

  @Test
  public void testListVariableExpansionMixedWithNonListVariable() throws Exception {
    assertThat(
            getCommandLineForFlagGroups(
                "flag_group(iterate_over = 'v1', flags = ['%{v1} %{v2}'])",
                createVariables("v1", "a1", "v1", "a2", "v2", "b")))
        .containsExactly("a1 b", "a2 b");
  }

  @Test
  public void testNestedListVariableExpansion() throws Exception {
    assertThat(
            getCommandLineForFlagGroups(
                "flag_group("
                    + "  iterate_over = 'v1',"
                    + "  flag_groups = [flag_group("
                    + "    iterate_over = 'v2',"
                    + "    flags = ['%{v1} %{v2}'],"
                    + "  )]"
                    + ")",
                createVariables("v1", "a1", "v1", "a2", "v2", "b1", "v2", "b2")))
        .containsExactly("a1 b1", "a1 b2", "a2 b1", "a2 b2");
  }

  @Test
  public void testListVariableExpansionMixedWithImplicitlyAccessedListVariableFails() {
    assertThat(
            getFlagGroupsExpansionError(
                "flag_group(iterate_over = 'v1', flags = ['%{v1} %{v2}'])",
                createVariables("v1", "a1", "v1", "a2", "v2", "b1", "v2", "b2")))
        .contains("Cannot expand variable 'v2': expected string, found sequence");
  }

  @Test
  public void testFlagGroupVariableExpansion() throws Exception {
    assertThat(
            getCommandLineForFlagGroups(
                ""
                    + "flag_group(iterate_over = 'v', flags = ['-f', '%{v}']),"
                    + "flag_group(flags = ['-end'])",
                createVariables("v", "1", "v", "2")))
        .containsExactly("-f", "1", "-f", "2", "-end");
    assertThat(
            getCommandLineForFlagGroups(
                ""
                    + "flag_group(iterate_over = 'v', flags = ['-f', '%{v}']),"
                    + "flag_group(iterate_over = 'v', flags = ['%{v}'])",
                createVariables("v", "1", "v", "2")))
        .containsExactly("-f", "1", "-f", "2", "1", "2");
    assertThat(
            getCommandLineForFlagGroups(
                ""
                    + "flag_group(iterate_over = 'v', flags = ['-f', '%{v}']),"
                    + "flag_group(iterate_over = 'v', flags = ['%{v}'])",
                createVariables("v", "1", "v", "2")))
        .containsExactly("-f", "1", "-f", "2", "1", "2");
  }

  private static ImmutableList<VariableValue> createNestedSequence(
      int depth, int count, String prefix) {
    ImmutableList.Builder<VariableValue> builder = ImmutableList.builder();
    if (depth == 0) {
      for (int i = 0; i < count; ++i) {
        String value = prefix + i;
        builder.add(new StringValue(value));
      }
    } else {
      for (int i = 0; i < count; ++i) {
        String value = prefix + i;
        builder.add(new Sequence(createNestedSequence(depth - 1, count, value)));
      }
    }
    return builder.build();
  }

  private static CcToolchainVariables createNestedVariables(String name, int depth, int count) {
    return CcToolchainVariables.builder()
        .addVariable(name, createNestedSequence(depth, count, ""))
        .build();
  }

  @Test
  public void testFlagTreeVariableExpansion() throws Exception {
    String nestedGroup =
        ""
            + "flag_group("
            + "  iterate_over = 'v',"
            + "  flag_groups = ["
            + "    flag_group(flags = ['-a']),"
            + "    flag_group(iterate_over = 'v', flags = ['%{v}']),"
            + "    flag_group(flags = ['-b']),"
            + "  ],"
            + ")";
    assertThat(getCommandLineForFlagGroups(nestedGroup, createNestedVariables("v", 1, 3)))
        .containsExactly(
            "-a", "00", "01", "02", "-b", "-a", "10", "11", "12", "-b", "-a", "20", "21", "22",
            "-b");

    ExpansionException e =
        assertThrows(
            ExpansionException.class,
            () -> getCommandLineForFlagGroups(nestedGroup, createNestedVariables("v", 2, 3)));
    assertThat(e).hasMessageThat().contains("'v'");

    AssertionError ae =
        assertThrows(
            AssertionError.class,
            () ->
                buildFeatures(
                    "features = [feature(",
                    "  name =  'a',",
                    "  flag_sets = [flag_set(",
                    "    action = 'c++-compile',",
                    "    flag_groups = [flag_group(",
                    "      flag_groups = [flag_group(flags = ['-f'])],",
                    "      flags = ['-f'],",
                    "    )],",
                    "  )],",
                    ")]"));
    assertThat(ae)
        .hasMessageThat()
        .contains("flag_group must not contain both a flag and another flag_group.");
  }

  @Test
  public void testImplies() throws Exception {
    CcToolchainFeatures features =
        buildFeatures(
            "features = [",
            "    feature(name = 'a', implies = ['b', 'c']),",
            "    feature(name = 'b'),",
            "    feature(name = 'c', implies = ['d']),",
            "    feature(name = 'd'),",
            "    feature(name = 'e'),",
            "]");
    assertThat(getEnabledFeatures(features, "a")).containsExactly("a", "b", "c", "d");
  }

  @Test
  public void testRequires() throws Exception {
    CcToolchainFeatures features =
        buildFeatures(
            "features = [",
            "    feature(name = 'a', requires = [feature_set(features = ['b'])]),",
            "    feature(name = 'b', requires = [feature_set(features = ['c'])]),",
            "    feature(name = 'c'),",
            "]");
    assertThat(getEnabledFeatures(features, "a")).isEmpty();
    assertThat(getEnabledFeatures(features, "a", "b")).isEmpty();
    assertThat(getEnabledFeatures(features, "a", "c")).containsExactly("c");
    assertThat(getEnabledFeatures(features, "a", "b", "c")).containsExactly("a", "b", "c");
  }

  @Test
  public void testDisabledRequirementChain() throws Exception {
    CcToolchainFeatures features =
        buildFeatures(
            "features = [",
            "    feature(name = 'a'),",
            "    feature(name = 'b', requires = [feature_set(features = ['c'])], implies ="
                + " ['a']),",
            "    feature(name = 'c'),",
            "]");
    assertThat(getEnabledFeatures(features, "b")).isEmpty();
    features =
        buildFeatures(
            "features = [",
            "    feature(name = 'a'),",
            "    feature(name = 'b', requires = [feature_set(features = ['a'])], implies ="
                + " ['c']),",
            "    feature(name = 'c'),",
            "    feature(name = 'd', requires = [feature_set(features = ['c'])], implies ="
                + " ['e']),",
            "    feature(name = 'e'),",
            "]");
    assertThat(getEnabledFeatures(features, "b", "d")).isEmpty();
  }

  @Test
  public void testEnabledRequirementChain() throws Exception {
    CcToolchainFeatures features =
        buildFeatures(
            "features = [",
            "    feature(name = '0', implies = ['a']),",
            "    feature(name = 'a'),",
            "    feature(name = 'b', requires = [feature_set(features = ['a'])], implies ="
                + " ['c']),",
            "    feature(name = 'c'),",
            "    feature(name = 'd', requires = [feature_set(features = ['c'])], implies ="
                + " ['e']),",
            "    feature(name = 'e'),",
            "]");
    assertThat(getEnabledFeatures(features, "0", "b", "d"))
        .containsExactly("0", "a", "b", "c", "d", "e");
  }

  @Test
  public void testLogicInRequirements() throws Exception {
    CcToolchainFeatures features =
        buildFeatures(
            "features = [",
            "    feature(",
            "        name = 'a',",
            "        requires = [",
            "            feature_set(features = ['b', 'c']),",
            "            feature_set(features = ['d']),",
            "        ],",
            "    ),",
            "    feature(name = 'b'),",
            "    feature(name = 'c'),",
            "    feature(name = 'd'),",
            "]");
    assertThat(getEnabledFeatures(features, "a", "b", "c")).containsExactly("a", "b", "c");
    assertThat(getEnabledFeatures(features, "a", "b")).containsExactly("b");
    assertThat(getEnabledFeatures(features, "a", "c")).containsExactly("c");
    assertThat(getEnabledFeatures(features, "a", "d")).containsExactly("a", "d");
  }

  @Test
  public void testImpliesImpliesRequires() throws Exception {
    CcToolchainFeatures features =
        buildFeatures(
            "features = [",
            "    feature(name = 'a', implies = ['b']),",
            "    feature(name = 'b', requires = [feature_set(features = ['c'])]),",
            "    feature(name = 'c'),",
            "]");
    assertThat(getEnabledFeatures(features, "a")).isEmpty();
  }

  @Test
  public void testMultipleImplies() throws Exception {
    CcToolchainFeatures features =
        buildFeatures(
            "features = [",
            "    feature(name = 'a', implies = ['b', 'c', 'd']),",
            "    feature(name = 'b'),",
            "    feature(name = 'c', requires = [feature_set(features = ['e'])]),",
            "    feature(name = 'd'),",
            "    feature(name = 'e'),",
            "]");
    assertThat(getEnabledFeatures(features, "a")).isEmpty();
    assertThat(getEnabledFeatures(features, "a", "e")).containsExactly("a", "b", "c", "d", "e");
  }

  @Test
  public void testDisabledFeaturesDoNotEnableImplications() throws Exception {
    CcToolchainFeatures features =
        buildFeatures(
            "features = [",
            "    feature(name = 'a', implies = ['b'], requires = [feature_set(features = ['c'])]),",
            "    feature(name = 'b'),",
            "    feature(name = 'c'),",
            "]");
    assertThat(getEnabledFeatures(features, "a")).isEmpty();
  }

  @Test
  public void testFeatureNameCollision() {
    AssertionError e =
        assertThrows(
            AssertionError.class,
            () ->
                buildFeatures(
                    "features = [",
                    "    feature(name = '+++collision+++'),",
                    "    feature(name = '+++collision+++'),",
                    "]"));
    assertThat(e)
        .hasMessageThat()
        .contains("feature or action config '+++collision+++' was specified multiple times.");
  }

  @Test
  public void testReferenceToUndefinedFeature() {
    AssertionError e =
        assertThrows(
            AssertionError.class,
            () -> buildFeatures("features = [feature(name = 'a', implies = ['<<<undefined>>>'])]"));
    assertThat(e)
        .hasMessageThat()
        .contains(
            "feature '<<<undefined>>>', which is referenced from feature 'a', is not defined");
  }

  @Test
  public void testImpliesWithCycle() throws Exception {
    CcToolchainFeatures features =
        buildFeatures(
            "features = [",
            "    feature(name = 'a', implies = ['b']),",
            "    feature(name = 'b', implies = ['a']),",
            "]");
    assertThat(getEnabledFeatures(features, "a")).containsExactly("a", "b");
    assertThat(getEnabledFeatures(features, "b")).containsExactly("a", "b");
  }

  @Test
  public void testMultipleImpliesCycle() throws Exception {
    CcToolchainFeatures features =
        buildFeatures(
            "features = [",
            "    feature(name = 'a', implies = ['b', 'c', 'd']),",
            "    feature(name = 'b'),",
            "    feature(name = 'c', requires = [feature_set(features = ['e'])]),",
            "    feature(name = 'd', requires = [feature_set(features = ['f'])]),",
            "    feature(name = 'e', requires = [feature_set(features = ['c'])]),",
            "    feature(name = 'f'),",
            "]");
    assertThat(getEnabledFeatures(features, "a", "e")).isEmpty();
    assertThat(getEnabledFeatures(features, "a", "e", "f"))
        .containsExactly("a", "b", "c", "d", "e", "f");
  }

  @Test
  public void testRequiresWithCycle() throws Exception {
    CcToolchainFeatures features =
        buildFeatures(
            "features = [",
            "    feature(name = 'a', requires = [feature_set(features = ['b'])]),",
            "    feature(name = 'b', requires = [feature_set(features = ['a'])]),",
            "    feature(name = 'c', implies = ['a']),",
            "    feature(name = 'd', implies = ['b']),",
            "]");
    assertThat(getEnabledFeatures(features, "c")).isEmpty();
    assertThat(getEnabledFeatures(features, "d")).isEmpty();
    assertThat(getEnabledFeatures(features, "c", "d")).containsExactly("a", "b", "c", "d");
  }

  @Test
  public void testImpliedByOneEnabledAndOneDisabledFeature() throws Exception {
    CcToolchainFeatures features =
        buildFeatures(
            "features = [",
            "    feature(name = 'a'),",
            "    feature(name = 'b', requires = [feature_set(features = ['a'])], implies ="
                + " ['d']),",
            "    feature(name = 'c', implies = ['d']),",
            "    feature(name = 'd'),",
            "]");
    assertThat(getEnabledFeatures(features, "b", "c")).containsExactly("c", "d");
  }

  @Test
  public void testRequiresOneEnabledAndOneUnsupportedFeature() throws Exception {
    CcToolchainFeatures features =
        buildFeatures(
            "features = [",
            "    feature(",
            "        name = 'a',",
            "        requires = [",
            "            feature_set(features = ['b']),",
            "            feature_set(features = ['c'])",
            "        ],",
            "    ),",
            "    feature(name = 'b'),",
            "    feature(name = 'c', requires = [feature_set(features = ['d'])]),",
            "    feature(name = 'd'),",
            "]");
    assertThat(getEnabledFeatures(features, "a", "b", "c")).containsExactly("a", "b");
  }

  @Test
  public void testFlagGroupsWithMissingVariableIsNotExpanded() throws Exception {
    FeatureConfiguration configuration =
        buildFeatures(
                "features = [feature(",
                "    name = 'a',",
                "    flag_sets = [",
                "        flag_set(",
                "            actions = ['c++-compile'],",
                "            flag_groups = [",
                "                flag_group(expand_if_available = 'v', flags = ['%{v}'])",
                "            ],",
                "        ),",
                "        flag_set(",
                "            actions = ['c++-compile'],",
                "            flag_groups = [flag_group(flags = ['unconditional'])],",
                "        ),",
                "    ],",
                ")]")
            .getFeatureConfiguration(ImmutableSet.of("a"));

    assertThat(configuration.getCommandLine(CppActionNames.CPP_COMPILE, createVariables()))
        .containsExactly("unconditional");
  }

  @Test
  public void testOnlyFlagGroupsWithAllVariablesPresentAreExpanded() throws Exception {
    FeatureConfiguration configuration =
        buildFeatures(
                "features = [feature(",
                "    name = 'a',",
                "    flag_sets = [",
                "        flag_set(",
                "            actions = ['c++-compile'],",
                "            flag_groups = [",
                "                flag_group(expand_if_available = 'v', flags = ['%{v}'])",
                "            ],",
                "        ),",
                "        flag_set(",
                "            actions = ['c++-compile'],",
                "            flag_groups = [",
                "                flag_group(",
                "                    expand_if_available = 'w',",
                "                    flag_groups = [flag_group(",
                "                    expand_if_available = 'v',",
                "                    flags = ['%{v}%{w}'])],",
                "                )",
                "            ],",
                "        ),",
                "        flag_set(",
                "            actions = ['c++-compile'],",
                "            flag_groups = [flag_group(flags = ['unconditional'])],",
                "        ),",
                "    ],",
                ")]")
            .getFeatureConfiguration(ImmutableSet.of("a"));

    assertThat(configuration.getCommandLine(CppActionNames.CPP_COMPILE, createVariables("v", "1")))
        .containsExactly("1", "unconditional");
  }

  @Test
  public void testOnlyInnerFlagGroupIsIteratedWithSequenceVariable() throws Exception {
    FeatureConfiguration configuration =
        buildFeatures(
                "features = [feature(",
                "    name = 'a',",
                "    flag_sets = [",
                "        flag_set(",
                "            actions = ['c++-compile'],",
                "            flag_groups = [",
                "                flag_group(",
                "                    expand_if_available = 'v',",
                "                    iterate_over = 'v',",
                "                    flags = ['%{v}']",
                "                ),",
                "            ],",
                "        ),",
                "        flag_set(",
                "            actions = ['c++-compile'],",
                "            flag_groups = [",
                "                flag_group(",
                "                    expand_if_available = 'w',",
                "                    flag_groups = [flag_group(",
                "                    iterate_over = 'v',",
                "                    expand_if_available = 'v',",
                "                    flags = ['%{v}%{w}'])],",
                "                ),",
                "            ],",
                "        ),",
                "        flag_set(",
                "            actions = ['c++-compile'],",
                "            flag_groups = [flag_group(flags = ['unconditional'])],",
                "        ),",
                "    ],",
                ")]")
            .getFeatureConfiguration(ImmutableSet.of("a"));

    assertThat(
            configuration.getCommandLine(
                CppActionNames.CPP_COMPILE, createVariables("v", "1", "v", "2")))
        .containsExactly("1", "2", "unconditional")
        .inOrder();
  }

  @Test
  public void testFlagSetsAreIteratedIndividuallyForSequenceVariables() throws Exception {
    FeatureConfiguration configuration =
        buildFeatures(
                "features = [feature(",
                "    name = 'a',",
                "    flag_sets = [",
                "        flag_set(",
                "            actions = ['c++-compile'],",
                "            flag_groups = [",
                "                flag_group(",
                "                    expand_if_available = 'v',",
                "                    iterate_over = 'v',",
                "                    flags = ['%{v}']",
                "                ),",
                "            ],",
                "        ),",
                "        flag_set(",
                "            actions = ['c++-compile'],",
                "            flag_groups = [",
                "                flag_group(",
                "                    iterate_over = 'v',",
                "                    expand_if_available = 'v',",
                "                    flags = ['%{v}%{w}']",
                "                ),",
                "            ],",
                "        ),",
                "        flag_set(",
                "            actions = ['c++-compile'],",
                "            flag_groups = [flag_group(flags = ['unconditional'])],",
                "        ),",
                "    ],",
                ")]")
            .getFeatureConfiguration(ImmutableSet.of("a"));

    assertThat(
            configuration.getCommandLine(
                CppActionNames.CPP_COMPILE, createVariables("v", "1", "v", "2", "w", "3")))
        .containsExactly("1", "2", "13", "23", "unconditional")
        .inOrder();
  }

  @Test
  public void testConfiguration() throws Exception {
    CcToolchainFeatures features =
        buildFeatures(
            "features = [",
            "    feature(",
            "        name = 'a',",
            "        flag_sets = [",
            "            flag_set(",
            "                actions = ['c++-compile'],",
            "                flag_groups = [flag_group(flags = ['-f', '%{v}'])],",
            "            ),",
            "        ],",
            "    ),",
            "    feature(name = 'b', implies = ['a']),",
            "]");
    assertThat(getEnabledFeatures(features, "b")).containsExactly("a", "b");
    assertThat(
            features
                .getFeatureConfiguration(ImmutableSet.of("b"))
                .getCommandLine(CppActionNames.CPP_COMPILE, createVariables("v", "1")))
        .containsExactly("-f", "1");

    CcToolchainFeatures deserialized = RoundTripping.roundTrip(features);
    assertThat(getEnabledFeatures(deserialized, "b")).containsExactly("a", "b");
    assertThat(
            features
                .getFeatureConfiguration(ImmutableSet.of("b"))
                .getCommandLine(CppActionNames.CPP_COMPILE, createVariables("v", "1")))
        .containsExactly("-f", "1");
  }

  @Test
  public void testDefaultFeatures() throws Exception {
    CcToolchainFeatures features =
        buildFeatures(
            "features = ["
                + "feature(name = 'no_legacy_features'),"
                + "feature(name = 'a'), feature(name = 'b', enabled = True)"
                + "]");
    assertThat(features.getDefaultFeaturesAndActionConfigs()).containsExactly("b");
  }

  @Test
  public void testDefaultActionConfigs() throws Exception {
    CcToolchainFeatures features =
        buildFeatures(
            "features = [feature(name = 'no_legacy_features')],",
            "action_configs = [",
            "    action_config(action_name = 'a'),",
            "    action_config(action_name = 'b', enabled = True),",
            "]");
    assertThat(features.getDefaultFeaturesAndActionConfigs()).containsExactly("b");
  }

  @Test
  public void testWithFeature_oneSetOneFeature() throws Exception {
    CcToolchainFeatures features =
        buildFeatures(
            "features = [",
            "feature(name = 'no_legacy_features'),",
            "    feature(",
            "        name = 'a',",
            "        flag_sets = [",
            "            flag_set(",
            "                with_features = [with_feature_set(features = ['b'])],",
            "                actions = ['c++-compile'],",
            "                flag_groups = [flag_group(flags = ['dummy_flag'])],",
            "            ),",
            "        ],",
            "    ),",
            "    feature(name = 'b'),",
            "]");
    assertThat(
            features
                .getFeatureConfiguration(ImmutableSet.of("a", "b"))
                .getCommandLine(CppActionNames.CPP_COMPILE, createVariables()))
        .containsExactly("dummy_flag");
    assertThat(
            features
                .getFeatureConfiguration(ImmutableSet.of("a"))
                .getCommandLine(CppActionNames.CPP_COMPILE, createVariables()))
        .doesNotContain("dummy_flag");
  }

  @Test
  public void testWithFeature_oneSetMultipleFeatures() throws Exception {
    CcToolchainFeatures features =
        buildFeatures(
            "features = [",
            "    feature(",
            "        name = 'a',",
            "        flag_sets = [",
            "            flag_set(",
            "                with_features = [with_feature_set(features = ['b', 'c'])],",
            "                actions = ['c++-compile'],",
            "                flag_groups = [flag_group(flags = ['dummy_flag'])],",
            "            ),",
            "        ],",
            "    ),",
            "    feature(name = 'b'),",
            "    feature(name = 'c'),",
            "]");
    assertThat(
            features
                .getFeatureConfiguration(ImmutableSet.of("a", "b", "c"))
                .getCommandLine(CppActionNames.CPP_COMPILE, createVariables()))
        .containsExactly("dummy_flag");
    assertThat(
            features
                .getFeatureConfiguration(ImmutableSet.of("a", "b"))
                .getCommandLine(CppActionNames.CPP_COMPILE, createVariables()))
        .doesNotContain("dummy_flag");
    assertThat(
            features
                .getFeatureConfiguration(ImmutableSet.of("a"))
                .getCommandLine(CppActionNames.CPP_COMPILE, createVariables()))
        .doesNotContain("dummy_flag");
  }

  @Test
  public void testWithFeature_mulipleSetsMultipleFeatures() throws Exception {
    CcToolchainFeatures features =
        buildFeatures(
            "features = [",
            "    feature(",
            "        name = 'a',",
            "        flag_sets = [",
            "            flag_set(",
            "                with_features = [",
            "                    with_feature_set(features = ['b1', 'c1']),",
            "                    with_feature_set(features = ['b2', 'c2']),",
            "                ],",
            "                actions = ['c++-compile'],",
            "                flag_groups = [flag_group(flags = ['dummy_flag'])],",
            "            ),",
            "        ],",
            "    ),",
            "    feature(name = 'b1'),",
            "    feature(name = 'c1'),",
            "    feature(name = 'b2'),",
            "    feature(name = 'c2'),",
            "]");
    assertThat(
            features
                .getFeatureConfiguration(ImmutableSet.of("a", "b1", "c1", "b2", "c2"))
                .getCommandLine(CppActionNames.CPP_COMPILE, createVariables()))
        .containsExactly("dummy_flag");
    assertThat(
            features
                .getFeatureConfiguration(ImmutableSet.of("a", "b1", "c1"))
                .getCommandLine(CppActionNames.CPP_COMPILE, createVariables()))
        .containsExactly("dummy_flag");
    assertThat(
            features
                .getFeatureConfiguration(ImmutableSet.of("a", "b1", "b2"))
                .getCommandLine(CppActionNames.CPP_COMPILE, createVariables()))
        .doesNotContain("dummy_flag");
  }

  @Test
  public void testWithFeature_notFeature() throws Exception {
    CcToolchainFeatures features =
        buildFeatures(
            "features = [",
            "    feature(",
            "        name = 'a',",
            "        flag_sets = [",
            "            flag_set(",
            "                with_features = [",
            "                    with_feature_set(not_features = ['x', 'y'], features = ['z']),",
            "                    with_feature_set(not_features = ['q']),",
            "                ],",
            "                actions = ['c++-compile'],",
            "                flag_groups = [flag_group(flags = ['dummy_flag'])],",
            "            ),",
            "        ],",
            "    ),",
            "    feature(name = 'x'),",
            "    feature(name = 'y'),",
            "    feature(name = 'z'),",
            "    feature(name = 'q'),",
            "]");
    assertThat(
            features
                .getFeatureConfiguration(ImmutableSet.of("a"))
                .getCommandLine(CppActionNames.CPP_COMPILE, createVariables()))
        .containsExactly("dummy_flag");
    assertThat(
            features
                .getFeatureConfiguration(ImmutableSet.of("a", "q"))
                .getCommandLine(CppActionNames.CPP_COMPILE, createVariables()))
        .doesNotContain("dummy_flag");
    assertThat(
            features
                .getFeatureConfiguration(ImmutableSet.of("a", "q", "z"))
                .getCommandLine(CppActionNames.CPP_COMPILE, createVariables()))
        .containsExactly("dummy_flag");
    assertThat(
            features
                .getFeatureConfiguration(ImmutableSet.of("a", "q", "x", "z"))
                .getCommandLine(CppActionNames.CPP_COMPILE, createVariables()))
        .doesNotContain("dummy_flag");
    assertThat(
            features
                .getFeatureConfiguration(ImmutableSet.of("a", "q", "x", "y", "z"))
                .getCommandLine(CppActionNames.CPP_COMPILE, createVariables()))
        .doesNotContain("dummy_flag");
  }

  @Test
  public void testActivateActionConfigFromFeature() throws Exception {
    CcToolchainFeatures toolchainFeatures =
        buildFeatures(
            "action_configs = [",
            "    action_config(",
            "        action_name = 'action-a',",
            "        tools = [",
            "            tool(",
            "                path = 'toolchain/feature-a',",
            "                with_features = [with_feature_set(features = ['feature-a'])],",
            "            ),",
            "        ],",
            "    ),",
            "],",
            "features = [",
            "    feature(name = 'activates-action-a', implies = ['action-a']),",
            "]");

    FeatureConfiguration featureConfiguration =
        toolchainFeatures.getFeatureConfiguration(ImmutableSet.of("activates-action-a"));

    assertThat(featureConfiguration.actionIsConfigured("action-a")).isTrue();
  }

  @Test
  public void testFeatureCanRequireActionConfig() throws Exception {
    CcToolchainFeatures toolchainFeatures =
        buildFeatures(
            "action_configs = [",
            "    action_config(",
            "        action_name = 'action-a',",
            "        tools = [",
            "            tool(",
            "                path = 'toolchain/feature-a',",
            "                with_features = [with_feature_set(features = ['feature-a'])],",
            "            ),",
            "        ],",
            "    ),",
            "],",
            "features = [",
            "    feature(",
            "        name = 'requires-action-a',",
            "        requires = [feature_set(features = ['action-a'])],",
            "    ),",
            "]");

    FeatureConfiguration featureConfigurationWithoutAction =
        toolchainFeatures.getFeatureConfiguration(ImmutableSet.of("requires-action-a"));
    assertThat(featureConfigurationWithoutAction.isEnabled("requires-action-a")).isFalse();

    FeatureConfiguration featureConfigurationWithAction =
        toolchainFeatures.getFeatureConfiguration(ImmutableSet.of("action-a", "requires-action-a"));
    assertThat(featureConfigurationWithAction.isEnabled("requires-action-a")).isTrue();
  }

  @Test
  public void testSimpleActionTool() throws Exception {
    FeatureConfiguration configuration =
        buildFeatures(
                "action_configs = [",
                "    action_config(",
                "        action_name = 'action-a',",
                "        tools = [tool(path = 'toolchain/a')],",
                "    ),",
                "],",
                "features = [",
                "    feature(name = 'activates-action-a', implies = ['action-a']),",
                "]")
            .getFeatureConfiguration(ImmutableSet.of("activates-action-a"));
    assertThat(configuration.getToolPathForAction("action-a")).isEqualTo("crosstool/toolchain/a");
  }

  @Test
  public void testActionToolFromFeatureSet() throws Exception {
    CcToolchainFeatures toolchainFeatures =
        buildFeatures(
            "action_configs = [",
            "    action_config(",
            "        action_name = 'action-a',",
            "        tools = [",
            "            tool(",
            "                path = 'toolchain/features-a-and-b',",
            "                with_features = [with_feature_set(features = ['feature-a',"
                + " 'feature-b'])],",
            "            ),",
            "            tool(",
            "                path = 'toolchain/feature-a-and-not-c',",
            "                with_features = [with_feature_set(features = ['feature-a'],"
                + " not_features = ['feature-c'])],",
            "            ),",
            "            tool(",
            "                path = 'toolchain/feature-b-or-c',",
            "                with_features = [",
            "                    with_feature_set(features = ['feature-b']),",
            "                    with_feature_set(features = ['feature-c'])",
            "                ],",
            "            ),",
            "            tool(path = 'toolchain/default'),",
            "        ],",
            "    ),",
            "],",
            "features = [",
            "    feature(name = 'feature-a'),",
            "    feature(name = 'feature-b'),",
            "    feature(name = 'feature-c'),",
            "    feature(name = 'activates-action-a', implies = ['action-a']),",
            "]");

    FeatureConfiguration featureAConfiguration =
        toolchainFeatures.getFeatureConfiguration(
            ImmutableSet.of("feature-a", "activates-action-a"));
    assertThat(featureAConfiguration.getToolPathForAction("action-a"))
        .isEqualTo("crosstool/toolchain/feature-a-and-not-c");

    FeatureConfiguration featureAAndCConfiguration =
        toolchainFeatures.getFeatureConfiguration(
            ImmutableSet.of("feature-a", "feature-c", "activates-action-a"));
    assertThat(featureAAndCConfiguration.getToolPathForAction("action-a"))
        .isEqualTo("crosstool/toolchain/feature-b-or-c");

    FeatureConfiguration featureBConfiguration =
        toolchainFeatures.getFeatureConfiguration(
            ImmutableSet.of("feature-b", "activates-action-a"));
    assertThat(featureBConfiguration.getToolPathForAction("action-a"))
        .isEqualTo("crosstool/toolchain/feature-b-or-c");

    FeatureConfiguration featureCConfiguration =
        toolchainFeatures.getFeatureConfiguration(
            ImmutableSet.of("feature-c", "activates-action-a"));
    assertThat(featureCConfiguration.getToolPathForAction("action-a"))
        .isEqualTo("crosstool/toolchain/feature-b-or-c");

    FeatureConfiguration featureAAndBConfiguration =
        toolchainFeatures.getFeatureConfiguration(
            ImmutableSet.of("feature-a", "feature-b", "activates-action-a"));
    assertThat(featureAAndBConfiguration.getToolPathForAction("action-a"))
        .isEqualTo("crosstool/toolchain/features-a-and-b");

    FeatureConfiguration noFeaturesConfiguration =
        toolchainFeatures.getFeatureConfiguration(ImmutableSet.of("activates-action-a"));
    assertThat(noFeaturesConfiguration.getToolPathForAction("action-a"))
        .isEqualTo("crosstool/toolchain/default");
  }

  @Test
  public void testErrorForNoMatchingTool() throws Exception {
    CcToolchainFeatures toolchainFeatures =
        buildFeatures(
            "action_configs = [",
            "    action_config(",
            "        action_name = 'action-a',",
            "        tools = [",
            "            tool(",
            "                path = 'toolchain/feature-a',",
            "                with_features = [with_feature_set(features = ['feature-a'])],",
            "            ),",
            "        ],",
            "    ),",
            "],",
            "features = [",
            "    feature(name = 'feature-a'),",
            "    feature(name = 'activates-action-a', implies = ['action-a']),",
            "]");

    FeatureConfiguration noFeaturesConfiguration =
        toolchainFeatures.getFeatureConfiguration(ImmutableSet.of("activates-action-a"));

    IllegalArgumentException e =
        assertThrows(
            IllegalArgumentException.class,
            () -> noFeaturesConfiguration.getToolPathForAction("action-a"));
    assertThat(e)
        .hasMessageThat()
        .contains("Matching tool for action action-a not found for given feature configuration");
  }

  @Test
  public void testActivateActionConfigDirectly() throws Exception {
    CcToolchainFeatures toolchainFeatures =
        buildFeatures(
            "action_configs = [",
            "    action_config(",
            "        action_name = 'action-a',",
            "        tools = [",
            "            tool(",
            "                path = 'toolchain/feature-a',",
            "                with_features = [with_feature_set(features = ['feature-a'])],",
            "            ),",
            "        ],",
            "    ),",
            "]");

    FeatureConfiguration featureConfiguration =
        toolchainFeatures.getFeatureConfiguration(ImmutableSet.of("action-a"));

    assertThat(featureConfiguration.actionIsConfigured("action-a")).isTrue();
  }

  @Test
  public void testActionConfigCanActivateFeature() throws Exception {
    CcToolchainFeatures toolchainFeatures =
        buildFeatures(
            "action_configs = [",
            "    action_config(",
            "        action_name = 'action-a',",
            "        tools = [",
            "            tool(",
            "                path = 'toolchain/feature-a',",
            "                with_features = [with_feature_set(features = ['feature-a'])],",
            "            ),",
            "        ],",
            "        implies = ['activated-feature'],",
            "    ),",
            "],",
            "features = [",
            "    feature(name = 'activated-feature'),",
            "]");

    FeatureConfiguration featureConfiguration =
        toolchainFeatures.getFeatureConfiguration(ImmutableSet.of("action-a"));

    assertThat(featureConfiguration.isEnabled("activated-feature")).isTrue();
  }

  @Test
  public void testInvalidActionConfigurationMultipleActionConfigsForAction() {
    AssertionError e =
        assertThrows(
            AssertionError.class,
            () ->
                buildFeatures(
                    "action_configs = [",
                    "    action_config(action_name = 'action-a'),",
                    "    action_config(action_name = 'action-a'),",
                    "]"));
    assertThat(e).hasMessageThat().contains("multiple action configs for action 'action-a'");
  }

  @Test
  public void testFlagsFromActionConfig() throws Exception {
    FeatureConfiguration featureConfiguration =
        buildFeatures(
                "action_configs = [",
                "    action_config(",
                "        action_name = 'c++-compile',",
                "        flag_sets = [flag_set(flag_groups = [flag_group(flags = ['foo'])])],",
                "    ),",
                "]")
            .getFeatureConfiguration(ImmutableSet.of("c++-compile"));
    List<String> commandLine =
        featureConfiguration.getCommandLine("c++-compile", createVariables());
    assertThat(commandLine).contains("foo");
  }

  @Test
  public void testErrorForFlagFromActionConfigWithSpecifiedAction() {
    AssertionError e =
        assertThrows(
            AssertionError.class,
            () ->
                buildFeatures(
                        "action_configs = [",
                        "    action_config(",
                        "        action_name = 'c++-compile',",
                        "        flag_sets = [",
                        "            flag_set(",
                        "                actions = ['c++-compile'],",
                        "                flag_groups = [flag_group(flags = ['foo'])],",
                        "            ),",
                        "        ],",
                        "    ),",
                        "]")
                    .getFeatureConfiguration(ImmutableSet.of("c++-compile")));
    assertThat(e)
        .hasMessageThat()
        .contains(String.format(ActionConfig.FLAG_SET_WITH_ACTION_ERROR, "c++-compile"));
  }

  @Test
  public void testProvidesCollision() {
    Exception e =
        assertThrows(
            Exception.class,
            () ->
                buildFeatures(
                        "features = [",
                        "    feature(name = 'a', provides = ['provides_string']),",
                        "    feature(name = 'b', provides = ['provides_string']),",
                        "]")
                    .getFeatureConfiguration(ImmutableSet.of("a", "b")));
    assertThat(e).hasMessageThat().contains("a b");
  }

  @Test
  public void testGetArtifactNameExtensionForCategory() throws Exception {
    CcToolchainFeatures toolchainFeatures =
        buildFeatures(
            "artifact_name_patterns = [",
            "    artifact_name_pattern(",
            "        category_name = 'object_file',",
            "        prefix = '',",
            "        extension = '.obj',",
            "    ),",
            "    artifact_name_pattern(",
            "        category_name = 'executable',",
            "        prefix = '',",
            "        extension = '',",
            "    ),",
            "    artifact_name_pattern(",
            "        category_name = 'static_library',",
            "        prefix = '',",
            "        extension = '.a',",
            "    ),",
            "]");
    assertThat(toolchainFeatures.getArtifactNameExtensionForCategory(ArtifactCategory.OBJECT_FILE))
        .isEqualTo(".obj");
    assertThat(toolchainFeatures.getArtifactNameExtensionForCategory(ArtifactCategory.EXECUTABLE))
        .isEmpty();
    assertThat(
            toolchainFeatures.getArtifactNameExtensionForCategory(ArtifactCategory.STATIC_LIBRARY))
        .isEqualTo(".a");
  }
}
