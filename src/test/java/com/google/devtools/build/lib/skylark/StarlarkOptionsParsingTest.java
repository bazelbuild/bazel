// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skylark;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.packages.SkylarkSemanticsOptions;
import com.google.devtools.build.lib.pkgcache.LoadingOptions;
import com.google.devtools.build.lib.pkgcache.PackageCacheOptions;
import com.google.devtools.build.lib.runtime.BlazeCommand;
import com.google.devtools.build.lib.runtime.BlazeCommandEventHandler.Options;
import com.google.devtools.build.lib.runtime.BlazeCommandResult;
import com.google.devtools.build.lib.runtime.ClientOptions;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.CommonCommandOptions;
import com.google.devtools.build.lib.runtime.KeepGoingOption;
import com.google.devtools.build.lib.runtime.StarlarkOptionsParser;
import com.google.devtools.build.lib.skylark.util.SkylarkTestCase;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;
import com.google.devtools.common.options.OptionsParsingResult;
import java.util.Arrays;
import java.util.List;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit test for the {@code StarlarkOptionsParser}. */
@RunWith(JUnit4.class)
public class StarlarkOptionsParsingTest extends SkylarkTestCase {

  private StarlarkOptionsParser starlarkOptionsParser;

  private static final List<Class<? extends OptionsBase>> requiredOptionsClasses =
      ImmutableList.of(
          PackageCacheOptions.class,
          SkylarkSemanticsOptions.class,
          KeepGoingOption.class,
          LoadingOptions.class,
          ClientOptions.class,
          Options.class,
          CommonCommandOptions.class);

  @Before
  public void setUp() throws Exception {
    optionsParser =
        OptionsParser.newOptionsParser(
            Iterables.concat(requiredOptionsClasses, ruleClassProvider.getConfigurationOptions()));
    starlarkOptionsParser =
        StarlarkOptionsParser.newStarlarkOptionsParserForTesting(
            skyframeExecutor, reporter, PathFragment.EMPTY_FRAGMENT, optionsParser);
  }

  @Command(
      name = "residue",
      builds = true,
      options = {
        PackageCacheOptions.class,
        SkylarkSemanticsOptions.class,
        KeepGoingOption.class,
        LoadingOptions.class,
        ClientOptions.class,
        Options.class,
      },
      allowResidue = true,
      shortDescription =
          "a dummy command for testing that allows residue and recognizes all"
              + " relevant options for starlark options parsing.",
      help = "")
  private static class ResidueCommand implements BlazeCommand {
    @Override
    public BlazeCommandResult exec(CommandEnvironment env, OptionsParsingResult options) {
      return BlazeCommandResult.exitCode(ExitCode.SUCCESS);
    }

    @Override
    public void editOptions(OptionsParser optionsParser) {}
  }

  private OptionsParsingResult parseStarlarkOptions(String options) throws Exception {
    starlarkOptionsParser.setResidueForTesting(Arrays.asList(options.split(" ")));
    starlarkOptionsParser.parse(
        ResidueCommand.class.getAnnotation(Command.class), new StoredEventHandler());
    return starlarkOptionsParser.getNativeOptionsParserFortesting();
  }

  private void writeBuildSetting(String type, String defaultValue, boolean isFlag)
      throws Exception {
    setSkylarkSemanticsOptions("--experimental_build_setting_api=True");

    String flag = isFlag ? "True" : "False";

    scratch.file(
        "test/build_setting.bzl",
        "def _build_setting_impl(ctx):",
        "  return []",
        type + "_setting = rule(",
        "  implementation = _build_setting_impl,",
        "  build_setting = config." + type + "(flag=" + flag + ")",
        ")");
    scratch.file(
        "test/BUILD",
        "load('//test:build_setting.bzl', '" + type + "_setting')",
        type
            + "_setting(name = 'my_"
            + type
            + "_setting', build_setting_default = "
            + defaultValue
            + ")");
  }

  private void writeBasicIntFlag() throws Exception {
    writeBuildSetting("int", "42", true);
  }

  // test --flag=value
  @Test
  public void testFlagEqualsValueForm() throws Exception {
    writeBasicIntFlag();

    OptionsParsingResult result = parseStarlarkOptions("--//test:my_int_setting=666");

    assertThat(result.getStarlarkOptions()).hasSize(1);
    assertThat(result.getStarlarkOptions().get("//test:my_int_setting")).isEqualTo(666);
    assertThat(result.getResidue()).isEmpty();
  }

  // test --flag value
  @Test
  public void testFlagSpaceValueForm() throws Exception {
    writeBasicIntFlag();

    OptionsParsingResult result = parseStarlarkOptions("--//test:my_int_setting 666");

    assertThat(result.getStarlarkOptions()).hasSize(1);
    assertThat(result.getStarlarkOptions().get("//test:my_int_setting")).isEqualTo(666);
    assertThat(result.getResidue()).isEmpty();
  }

  // test --fake_flag=value
  @Test
  public void testBadFlag_equalsForm() throws Exception {
    scratch.file("test/BUILD");
    reporter.removeHandler(failFastHandler);

    OptionsParsingException e =
        assertThrows(
            OptionsParsingException.class,
            () -> parseStarlarkOptions("--//fake_flag=blahblahblah"));

    assertThat(e).hasMessageThat().contains("Error loading option //fake_flag");
  }

  // test --fake_flag value
  @Test
  public void testBadFlag_spaceForm() throws Exception {
    scratch.file("test/BUILD");
    reporter.removeHandler(failFastHandler);

    OptionsParsingException e =
        assertThrows(
            OptionsParsingException.class,
            () -> parseStarlarkOptions("--//fake_flag blahblahblah"));

    assertThat(e).hasMessageThat().contains("Error loading option //fake_flag");
  }

  // test --fake_flag
  @Test
  public void testBadFlag_boolForm() throws Exception {
    scratch.file("test/BUILD");
    reporter.removeHandler(failFastHandler);

    OptionsParsingException e =
        assertThrows(OptionsParsingException.class, () -> parseStarlarkOptions("--//fake_flag"));

    assertThat(e).hasMessageThat().contains("Error loading option //fake_flag");
  }

  // test -flag=value (Note - there's currently no way in real life to allow single dash long form
  // options.)
  @Test
  public void testSingleDash_notAllowed() throws Exception {
    optionsParser.setAllowSingleDashLongOptions(false);

    writeBasicIntFlag();

    OptionsParsingResult result = parseStarlarkOptions("-//test:my_int_setting=666");

    assertThat(result.getStarlarkOptions()).isEmpty();
    assertThat(result.getResidue()).containsExactly("-//test:my_int_setting=666");
  }

  // test --non_flag_setting=value
  @Test
  public void testNonFlagParsing() throws Exception {
    setSkylarkSemanticsOptions("--experimental_build_setting_api=True");

    scratch.file(
        "test/build_setting.bzl",
        "def _build_setting_impl(ctx):",
        "  return []",
        "int_flag = rule(",
        "  implementation = _build_setting_impl,",
        "  build_setting = config.int(flag=False)",
        ")");
    scratch.file(
        "test/BUILD",
        "load('//test:build_setting.bzl', 'int_flag')",
        "int_flag(name = 'my_int_setting', build_setting_default = 42)");

    OptionsParsingException e =
        assertThrows(
            OptionsParsingException.class,
            () -> parseStarlarkOptions("--//test:my_int_setting=666"));

    assertThat(e).hasMessageThat().isEqualTo("Unrecognized option: //test:my_int_setting=666");
  }

  private void writeBasicBoolFlag() throws Exception {
    writeBuildSetting("bool", "True", true);
  }

  // test --bool_flag
  @Test
  public void testBooleanFlag() throws Exception {
    writeBasicBoolFlag();

    OptionsParsingResult result = parseStarlarkOptions("--//test:my_bool_setting");

    assertThat(result.getStarlarkOptions()).hasSize(1);
    assertThat(result.getStarlarkOptions().get("//test:my_bool_setting")).isEqualTo(true);
    assertThat(result.getResidue()).isEmpty();
  }

  // test --nobool_flag
  @Test
  public void testNoPrefixedBooleanFlag() throws Exception {
    writeBasicBoolFlag();

    OptionsParsingResult result = parseStarlarkOptions("--no//test:my_bool_setting");

    assertThat(result.getStarlarkOptions()).hasSize(1);
    assertThat(result.getStarlarkOptions().get("//test:my_bool_setting")).isEqualTo(false);
    assertThat(result.getResidue()).isEmpty();
  }

  // test --noint_flag
  @Test
  public void testNoPrefixedNonBooleanFlag() throws Exception {
    writeBasicIntFlag();

    OptionsParsingException e =
        assertThrows(
            OptionsParsingException.class, () -> parseStarlarkOptions("--no//test:my_int_setting"));

    assertThat(e)
        .hasMessageThat()
        .isEqualTo("Illegal use of 'no' prefix on non-boolean option: //test:my_int_setting");
  }

  // test --int_flag
  @Test
  public void testFlagWithoutValue() throws Exception {
    writeBasicIntFlag();

    OptionsParsingException e =
        assertThrows(
            OptionsParsingException.class, () -> parseStarlarkOptions("--//test:my_int_setting"));

    assertThat(e).hasMessageThat().isEqualTo("Expected value after --//test:my_int_setting");
  }

  // test --flag --flag
  @Test
  public void testRepeatFlagLastOneWins() throws Exception {
    writeBasicIntFlag();

    OptionsParsingResult result =
        parseStarlarkOptions("--//test:my_int_setting=4 --//test:my_int_setting=7");

    assertThat(result.getStarlarkOptions()).hasSize(1);
    assertThat(result.getStarlarkOptions().get("//test:my_int_setting")).isEqualTo(7);
    assertThat(result.getResidue()).isEmpty();
  }

  // test --flagA=valueA --flagB=valueB
  @Test
  public void testMultipleFlags() throws Exception {
    setSkylarkSemanticsOptions("--experimental_build_setting_api=True");

    scratch.file(
        "test/build_setting.bzl",
        "def _build_setting_impl(ctx):",
        "  return []",
        "int_flag = rule(",
        "  implementation = _build_setting_impl,",
        "  build_setting = config.int(flag=True)",
        ")");
    scratch.file(
        "test/BUILD",
        "load('//test:build_setting.bzl', 'int_flag')",
        "int_flag(name = 'my_int_setting', build_setting_default = 42)",
        "int_flag(name = 'my_other_int_setting', build_setting_default = 77)");

    OptionsParsingResult result =
        parseStarlarkOptions("--//test:my_int_setting=0 --//test:my_other_int_setting=0");

    assertThat(result.getResidue()).isEmpty();
    assertThat(result.getStarlarkOptions()).hasSize(2);
    assertThat(result.getStarlarkOptions().get("//test:my_int_setting")).isEqualTo(0);
    assertThat(result.getStarlarkOptions().get("//test:my_other_int_setting")).isEqualTo(0);
  }

  // test --non_build_setting
  @Test
  public void testNonBuildSetting() throws Exception {
    scratch.file(
        "test/rules.bzl",
        "def _impl(ctx):",
        "  return []",
        "my_rule = rule(",
        "  implementation = _impl,",
        ")");
    scratch.file("test/BUILD", "load('//test:rules.bzl', 'my_rule')", "my_rule(name = 'my_rule')");
    OptionsParsingException e =
        assertThrows(OptionsParsingException.class, () -> parseStarlarkOptions("--//test:my_rule"));
    assertThat(e).hasMessageThat().isEqualTo("Unrecognized option: //test:my_rule");
  }

  // test --non_rule_configured_target
  @Test
  public void testNonRuleConfiguredTarget() throws Exception {
    scratch.file(
        "test/BUILD",
        "genrule(",
        "  name = 'my_gen',",
        "  srcs = ['x.in'],",
        "  outs = ['x.cc'],",
        "  cmd = '$(locations :tool) $< >$@',",
        "  tools = [':tool'],",
        ")",
        "cc_library(name = 'tool-dep')");
    OptionsParsingException e =
        assertThrows(OptionsParsingException.class, () -> parseStarlarkOptions("--//test:x.in"));
    assertThat(e).hasMessageThat().isEqualTo("Unrecognized option: //test:x.in");
  }

  // test --int_flag=non_int_value
  @Test
  public void testWrongValueType_int() throws Exception {
    writeBasicIntFlag();

    OptionsParsingException e =
        assertThrows(
            OptionsParsingException.class,
            () -> parseStarlarkOptions("--//test:my_int_setting=woohoo"));

    assertThat(e)
        .hasMessageThat()
        .isEqualTo("While parsing option //test:my_int_setting=woohoo: 'woohoo' is not a int");
  }

  // test --bool_flag=non_bool_value
  @Test
  public void testWrongValueType_bool() throws Exception {
    writeBasicBoolFlag();

    OptionsParsingException e =
        assertThrows(
            OptionsParsingException.class,
            () -> parseStarlarkOptions("--//test:my_bool_setting=woohoo"));

    assertThat(e)
        .hasMessageThat()
        .isEqualTo("While parsing option //test:my_bool_setting=woohoo: 'woohoo' is not a boolean");
  }
}
