// Copyright 2017 The Bazel Authors. All rights reserved.
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

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.fail;

import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ListMultimap;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ServerDirectories;
import com.google.devtools.build.lib.bazel.rules.BazelRulesModule;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.runtime.BlazeOptionHandler.RcChunkOfArgs;
import com.google.devtools.build.lib.runtime.proto.InvocationPolicyOuterClass.InvocationPolicy;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;
import com.google.devtools.common.options.OptionsProvider;
import com.google.devtools.common.options.TestOptions;
import java.util.Arrays;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests {@link BlazeOptionHandler}.
 *
 * <p>Avoids testing anything that is controlled by the {@link BlazeCommandDispatcher}, for
 * isolation. As a part of this, this test intentionally avoids testing how errors and informational
 * messages are logged, to minimize dependence on the ui, and only checks for the existence of these
 * messages.
 */
@RunWith(JUnit4.class)
public class BlazeOptionHandlerTest {

  private final Scratch scratch = new Scratch();
  private final StoredEventHandler eventHandler = new StoredEventHandler();
  private OptionsParser parser;
  private BlazeRuntime runtime;
  private BlazeOptionHandler optionHandler;

  @Before
  public void initStuff() throws Exception {
    parser =
        OptionsParser.newOptionsParser(
            ImmutableList.of(TestOptions.class, CommonCommandOptions.class, ClientOptions.class));
    parser.setAllowResidue(true);
    String productName = TestConstants.PRODUCT_NAME;
    ServerDirectories serverDirectories =
        new ServerDirectories(scratch.dir("install_base"), scratch.dir("output_base"));
    this.runtime =
        new BlazeRuntime.Builder()
            .setFileSystem(scratch.getFileSystem())
            .setServerDirectories(serverDirectories)
            .setProductName(productName)
            .setStartupOptionsProvider(
                OptionsParser.newOptionsParser(BlazeServerStartupOptions.class))
            .addBlazeModule(new BazelRulesModule())
            .build();
    this.runtime.overrideCommands(ImmutableList.of(new C0Command()));

    BlazeDirectories directories =
        new BlazeDirectories(serverDirectories, scratch.dir("workspace"), productName);
    runtime.initWorkspace(directories, /*binTools=*/ null);
  }

  private void makeFixedPointExpandingConfigOptionHandler() {
    optionHandler =
        BlazeOptionHandler.getHandler(
            runtime,
            runtime.getWorkspace(),
            new C0Command(),
            C0Command.class.getAnnotation(Command.class),
            parser,
            InvocationPolicy.getDefaultInstance(),
            false);
  }

  private void makeInPlaceExpandingConfigOptionHandler() {
    optionHandler =
        BlazeOptionHandler.getHandler(
            runtime,
            runtime.getWorkspace(),
            new C0Command(),
            C0Command.class.getAnnotation(Command.class),
            parser,
            InvocationPolicy.getDefaultInstance(),
            true);
  }

  @Command(
    name = "c0",
    shortDescription = "c0 desc",
    help = "c0 help",
    options = {TestOptions.class}
  )
  private static class C0Command implements BlazeCommand {
    @Override
    public ExitCode exec(CommandEnvironment env, OptionsProvider options) {
      throw new UnsupportedOperationException();
    }

    @Override
    public void editOptions(OptionsParser optionsParser) {}
  }

  private ListMultimap<String, RcChunkOfArgs> structuredArgsFrom2SimpleRcsWithOnlyResidue() {
    ListMultimap<String, RcChunkOfArgs> structuredArgs = ArrayListMultimap.create();
    // first add all lines of rc1, then rc2, to simulate a simple, import free, 2 rc file setup.
    structuredArgs.put("c0", new RcChunkOfArgs("rc1", ImmutableList.of("a")));
    structuredArgs.put("c0:config", new RcChunkOfArgs("rc1", ImmutableList.of("b")));

    structuredArgs.put("common", new RcChunkOfArgs("rc2", ImmutableList.of("c")));
    structuredArgs.put("c0", new RcChunkOfArgs("rc2", ImmutableList.of("d", "e")));
    structuredArgs.put("c1:other", new RcChunkOfArgs("rc2", ImmutableList.of("f", "g")));
    return structuredArgs;
  }

  private ListMultimap<String, RcChunkOfArgs> structuredArgsFrom2SimpleRcsWithFlags() {
    ListMultimap<String, RcChunkOfArgs> structuredArgs = ArrayListMultimap.create();
    structuredArgs.put(
        "c0", new RcChunkOfArgs("rc1", ImmutableList.of("--test_multiple_string=foo")));
    structuredArgs.put(
        "c0:config", new RcChunkOfArgs("rc1", ImmutableList.of("--test_multiple_string=config")));

    structuredArgs.put(
        "common", new RcChunkOfArgs("rc2", ImmutableList.of("--test_multiple_string=common")));
    structuredArgs.put(
        "c0", new RcChunkOfArgs("rc2", ImmutableList.of("--test_multiple_string=bar")));
    structuredArgs.put(
        "c1:other", new RcChunkOfArgs("rc2", ImmutableList.of("--test_multiple_string=other")));
    return structuredArgs;
  }

  private ListMultimap<String, RcChunkOfArgs> structuredArgsFromImportedRcsWithOnlyResidue() {
    ListMultimap<String, RcChunkOfArgs> structuredArgs = ArrayListMultimap.create();
    // first add all lines of rc1, then rc2, but then jump back to 1 as if rc2 was loaded in an
    // import statement halfway through rc1.
    structuredArgs.put("c0", new RcChunkOfArgs("rc1", ImmutableList.of("a")));
    structuredArgs.put("c0:config", new RcChunkOfArgs("rc1", ImmutableList.of("b")));

    structuredArgs.put("common", new RcChunkOfArgs("rc2", ImmutableList.of("c")));
    structuredArgs.put("c0", new RcChunkOfArgs("rc2", ImmutableList.of("d", "e")));
    structuredArgs.put("c1:other", new RcChunkOfArgs("rc2", ImmutableList.of("f", "g")));

    structuredArgs.put("c0", new RcChunkOfArgs("rc1", ImmutableList.of("h")));
    return structuredArgs;
  }

  private void testStructureRcOptionsAndConfigs_argumentless() throws Exception {
    ListMultimap<String, RcChunkOfArgs> structuredRc =
        BlazeOptionHandler.structureRcOptionsAndConfigs(
            eventHandler,
            Arrays.asList("rc1", "rc2"),
            Arrays.asList(),
            ImmutableSet.of("c0", "c1"));
    assertThat(structuredRc).isEmpty();
    assertThat(eventHandler.isEmpty()).isTrue();
  }

  @Test
  public void testStructureRcOptionsAndConfigs_argumentless_fixedPoint() throws Exception {
    makeFixedPointExpandingConfigOptionHandler();
    testStructureRcOptionsAndConfigs_argumentless();
  }

  @Test
  public void testStructureRcOptionsAndConfigs_argumentless_inPlace() throws Exception {
    makeInPlaceExpandingConfigOptionHandler();
    testStructureRcOptionsAndConfigs_argumentless();
  }

  private void testStructureRcOptionsAndConfigs_configOnly() throws Exception {
    BlazeOptionHandler.structureRcOptionsAndConfigs(
        eventHandler,
        Arrays.asList("rc1", "rc2"),
        Arrays.asList(new ClientOptions.OptionOverride(0, "c0:none", "a")),
        ImmutableSet.of("c0"));
    assertThat(eventHandler.isEmpty()).isTrue();
  }

  @Test
  public void testStructureRcOptionsAndConfigs_configOnly_fixedPoint() throws Exception {
    makeFixedPointExpandingConfigOptionHandler();
    testStructureRcOptionsAndConfigs_configOnly();
  }

  @Test
  public void testStructureRcOptionsAndConfigs_configOnly_inPlace() throws Exception {
    makeInPlaceExpandingConfigOptionHandler();
    testStructureRcOptionsAndConfigs_configOnly();
  }

  private void testStructureRcOptionsAndConfigs_invalidCommand() throws Exception {
    BlazeOptionHandler.structureRcOptionsAndConfigs(
        eventHandler,
        Arrays.asList("rc1", "rc2"),
        Arrays.asList(new ClientOptions.OptionOverride(0, "c1", "a")),
        ImmutableSet.of("c0"));
    assertThat(eventHandler.getEvents())
        .contains(
            Event.warn("while reading option defaults file 'rc1':\n  invalid command name 'c1'."));
  }

  @Test
  public void testStructureRcOptionsAndConfigs_invalidCommand_fixedPoint() throws Exception {
    makeFixedPointExpandingConfigOptionHandler();
    testStructureRcOptionsAndConfigs_invalidCommand();
  }

  @Test
  public void testStructureRcOptionsAndConfigs_invalidCommand_inPlace() throws Exception {
    makeInPlaceExpandingConfigOptionHandler();
    testStructureRcOptionsAndConfigs_invalidCommand();
  }

  private void testStructureRcOptionsAndConfigs_twoRcs() throws Exception {
    ListMultimap<String, RcChunkOfArgs> structuredRc =
        BlazeOptionHandler.structureRcOptionsAndConfigs(
            eventHandler,
            Arrays.asList("rc1", "rc2"),
            Arrays.asList(
                new ClientOptions.OptionOverride(0, "c0", "a"),
                new ClientOptions.OptionOverride(0, "c0:config", "b"),
                new ClientOptions.OptionOverride(1, "common", "c"),
                new ClientOptions.OptionOverride(1, "c0", "d"),
                new ClientOptions.OptionOverride(1, "c0", "e"),
                new ClientOptions.OptionOverride(1, "c1:other", "f"),
                new ClientOptions.OptionOverride(1, "c1:other", "g")),
            ImmutableSet.of("c0", "c1"));
    assertThat(structuredRc).isEqualTo(structuredArgsFrom2SimpleRcsWithOnlyResidue());
    assertThat(eventHandler.isEmpty()).isTrue();
  }

  @Test
  public void testStructureRcOptionsAndConfigs_twoRcs_fixedPoint() throws Exception {
    makeFixedPointExpandingConfigOptionHandler();
    testStructureRcOptionsAndConfigs_twoRcs();
  }

  @Test
  public void testStructureRcOptionsAndConfigs_twoRcs_inPlace() throws Exception {
    makeInPlaceExpandingConfigOptionHandler();
    testStructureRcOptionsAndConfigs_twoRcs();
  }

  private void testStructureRcOptionsAndConfigs_importedRcs() throws Exception {
    ListMultimap<String, RcChunkOfArgs> structuredRc =
        BlazeOptionHandler.structureRcOptionsAndConfigs(
            eventHandler,
            Arrays.asList("rc1", "rc2"),
            Arrays.asList(
                new ClientOptions.OptionOverride(0, "c0", "a"),
                new ClientOptions.OptionOverride(0, "c0:config", "b"),
                new ClientOptions.OptionOverride(1, "common", "c"),
                new ClientOptions.OptionOverride(1, "c0", "d"),
                new ClientOptions.OptionOverride(1, "c0", "e"),
                new ClientOptions.OptionOverride(1, "c1:other", "f"),
                new ClientOptions.OptionOverride(1, "c1:other", "g"),
                new ClientOptions.OptionOverride(0, "c0", "h")),
            ImmutableSet.of("c0", "c1"));
    assertThat(structuredRc).isEqualTo(structuredArgsFromImportedRcsWithOnlyResidue());
    assertThat(eventHandler.isEmpty()).isTrue();
  }

  @Test
  public void testStructureRcOptionsAndConfigs_importedRcs_fixedPoint() throws Exception {
    makeFixedPointExpandingConfigOptionHandler();
    testStructureRcOptionsAndConfigs_importedRcs();
  }

  @Test
  public void testStructureRcOptionsAndConfigs_importedRcs_inPlace() throws Exception {
    makeInPlaceExpandingConfigOptionHandler();
    testStructureRcOptionsAndConfigs_importedRcs();
  }

  private void testStructureRcOptionsAndConfigs_badOverrideIndex() throws Exception {
    ListMultimap<String, RcChunkOfArgs> structuredRc =
        BlazeOptionHandler.structureRcOptionsAndConfigs(
            eventHandler,
            Arrays.asList("rc1", "rc2"),
            Arrays.asList(
                new ClientOptions.OptionOverride(0, "c0", "a"),
                new ClientOptions.OptionOverride(0, "c0:config", "b"),
                new ClientOptions.OptionOverride(2, "c4:other", "z"),
                new ClientOptions.OptionOverride(-1, "c3:other", "q"),
                new ClientOptions.OptionOverride(1, "common", "c"),
                new ClientOptions.OptionOverride(1, "c0", "d"),
                new ClientOptions.OptionOverride(1, "c0", "e"),
                new ClientOptions.OptionOverride(1, "c1:other", "f"),
                new ClientOptions.OptionOverride(1, "c1:other", "g")),
            ImmutableSet.of("c0", "c1"));
    assertThat(structuredRc).isEqualTo(structuredArgsFrom2SimpleRcsWithOnlyResidue());
    assertThat(eventHandler.getEvents())
        .containsAllOf(
            Event.warn("inconsistency in generated command line args. Ignoring bogus argument\n"),
            Event.warn("inconsistency in generated command line args. Ignoring bogus argument\n"));
  }

  @Test
  public void testStructureRcOptionsAndConfigs_badOverrideIndex_fixedPoint() throws Exception {
    makeFixedPointExpandingConfigOptionHandler();
    testStructureRcOptionsAndConfigs_badOverrideIndex();
  }

  @Test
  public void testStructureRcOptionsAndConfigs_badOverrideIndex_inPlace() throws Exception {
    makeInPlaceExpandingConfigOptionHandler();
    testStructureRcOptionsAndConfigs_badOverrideIndex();
  }

  private void testParseRcOptions_empty() throws Exception {
    optionHandler.parseRcOptions(eventHandler, ArrayListMultimap.create());
    assertThat(eventHandler.getEvents()).isEmpty();
    assertThat(parser.getResidue()).isEmpty();
  }

  @Test
  public void testParseRcOptions_empty_fixedPoint() throws Exception {
    makeFixedPointExpandingConfigOptionHandler();
    testParseRcOptions_empty();
  }

  @Test
  public void testParseRcOptions_empty_inPlace() throws Exception {
    makeInPlaceExpandingConfigOptionHandler();
    testParseRcOptions_empty();
  }

  private void testParseRcOptions_flatRcs_residue() throws Exception {
    optionHandler.parseRcOptions(eventHandler, structuredArgsFrom2SimpleRcsWithOnlyResidue());
    assertThat(eventHandler.getEvents()).isEmpty();
    assertThat(parser.getResidue()).containsExactly("c", "a", "d", "e").inOrder();
  }

  @Test
  public void testParseRcOptions_flatRcs_residue_fixedPoint() throws Exception {
    makeFixedPointExpandingConfigOptionHandler();
    testParseRcOptions_flatRcs_residue();
  }

  @Test
  public void testParseRcOptions_flatRcs_residue_inPlace() throws Exception {
    makeInPlaceExpandingConfigOptionHandler();
    testParseRcOptions_flatRcs_residue();
  }

  private void testParseRcOptions_flatRcs_flags() throws Exception {
    optionHandler.parseRcOptions(eventHandler, structuredArgsFrom2SimpleRcsWithFlags());
    assertThat(eventHandler.getEvents()).isEmpty();
    TestOptions options = parser.getOptions(TestOptions.class);
    assertThat(options).isNotNull();
    assertThat(options.testMultipleString).containsExactly("common", "foo", "bar").inOrder();
  }

  @Test
  public void testParseRcOptions_flatRcs_flags_fixedPoint() throws Exception {
    makeFixedPointExpandingConfigOptionHandler();
    testParseRcOptions_flatRcs_flags();
  }

  @Test
  public void testParseRcOptions_flatRcs_flags_inPlace() throws Exception {
    makeInPlaceExpandingConfigOptionHandler();
    testParseRcOptions_flatRcs_flags();
  }

  private void testParseRcOptions_importedRcs_residue() throws Exception {
    optionHandler.parseRcOptions(eventHandler, structuredArgsFromImportedRcsWithOnlyResidue());
    assertThat(eventHandler.getEvents()).isEmpty();
    assertThat(parser.getResidue()).containsExactly("c", "a", "d", "e", "h").inOrder();
  }

  @Test
  public void testParseRcOptions_importedRcs_residue_fixedPoint() throws Exception {
    makeFixedPointExpandingConfigOptionHandler();
    testParseRcOptions_importedRcs_residue();
  }

  @Test
  public void testParseRcOptions_importedRcs_residue_inPlace() throws Exception {
    makeInPlaceExpandingConfigOptionHandler();
    testParseRcOptions_importedRcs_residue();
  }

  private void testExpandConfigOptions_configless() throws Exception {
    optionHandler.expandConfigOptions(eventHandler, structuredArgsFrom2SimpleRcsWithOnlyResidue());
    assertThat(parser.getResidue()).isEmpty();
  }

  @Test
  public void testExpandConfigOptions_configless_fixedPoint() throws Exception {
    makeFixedPointExpandingConfigOptionHandler();
    testExpandConfigOptions_configless();
  }

  @Test
  public void testExpandConfigOptions_configless_inPlace() throws Exception {
    makeInPlaceExpandingConfigOptionHandler();
    testExpandConfigOptions_configless();
  }

  private void testExpandConfigOptions_withConfig() throws Exception {
    parser.parse("--config=config");
    optionHandler.expandConfigOptions(eventHandler, structuredArgsFrom2SimpleRcsWithOnlyResidue());
    assertThat(parser.getResidue()).containsExactly("b");
    assertThat(optionHandler.getRcfileNotes())
        .containsExactly("Found applicable config definition c0:config in file rc1: b");
  }

  @Test
  public void testExpandConfigOptions_withConfig_fixedPoint() throws Exception {
    makeFixedPointExpandingConfigOptionHandler();
    testExpandConfigOptions_withConfig();
  }

  @Test
  public void testExpandConfigOptions_withConfig_inPlace() throws Exception {
    makeInPlaceExpandingConfigOptionHandler();
    testExpandConfigOptions_withConfig();
  }

  private void testExpandConfigOptions_withConfigForUnapplicableCommand() throws Exception {
    parser.parse("--config=other");
    optionHandler.expandConfigOptions(eventHandler, structuredArgsFrom2SimpleRcsWithOnlyResidue());
    assertThat(parser.getResidue()).isEmpty();
    assertThat(optionHandler.getRcfileNotes()).isEmpty();
  }

  @Test
  public void testExpandConfigOptions_withConfigForUnapplicableCommand_fixedPoint()
      throws Exception {
    makeFixedPointExpandingConfigOptionHandler();
    testExpandConfigOptions_withConfigForUnapplicableCommand();
    assertThat(eventHandler.getEvents())
        .contains(Event.warn("Config values are not defined in any .rc file: other"));
  }

  @Test
  public void testExpandConfigOptions_withConfigForUnapplicableCommand_inPlace() throws Exception {
    makeInPlaceExpandingConfigOptionHandler();
    testExpandConfigOptions_withConfigForUnapplicableCommand();
    assertThat(eventHandler.getEvents())
        .contains(Event.warn("Config value other is not defined in any .rc file"));
  }

  private void testAllowUndefinedConfig() throws Exception {
    parser.parse("--config=invalid", "--allow_undefined_configs");
    optionHandler.expandConfigOptions(eventHandler, ArrayListMultimap.create());
    assertThat(parser.getResidue()).isEmpty();
    assertThat(optionHandler.getRcfileNotes()).isEmpty();
  }

  @Test
  public void testAllowUndefinedConfig_fixedPoint() throws Exception {
    makeFixedPointExpandingConfigOptionHandler();
    testAllowUndefinedConfig();
    assertThat(eventHandler.getEvents())
        .contains(Event.warn("Config values are not defined in any .rc file: invalid"));
  }

  @Test
  public void testAllowUndefinedConfig_inPlace() throws Exception {
    makeInPlaceExpandingConfigOptionHandler();
    testAllowUndefinedConfig();
    assertThat(eventHandler.getEvents())
        .contains(Event.warn("Config value invalid is not defined in any .rc file"));
  }

  private void testNoAllowUndefinedConfig() throws OptionsParsingException {
    parser.parse("--config=invalid", "--noallow_undefined_configs");
    optionHandler.expandConfigOptions(eventHandler, ArrayListMultimap.create());
  }

  @Test
  public void testNoAllowUndefinedConfig_fixedPoint() {
    makeFixedPointExpandingConfigOptionHandler();
    try {
      testNoAllowUndefinedConfig();
      fail();
    } catch (OptionsParsingException e) {
      assertThat(e)
          .hasMessageThat()
          .contains("Config values are not defined in any .rc file: invalid");
    }
  }

  @Test
  public void testNoAllowUndefinedConfig_inPlace() {
    makeInPlaceExpandingConfigOptionHandler();
    try {
      testNoAllowUndefinedConfig();
      fail();
    } catch (OptionsParsingException e) {
      assertThat(e)
          .hasMessageThat()
          .contains("Config value invalid is not defined in any .rc file");
    }
  }

  private void testParseOptions_argless() {
    optionHandler.parseOptions(ImmutableList.of("c0"), eventHandler);
    assertThat(eventHandler.getEvents()).isEmpty();
    assertThat(parser.getResidue()).isEmpty();
    assertThat(optionHandler.getRcfileNotes()).isEmpty();
  }

  @Test
  public void testParseOptions_argless_fixedPoint() {
    makeFixedPointExpandingConfigOptionHandler();
    testParseOptions_argless();
  }

  @Test
  public void testParseOptions_argless_inPlace() {
    makeInPlaceExpandingConfigOptionHandler();
    testParseOptions_argless();
  }

  private void testParseOptions_residue() {
    optionHandler.parseOptions(ImmutableList.of("c0", "res"), eventHandler);
    assertThat(eventHandler.getEvents()).isEmpty();
    assertThat(parser.getResidue()).contains("res");
    assertThat(optionHandler.getRcfileNotes()).isEmpty();
  }

  @Test
  public void testParseOptions_residue_fixedPoint() {
    makeFixedPointExpandingConfigOptionHandler();
    testParseOptions_residue();
  }

  @Test
  public void testParseOptions_residue_inPlace() {
    makeInPlaceExpandingConfigOptionHandler();
    testParseOptions_residue();
  }

  private void testParseOptions_explicitOption() {
    optionHandler.parseOptions(
        ImmutableList.of("c0", "--test_multiple_string=explicit"), eventHandler);
    assertThat(eventHandler.getEvents()).isEmpty();
    assertThat(parser.getResidue()).isEmpty();
    assertThat(optionHandler.getRcfileNotes()).isEmpty();
    TestOptions options = parser.getOptions(TestOptions.class);
    assertThat(options).isNotNull();
    assertThat(options.testMultipleString).containsExactly("explicit");
  }

  @Test
  public void testParseOptions_explicitOption_fixedPoint() {
    makeFixedPointExpandingConfigOptionHandler();
    testParseOptions_explicitOption();
  }

  @Test
  public void testParseOptions_explicitOption_inPlace() {
    makeInPlaceExpandingConfigOptionHandler();
    testParseOptions_explicitOption();
  }

  private void testParseOptions_rcOption() {
    optionHandler.parseOptions(
        ImmutableList.of(
            "c0",
            "--default_override=0:c0=--test_multiple_string=rc_a",
            "--default_override=0:c0=--test_multiple_string=rc_b",
            "--rc_source=/somewhere/.blazerc"),
        eventHandler);
    assertThat(eventHandler.getEvents()).isEmpty();
    assertThat(parser.getResidue()).isEmpty();
    // Check that multiple options in the same rc chunk are collapsed into 1 announce_rc entry.
    assertThat(optionHandler.getRcfileNotes())
        .containsExactly(
            "Reading rc options for 'c0' from /somewhere/.blazerc:\n"
                + "  'c0' options: --test_multiple_string=rc_a --test_multiple_string=rc_b");
    TestOptions options = parser.getOptions(TestOptions.class);
    assertThat(options).isNotNull();
    assertThat(options.testMultipleString).containsExactly("rc_a", "rc_b");
  }

  @Test
  public void testParseOptions_rcOption_fixedPoint() {
    makeFixedPointExpandingConfigOptionHandler();
    testParseOptions_rcOption();
  }

  @Test
  public void testParseOptions_rcOption_inPlace() {
    makeInPlaceExpandingConfigOptionHandler();
    testParseOptions_rcOption();
  }

  private void testParseOptions_multipleRcs() {
    optionHandler.parseOptions(
        ImmutableList.of(
            "c0",
            "--default_override=0:c0=--test_multiple_string=rc1_a",
            "--default_override=1:c0=--test_multiple_string=rc2",
            "--default_override=0:c0=--test_multiple_string=rc1_b",
            "--rc_source=/somewhere/.blazerc",
            "--rc_source=/some/other/.blazerc"),
        eventHandler);
    assertThat(eventHandler.getEvents()).isEmpty();
    assertThat(parser.getResidue()).isEmpty();
    assertThat(optionHandler.getRcfileNotes())
        .containsExactly(
            "Reading rc options for 'c0' from /somewhere/.blazerc:\n"
                + "  'c0' options: --test_multiple_string=rc1_a",
            "Reading rc options for 'c0' from /some/other/.blazerc:\n"
                + "  'c0' options: --test_multiple_string=rc2",
            "Reading rc options for 'c0' from /somewhere/.blazerc:\n"
                + "  'c0' options: --test_multiple_string=rc1_b");
    TestOptions options = parser.getOptions(TestOptions.class);
    assertThat(options).isNotNull();
    assertThat(options.testMultipleString).containsExactly("rc1_a", "rc2", "rc1_b").inOrder();
  }

  @Test
  public void testParseOptions_multipleRcs_fixedPoint() {
    makeFixedPointExpandingConfigOptionHandler();
    testParseOptions_multipleRcs();
  }

  @Test
  public void testParseOptions_multipleRcs_inPlace() {
    makeInPlaceExpandingConfigOptionHandler();
    testParseOptions_multipleRcs();
  }

  private void testParseOptions_multipleRcsWithMultipleCommands() {
    optionHandler.parseOptions(
        ImmutableList.of(
            "c0",
            "--default_override=0:c0=--test_multiple_string=rc1_a",
            "--default_override=1:c0=--test_multiple_string=rc2",
            "--default_override=1:common=--test_multiple_string=rc2_common",
            "--default_override=0:c0=--test_multiple_string=rc1_b",
            "--default_override=0:common=--test_multiple_string=rc1_common",
            "--rc_source=/somewhere/.blazerc",
            "--rc_source=/some/other/.blazerc"),
        eventHandler);
    assertThat(eventHandler.getEvents()).isEmpty();
    assertThat(parser.getResidue()).isEmpty();
    assertThat(optionHandler.getRcfileNotes())
        .containsExactly(
            "Reading rc options for 'c0' from /some/other/.blazerc:\n"
                + "  Inherited 'common' options: --test_multiple_string=rc2_common",
            "Reading rc options for 'c0' from /somewhere/.blazerc:\n"
                + "  Inherited 'common' options: --test_multiple_string=rc1_common",
            "Reading rc options for 'c0' from /somewhere/.blazerc:\n"
                + "  'c0' options: --test_multiple_string=rc1_a",
            "Reading rc options for 'c0' from /some/other/.blazerc:\n"
                + "  'c0' options: --test_multiple_string=rc2",
            "Reading rc options for 'c0' from /somewhere/.blazerc:\n"
                + "  'c0' options: --test_multiple_string=rc1_b");
    TestOptions options = parser.getOptions(TestOptions.class);
    assertThat(options).isNotNull();
    assertThat(options.testMultipleString)
        .containsExactly("rc2_common", "rc1_common", "rc1_a", "rc2", "rc1_b")
        .inOrder();
  }

  @Test
  public void testParseOptions_multipleRcsWithMultipleCommands_fixedPoint() {
    makeFixedPointExpandingConfigOptionHandler();
    testParseOptions_multipleRcsWithMultipleCommands();
  }

  @Test
  public void testParseOptions_multipleRcsWithMultipleCommands_inPlace() {
    makeInPlaceExpandingConfigOptionHandler();
    testParseOptions_multipleRcsWithMultipleCommands();
  }

  private void testParseOptions_rcOptionAndExplicit() {
    optionHandler.parseOptions(
        ImmutableList.of(
            "c0",
            "--default_override=0:c0=--test_multiple_string=rc",
            "--rc_source=/somewhere/.blazerc",
            "--test_multiple_string=explicit"),
        eventHandler);
    assertThat(eventHandler.getEvents()).isEmpty();
    assertThat(parser.getResidue()).isEmpty();
    assertThat(optionHandler.getRcfileNotes())
        .containsExactly(
            "Reading rc options for 'c0' from /somewhere/.blazerc:\n"
                + "  'c0' options: --test_multiple_string=rc");
    TestOptions options = parser.getOptions(TestOptions.class);
    assertThat(options).isNotNull();
    assertThat(options.testMultipleString).containsExactly("rc", "explicit").inOrder();
  }

  @Test
  public void testParseOptions_rcOptionAndExplicit_fixedPoint() {
    makeFixedPointExpandingConfigOptionHandler();
    testParseOptions_rcOptionAndExplicit();
  }

  @Test
  public void testParseOptions_rcOptionAndExplicit_inPlace() {
    makeInPlaceExpandingConfigOptionHandler();
    testParseOptions_rcOptionAndExplicit();
  }

  private void testParseOptions_multiCommandRcOptionAndExplicit() {
    optionHandler.parseOptions(
        ImmutableList.of(
            "c0",
            "--default_override=0:c0=--test_multiple_string=rc_c0_1",
            "--default_override=0:common=--test_multiple_string=rc_common",
            "--default_override=0:c0=--test_multiple_string=rc_c0_2",
            "--rc_source=/somewhere/.blazerc",
            "--test_multiple_string=explicit"),
        eventHandler);
    assertThat(eventHandler.getEvents()).isEmpty();
    assertThat(parser.getResidue()).isEmpty();
    assertThat(optionHandler.getRcfileNotes())
        .containsExactly(
            "Reading rc options for 'c0' from /somewhere/.blazerc:\n"
                + "  Inherited 'common' options: --test_multiple_string=rc_common",
            "Reading rc options for 'c0' from /somewhere/.blazerc:\n"
                + "  'c0' options: --test_multiple_string=rc_c0_1 --test_multiple_string=rc_c0_2");
    TestOptions options = parser.getOptions(TestOptions.class);
    assertThat(options).isNotNull();
    assertThat(options.testMultipleString)
        .containsExactly("rc_common", "rc_c0_1", "rc_c0_2", "explicit")
        .inOrder();
  }

  @Test
  public void testParseOptions_multiCommandRcOptionAndExplicit_fixedPoint() {
    makeFixedPointExpandingConfigOptionHandler();
    testParseOptions_multiCommandRcOptionAndExplicit();
  }

  @Test
  public void testParseOptions_multiCommandRcOptionAndExplicit_inPlace() {
    makeInPlaceExpandingConfigOptionHandler();
    testParseOptions_multiCommandRcOptionAndExplicit();
  }

  private void testParseOptions_multipleRcsWithMultipleCommandsPlusExplicitOption() {
    optionHandler.parseOptions(
        ImmutableList.of(
            "c0",
            "--default_override=0:c0=--test_multiple_string=rc1_a",
            "--default_override=1:c0=--test_multiple_string=rc2",
            "--test_multiple_string=explicit",
            "--default_override=1:common=--test_multiple_string=rc2_common",
            "--default_override=0:c0=--test_multiple_string=rc1_b",
            "--default_override=0:common=--test_multiple_string=rc1_common",
            "--rc_source=/somewhere/.blazerc",
            "--rc_source=/some/other/.blazerc"),
        eventHandler);
    assertThat(eventHandler.getEvents()).isEmpty();
    assertThat(parser.getResidue()).isEmpty();
    assertThat(optionHandler.getRcfileNotes())
        .containsExactly(
            "Reading rc options for 'c0' from /some/other/.blazerc:\n"
                + "  Inherited 'common' options: --test_multiple_string=rc2_common",
            "Reading rc options for 'c0' from /somewhere/.blazerc:\n"
                + "  Inherited 'common' options: --test_multiple_string=rc1_common",
            "Reading rc options for 'c0' from /somewhere/.blazerc:\n"
                + "  'c0' options: --test_multiple_string=rc1_a",
            "Reading rc options for 'c0' from /some/other/.blazerc:\n"
                + "  'c0' options: --test_multiple_string=rc2",
            "Reading rc options for 'c0' from /somewhere/.blazerc:\n"
                + "  'c0' options: --test_multiple_string=rc1_b");
    TestOptions options = parser.getOptions(TestOptions.class);
    assertThat(options).isNotNull();
    assertThat(options.testMultipleString)
        .containsExactly("rc2_common", "rc1_common", "rc1_a", "rc2", "rc1_b", "explicit")
        .inOrder();
  }

  @Test
  public void testParseOptions_multipleRcsWithMultipleCommandsPlusExplicitOption_fixedPoint() {
    makeFixedPointExpandingConfigOptionHandler();
    testParseOptions_multipleRcsWithMultipleCommandsPlusExplicitOption();
  }

  @Test
  public void testParseOptions_multipleRcsWithMultipleCommandsPlusExplicitOption_inPlace() {
    makeInPlaceExpandingConfigOptionHandler();
    testParseOptions_multipleRcsWithMultipleCommandsPlusExplicitOption();
  }

  private void testParseOptions_explicitConfig() {
    optionHandler.parseOptions(
        ImmutableList.of(
            "c0",
            "--default_override=0:c0=--test_multiple_string=rc",
            "--default_override=0:c0:conf=--test_multiple_string=config",
            "--rc_source=/somewhere/.blazerc",
            "--test_multiple_string=explicit",
            "--config=conf"),
        eventHandler);
    assertThat(eventHandler.getEvents()).isEmpty();
    assertThat(parser.getResidue()).isEmpty();
    assertThat(optionHandler.getRcfileNotes())
        .containsExactly(
            "Reading rc options for 'c0' from /somewhere/.blazerc:\n"
                + "  'c0' options: --test_multiple_string=rc",
            "Found applicable config definition c0:conf in file /somewhere/.blazerc: "
                + "--test_multiple_string=config");
  }

  @Test
  public void testParseOptions_explicitConfig_fixedPoint() {
    makeFixedPointExpandingConfigOptionHandler();
    testParseOptions_explicitConfig();

    // "config" is lower priority (occurs earlier in the list) than "explicit" in the fix-point
    // expansion, despite --config=conf occurring later.
    TestOptions options = parser.getOptions(TestOptions.class);
    assertThat(options).isNotNull();
    assertThat(options.testMultipleString).containsExactly("rc", "config", "explicit").inOrder();
  }

  @Test
  public void testParseOptions_explicitConfig_inPlace() {
    makeInPlaceExpandingConfigOptionHandler();
    testParseOptions_explicitConfig();

    // "config" is expanded from --config=conf, which occurs last.
    TestOptions options = parser.getOptions(TestOptions.class);
    assertThat(options).isNotNull();
    assertThat(options.testMultipleString).containsExactly("rc", "explicit", "config").inOrder();
  }

  private void testParseOptions_rcSpecifiedConfig() {
    optionHandler.parseOptions(
        ImmutableList.of(
            "c0",
            "--default_override=0:c0=--config=conf",
            "--default_override=0:c0=--test_multiple_string=rc",
            "--default_override=0:c0:conf=--test_multiple_string=config",
            "--rc_source=/somewhere/.blazerc",
            "--test_multiple_string=explicit"),
        eventHandler);
    assertThat(eventHandler.getEvents()).isEmpty();
    assertThat(parser.getResidue()).isEmpty();
    assertThat(optionHandler.getRcfileNotes())
        .containsExactly(
            "Reading rc options for 'c0' from /somewhere/.blazerc:\n"
                + "  'c0' options: --config=conf --test_multiple_string=rc",
            "Found applicable config definition c0:conf in file /somewhere/.blazerc: "
                + "--test_multiple_string=config");
  }

  @Test
  public void testParseOptions_rcSpecifiedConfig_fixedPoint() {
    makeFixedPointExpandingConfigOptionHandler();
    testParseOptions_rcSpecifiedConfig();

    // "config" is higher priority (occurs later in the list) than "rc" in the fix-point
    // expansion, despite --config=conf occurring before the explicit mention of "rc".
    TestOptions options = parser.getOptions(TestOptions.class);
    assertThat(options).isNotNull();
    assertThat(options.testMultipleString).containsExactly("rc", "config", "explicit").inOrder();
  }

  @Test
  public void testParseOptions_rcSpecifiedConfig_inPlace() {
    makeInPlaceExpandingConfigOptionHandler();
    testParseOptions_rcSpecifiedConfig();

    // "config" is expanded from --config=conf, which occurs before the explicit mention of "rc".
    TestOptions options = parser.getOptions(TestOptions.class);
    assertThat(options).isNotNull();
    assertThat(options.testMultipleString).containsExactly("config", "rc", "explicit").inOrder();
  }

  private void testParseOptions_recursiveConfig() {
    optionHandler.parseOptions(
        ImmutableList.of(
            "c0",
            "--default_override=0:c0=--config=conf",
            "--default_override=0:c0=--test_multiple_string=rc",
            "--default_override=0:c0:other=--test_multiple_string=other",
            "--default_override=0:c0:conf=--test_multiple_string=config1",
            "--default_override=0:c0:conf=--config=other",
            "--default_override=0:common:other=--test_multiple_string=othercommon",
            "--rc_source=/somewhere/.blazerc",
            "--test_multiple_string=explicit"),
        eventHandler);
    assertThat(eventHandler.getEvents()).isEmpty();
    assertThat(parser.getResidue()).isEmpty();
    assertThat(optionHandler.getRcfileNotes())
        .containsExactly(
            "Reading rc options for 'c0' from /somewhere/.blazerc:\n"
                + "  'c0' options: --config=conf --test_multiple_string=rc",
            "Found applicable config definition c0:conf in file /somewhere/.blazerc: "
                + "--test_multiple_string=config1 --config=other",
            "Found applicable config definition common:other in file /somewhere/.blazerc: "
                + "--test_multiple_string=othercommon",
            "Found applicable config definition c0:other in file /somewhere/.blazerc: "
                + "--test_multiple_string=other");
  }

  @Test
  public void testParseOptions_recursiveConfig_fixedPoint() {
    makeFixedPointExpandingConfigOptionHandler();
    testParseOptions_recursiveConfig();

    // The 2nd config, --config=other, is expanded after the config that added it.
    TestOptions options = parser.getOptions(TestOptions.class);
    assertThat(options).isNotNull();
    assertThat(options.testMultipleString)
        .containsExactly("rc", "config1", "othercommon", "other", "explicit")
        .inOrder();
  }

  @Test
  public void testParseOptions_recursiveConfig_inPlace() {
    makeInPlaceExpandingConfigOptionHandler();
    testParseOptions_recursiveConfig();

    // The 2nd config, --config=other, is added by --config=conf after conf adds its own value.
    TestOptions options = parser.getOptions(TestOptions.class);
    assertThat(options).isNotNull();
    assertThat(options.testMultipleString)
        .containsExactly("config1", "othercommon", "other", "rc", "explicit")
        .inOrder();
  }

  private void parseComplexConfigOrderCommandLine() {
    optionHandler.parseOptions(
        ImmutableList.of(
            "c0",
            "--default_override=0:c0=--test_multiple_string=rc1",
            "--default_override=0:c0=--config=foo",
            "--default_override=0:c0=--test_multiple_string=rc2",
            "--default_override=0:common:baz=--test_multiple_string=baz1",
            "--default_override=0:c0:baz=--test_multiple_string=baz2",
            "--default_override=0:common:foo=--test_multiple_string=foo1",
            "--default_override=0:common:foo=--config=bar",
            "--default_override=0:c0:foo=--test_multiple_string=foo3",
            "--default_override=0:common:foo=--test_multiple_string=foo2",
            "--default_override=0:c0:foo=--test_multiple_string=foo4",
            "--default_override=0:common:bar=--test_multiple_string=bar1",
            "--default_override=0:c0:bar=--test_multiple_string=bar2",
            "--rc_source=/somewhere/.blazerc",
            "--test_multiple_string=explicit1",
            "--config=baz",
            "--test_multiple_string=explicit2"),
        eventHandler);
    assertThat(eventHandler.getEvents()).isEmpty();
    assertThat(parser.getResidue()).isEmpty();
  }

  @Test
  public void testParseOptions_complexConfigOrder_fixedPoint() {
    makeFixedPointExpandingConfigOptionHandler();
    parseComplexConfigOrderCommandLine();
    assertThat(optionHandler.getRcfileNotes())
        .containsExactly(
            "Reading rc options for 'c0' from /somewhere/.blazerc:\n  'c0' options: "
                + "--test_multiple_string=rc1 --config=foo --test_multiple_string=rc2",
            "Found applicable config definition common:foo in file /somewhere/.blazerc: "
                + "--test_multiple_string=foo1 --config=bar --test_multiple_string=foo2",
            "Found applicable config definition common:baz in file /somewhere/.blazerc: "
                + "--test_multiple_string=baz1",
            "Found applicable config definition c0:foo in file /somewhere/.blazerc: "
                + "--test_multiple_string=foo3 --test_multiple_string=foo4",
            "Found applicable config definition c0:baz in file /somewhere/.blazerc: "
                + "--test_multiple_string=baz2",
            "Found applicable config definition common:bar in file /somewhere/.blazerc: "
                + "--test_multiple_string=bar1",
            "Found applicable config definition c0:bar in file /somewhere/.blazerc: "
                + "--test_multiple_string=bar2");
    TestOptions options = parser.getOptions(TestOptions.class);
    assertThat(options).isNotNull();
    assertThat(options.testMultipleString)
        .containsExactly(
            "rc1",
            "rc2",
            "foo1",
            "foo2",
            "baz1",
            "foo3",
            "foo4",
            "baz2",
            "bar1",
            "bar2",
            "explicit1",
            "explicit2")
        .inOrder();
  }

  @Test
  public void testParseOptions_complexConfigOrder_inPlace() {
    makeInPlaceExpandingConfigOptionHandler();
    parseComplexConfigOrderCommandLine();
    assertThat(optionHandler.getRcfileNotes())
        .containsExactly(
            "Reading rc options for 'c0' from /somewhere/.blazerc:\n  'c0' options: "
                + "--test_multiple_string=rc1 --config=foo --test_multiple_string=rc2",
            "Found applicable config definition common:foo in file /somewhere/.blazerc: "
                + "--test_multiple_string=foo1 --config=bar --test_multiple_string=foo2",
            "Found applicable config definition common:bar in file /somewhere/.blazerc: "
                + "--test_multiple_string=bar1",
            "Found applicable config definition c0:bar in file /somewhere/.blazerc: "
                + "--test_multiple_string=bar2",
            "Found applicable config definition c0:foo in file /somewhere/.blazerc: "
                + "--test_multiple_string=foo3 --test_multiple_string=foo4",
            "Found applicable config definition common:baz in file /somewhere/.blazerc: "
                + "--test_multiple_string=baz1",
            "Found applicable config definition c0:baz in file /somewhere/.blazerc: "
                + "--test_multiple_string=baz2");
    TestOptions options = parser.getOptions(TestOptions.class);
    assertThat(options).isNotNull();
    assertThat(options.testMultipleString)
        .containsExactly(
            "rc1",
            "foo1",
            "bar1",
            "bar2",
            "foo2",
            "foo3",
            "foo4",
            "rc2",
            "explicit1",
            "baz1",
            "baz2",
            "explicit2")
        .inOrder();
  }

  private void parseConfigDoubleRecursionCommandLine() {
    optionHandler.parseOptions(
        ImmutableList.of(
            "c0",
            "--default_override=0:c0=--config=foo",
            "--default_override=0:c0=--test_multiple_string=rc",
            "--default_override=0:c0:foo=--test_multiple_string=foo",
            "--default_override=0:c0:foo=--config=bar",
            "--default_override=0:c0:foo=--config=bar",
            "--default_override=0:c0:bar=--test_multiple_string=bar",
            "--rc_source=/somewhere/.blazerc",
            "--test_multiple_string=explicit"),
        eventHandler);
    assertThat(parser.getResidue()).isEmpty();
  }

  @Test
  public void testParseOptions_repeatSubConfig_fixedPoint() {
    makeFixedPointExpandingConfigOptionHandler();
    parseConfigDoubleRecursionCommandLine();
    assertThat(eventHandler.getEvents()).isEmpty();
    assertThat(optionHandler.getRcfileNotes())
        .containsExactly(
            "Reading rc options for 'c0' from /somewhere/.blazerc:\n"
                + "  'c0' options: --config=foo --test_multiple_string=rc",
            "Found applicable config definition c0:foo in file /somewhere/.blazerc: "
                + "--test_multiple_string=foo --config=bar --config=bar",
            "Found applicable config definition c0:bar in file /somewhere/.blazerc: "
                + "--test_multiple_string=bar");
    TestOptions options = parser.getOptions(TestOptions.class);
    assertThat(options).isNotNull();
    // Bar is not repeated, it was already expanded once and the fixed point expansion
    // does not attempt to expand configs a second time.
    assertThat(options.testMultipleString)
        .containsExactly("rc", "foo", "bar", "explicit")
        .inOrder();
  }

  @Test
  public void testParseOptions_repeatSubConfig_inPlace() {
    makeInPlaceExpandingConfigOptionHandler();
    parseConfigDoubleRecursionCommandLine();
    assertThat(eventHandler.getEvents())
        .containsExactly(
            Event.warn(
                "The following configs were expanded more than once: [bar]. For repeatable flags, "
                    + "repeats are counted twice and may lead to unexpected behavior."));
    assertThat(optionHandler.getRcfileNotes())
        .containsExactly(
            "Reading rc options for 'c0' from /somewhere/.blazerc:\n"
                + "  'c0' options: --config=foo --test_multiple_string=rc",
            "Found applicable config definition c0:foo in file /somewhere/.blazerc: "
                + "--test_multiple_string=foo --config=bar --config=bar",
            "Found applicable config definition c0:bar in file /somewhere/.blazerc: "
                + "--test_multiple_string=bar",
            "Found applicable config definition c0:bar in file /somewhere/.blazerc: "
                + "--test_multiple_string=bar");
    TestOptions options = parser.getOptions(TestOptions.class);
    assertThat(options).isNotNull();
    // Bar is repeated, since it was included twice.
    assertThat(options.testMultipleString)
        .containsExactly("foo", "bar", "bar", "rc", "explicit")
        .inOrder();
  }

  private void parseRepeatConfigs() {
    optionHandler.parseOptions(
        ImmutableList.of(
            "c0",
            "--default_override=0:c0:foo=--test_multiple_string=foo",
            "--default_override=0:c0:foo=--config=bar",
            "--default_override=0:c0:bar=--test_multiple_string=bar",
            "--default_override=0:c0:baz=--test_multiple_string=baz",
            "--rc_source=/somewhere/.blazerc",
            "--config=foo",
            "--config=baz",
            "--config=foo",
            "--config=bar"),
        eventHandler);
    assertThat(parser.getResidue()).isEmpty();
  }

  @Test
  public void testParseOptions_repeatConfig_fixedPoint() {
    makeFixedPointExpandingConfigOptionHandler();
    parseRepeatConfigs();
    assertThat(eventHandler.getEvents()).isEmpty();
    assertThat(optionHandler.getRcfileNotes())
        .containsExactly(
            "Found applicable config definition c0:foo in file /somewhere/.blazerc: "
                + "--test_multiple_string=foo --config=bar",
            "Found applicable config definition c0:baz in file /somewhere/.blazerc: "
                + "--test_multiple_string=baz",
            "Found applicable config definition c0:bar in file /somewhere/.blazerc: "
                + "--test_multiple_string=bar");
    TestOptions options = parser.getOptions(TestOptions.class);
    assertThat(options).isNotNull();
    // Foo and bar are not repeated, despite repeat mentions.
    assertThat(options.testMultipleString).containsExactly("foo", "baz", "bar").inOrder();
  }

  @Test
  public void testParseOptions_repeatConfig_inPlace() {
    makeInPlaceExpandingConfigOptionHandler();
    parseRepeatConfigs();
    assertThat(eventHandler.getEvents())
        .containsExactly(
            Event.warn(
                "The following configs were expanded more than once: [foo, bar]. For repeatable "
                    + "flags, repeats are counted twice and may lead to unexpected behavior."));
    assertThat(optionHandler.getRcfileNotes())
        .containsExactly(
            "Found applicable config definition c0:foo in file /somewhere/.blazerc: "
                + "--test_multiple_string=foo --config=bar",
            "Found applicable config definition c0:bar in file /somewhere/.blazerc: "
                + "--test_multiple_string=bar",
            "Found applicable config definition c0:baz in file /somewhere/.blazerc: "
                + "--test_multiple_string=baz",
            "Found applicable config definition c0:foo in file /somewhere/.blazerc: "
                + "--test_multiple_string=foo --config=bar",
            "Found applicable config definition c0:bar in file /somewhere/.blazerc: "
                + "--test_multiple_string=bar",
            "Found applicable config definition c0:bar in file /somewhere/.blazerc: "
                + "--test_multiple_string=bar");
    TestOptions options = parser.getOptions(TestOptions.class);
    assertThat(options).isNotNull();
    // Bar is repeated, since it was included twice.
    assertThat(options.testMultipleString)
        .containsExactly("foo", "bar", "baz", "foo", "bar", "bar")
        .inOrder();
  }

  private void parseConfigCycleLength1CommandLine() {
    optionHandler.parseOptions(
        ImmutableList.of(
            "c0",
            "--default_override=0:c0=--config=foo",
            "--default_override=0:c0=--test_multiple_string=rc",
            "--default_override=0:c0:foo=--test_multiple_string=foo",
            "--default_override=0:c0:foo=--config=foo",
            "--rc_source=/somewhere/.blazerc",
            "--test_multiple_string=explicit"),
        eventHandler);
  }

  @Test
  public void testParseOptions_configCycleLength1_fixedPoint() {
    makeFixedPointExpandingConfigOptionHandler();
    parseConfigCycleLength1CommandLine();
    assertThat(eventHandler.getEvents()).isEmpty();
    assertThat(parser.getResidue()).isEmpty();
    assertThat(optionHandler.getRcfileNotes())
        .containsExactly(
            "Reading rc options for 'c0' from /somewhere/.blazerc:\n"
                + "  'c0' options: --config=foo --test_multiple_string=rc",
            "Found applicable config definition c0:foo in file /somewhere/.blazerc: "
                + "--test_multiple_string=foo --config=foo");
    TestOptions options = parser.getOptions(TestOptions.class);
    assertThat(options).isNotNull();
    // The cycle is not expanded, since foo was already expanded once and the fixed point expansion
    // does not attempt to expand configs a second time.
    assertThat(options.testMultipleString).containsExactly("rc", "foo", "explicit").inOrder();
  }

  @Test
  public void testParseOptions_configCycleLength1_inPlace() {
    makeInPlaceExpandingConfigOptionHandler();
    parseConfigCycleLength1CommandLine();
    assertThat(eventHandler.getEvents())
        .contains(
            Event.error(
                "Config expansion has a cycle: config value foo expands to itself, see "
                    + "inheritance chain [foo]"));
  }

  private void parseConfigCycleLength2CommandLine() {
    optionHandler.parseOptions(
        ImmutableList.of(
            "c0",
            "--default_override=0:c0=--config=foo",
            "--default_override=0:c0=--test_multiple_string=rc",
            "--default_override=0:c0:foo=--test_multiple_string=foo",
            "--default_override=0:c0:foo=--config=bar",
            "--default_override=0:c0:bar=--test_multiple_string=bar",
            "--default_override=0:c0:bar=--config=foo",
            "--rc_source=/somewhere/.blazerc",
            "--test_multiple_string=explicit"),
        eventHandler);
  }

  @Test
  public void testParseOptions_configCycleLength2_fixedPoint() {
    makeFixedPointExpandingConfigOptionHandler();
    parseConfigCycleLength2CommandLine();
    assertThat(eventHandler.getEvents()).isEmpty();
    assertThat(parser.getResidue()).isEmpty();
    assertThat(optionHandler.getRcfileNotes())
        .containsExactly(
            "Reading rc options for 'c0' from /somewhere/.blazerc:\n"
                + "  'c0' options: --config=foo --test_multiple_string=rc",
            "Found applicable config definition c0:foo in file /somewhere/.blazerc: "
                + "--test_multiple_string=foo --config=bar",
            "Found applicable config definition c0:bar in file /somewhere/.blazerc: "
                + "--test_multiple_string=bar --config=foo");
    TestOptions options = parser.getOptions(TestOptions.class);
    assertThat(options).isNotNull();
    // The cycle is not expanded, since foo was already expanded once and the fixed point expansion
    // does not attempt to expand configs a second time.
    assertThat(options.testMultipleString)
        .containsExactly("rc", "foo", "bar", "explicit")
        .inOrder();
  }

  @Test
  public void testParseOptions_configCycleLength2_inPlace() {
    makeInPlaceExpandingConfigOptionHandler();
    parseConfigCycleLength2CommandLine();
    assertThat(eventHandler.getEvents())
        .contains(
            Event.error(
                "Config expansion has a cycle: config value foo expands to itself, see "
                    + "inheritance chain [foo, bar]"));
  }

  private void recursivelyIncludedRepeatConfigCommandLine() {
    optionHandler.parseOptions(
        ImmutableList.of(
            "c0",
            "--default_override=0:c0=--config=other",
            "--default_override=0:c0=--config=conf",
            "--default_override=0:c0=--test_multiple_string=rc",
            "--default_override=0:c0:other=--test_multiple_string=other",
            "--default_override=0:c0:conf=--test_multiple_string=config1",
            "--default_override=0:c0:conf=--config=other",
            "--default_override=0:common:other=--test_multiple_string=othercommon",
            "--rc_source=/somewhere/.blazerc",
            "--test_multiple_string=explicit"),
        eventHandler);
    assertThat(parser.getResidue()).isEmpty();
  }

  @Test
  public void testParseOptions_recursiveConfigWasAlreadyPresent_fixedPoint() {
    makeFixedPointExpandingConfigOptionHandler();
    recursivelyIncludedRepeatConfigCommandLine();
    assertThat(eventHandler.getEvents()).isEmpty();

    // The 2nd config, --config=other, is expanded at the same time as --config=conf, since they are
    // both initially present. The "common" definition is therefore first. other is not reexpanded
    // when it is added by --config=conf, since it was already included.
    assertThat(optionHandler.getRcfileNotes())
        .containsExactly(
            "Reading rc options for 'c0' from /somewhere/.blazerc:\n"
                + "  'c0' options: --config=other --config=conf --test_multiple_string=rc",
            "Found applicable config definition common:other in file /somewhere/.blazerc: "
                + "--test_multiple_string=othercommon",
            "Found applicable config definition c0:conf in file /somewhere/.blazerc: "
                + "--test_multiple_string=config1 --config=other",
            "Found applicable config definition c0:other in file /somewhere/.blazerc: "
                + "--test_multiple_string=other");
    TestOptions options = parser.getOptions(TestOptions.class);
    assertThat(options).isNotNull();
    assertThat(options.testMultipleString)
        .containsExactly("rc", "othercommon", "other", "config1", "explicit")
        .inOrder();
  }

  @Test
  public void testParseOptions_recursiveConfigWasAlreadyPresent_inPlace() {
    makeInPlaceExpandingConfigOptionHandler();
    recursivelyIncludedRepeatConfigCommandLine();
    assertThat(eventHandler.getEvents())
        .containsExactly(
            Event.warn(
                "The following configs were expanded more than once: [other]. For repeatable "
                    + "flags, repeats are counted twice and may lead to unexpected behavior."));

    // The 2nd config, --config=other, is expanded twice at the same time as --config=conf,
    // both initially present. The "common" definition is therefore first. other is expanded twice.
    assertThat(optionHandler.getRcfileNotes())
        .containsExactly(
            "Reading rc options for 'c0' from /somewhere/.blazerc:\n"
                + "  'c0' options: --config=other --config=conf --test_multiple_string=rc",
            "Found applicable config definition common:other in file /somewhere/.blazerc: "
                + "--test_multiple_string=othercommon",
            "Found applicable config definition c0:other in file /somewhere/.blazerc: "
                + "--test_multiple_string=other",
            "Found applicable config definition c0:conf in file /somewhere/.blazerc: "
                + "--test_multiple_string=config1 --config=other",
            "Found applicable config definition common:other in file /somewhere/.blazerc: "
                + "--test_multiple_string=othercommon",
            "Found applicable config definition c0:other in file /somewhere/.blazerc: "
                + "--test_multiple_string=other");
    TestOptions options = parser.getOptions(TestOptions.class);
    assertThat(options).isNotNull();
    assertThat(options.testMultipleString)
        .containsExactly(
            "othercommon", "other", "config1", "othercommon", "other", "rc", "explicit")
        .inOrder();
  }

  private static final ImmutableList<String> GREEK_ALPHABET_CHAIN =
      ImmutableList.of(
          "--default_override=0:c0:alpha=--test_multiple_string=alpha",
          "--default_override=0:c0:alpha=--config=beta",
          "--default_override=0:c0:beta=--test_multiple_string=beta",
          "--default_override=0:c0:beta=--config=gamma",
          "--default_override=0:c0:gamma=--test_multiple_string=gamma",
          "--default_override=0:c0:gamma=--config=delta",
          "--default_override=0:c0:delta=--test_multiple_string=delta",
          "--default_override=0:c0:delta=--config=epsilon",
          "--default_override=0:c0:epsilon=--test_multiple_string=epsilon",
          "--default_override=0:c0:epsilon=--config=zeta",
          "--default_override=0:c0:zeta=--test_multiple_string=zeta",
          "--default_override=0:c0:zeta=--config=eta",
          "--default_override=0:c0:eta=--test_multiple_string=eta",
          "--default_override=0:c0:eta=--config=theta",
          "--default_override=0:c0:theta=--test_multiple_string=theta",
          "--default_override=0:c0:theta=--config=iota",
          "--default_override=0:c0:iota=--test_multiple_string=iota",
          "--default_override=0:c0:iota=--config=kappa",
          "--default_override=0:c0:kappa=--test_multiple_string=kappa",
          "--default_override=0:c0:kappa=--config=lambda",
          "--default_override=0:c0:lambda=--test_multiple_string=lambda",
          "--default_override=0:c0:lambda=--config=mu",
          "--default_override=0:c0:mu=--test_multiple_string=mu");

  private void testParseOptions_longChainOfConfigs_12long() {
    ImmutableList<String> args =
        ImmutableList.<String>builder()
            .add("c0")
            .addAll(GREEK_ALPHABET_CHAIN)
            .add("--rc_source=/somewhere/.blazerc")
            .add("--config=alpha")
            .build();

    optionHandler.parseOptions(args, eventHandler);
    assertThat(parser.getResidue()).isEmpty();
    assertThat(optionHandler.getRcfileNotes())
        .containsExactly(
            "Found applicable config definition c0:alpha in file /somewhere/.blazerc: "
                + "--test_multiple_string=alpha --config=beta",
            "Found applicable config definition c0:beta in file /somewhere/.blazerc: "
                + "--test_multiple_string=beta --config=gamma",
            "Found applicable config definition c0:gamma in file /somewhere/.blazerc: "
                + "--test_multiple_string=gamma --config=delta",
            "Found applicable config definition c0:delta in file /somewhere/.blazerc: "
                + "--test_multiple_string=delta --config=epsilon",
            "Found applicable config definition c0:epsilon in file /somewhere/.blazerc: "
                + "--test_multiple_string=epsilon --config=zeta",
            "Found applicable config definition c0:zeta in file /somewhere/.blazerc: "
                + "--test_multiple_string=zeta --config=eta",
            "Found applicable config definition c0:eta in file /somewhere/.blazerc: "
                + "--test_multiple_string=eta --config=theta",
            "Found applicable config definition c0:theta in file /somewhere/.blazerc: "
                + "--test_multiple_string=theta --config=iota",
            "Found applicable config definition c0:iota in file /somewhere/.blazerc: "
                + "--test_multiple_string=iota --config=kappa",
            "Found applicable config definition c0:kappa in file /somewhere/.blazerc: "
                + "--test_multiple_string=kappa --config=lambda",
            "Found applicable config definition c0:lambda in file /somewhere/.blazerc: "
                + "--test_multiple_string=lambda --config=mu",
            "Found applicable config definition c0:mu in file /somewhere/.blazerc: "
                + "--test_multiple_string=mu");
    TestOptions options = parser.getOptions(TestOptions.class);
    assertThat(options).isNotNull();
    assertThat(options.testMultipleString)
        .containsExactly(
            "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa",
            "lambda", "mu")
        .inOrder();
  }

  @Test
  public void testParseOptions_longChain_FixedPoint() {
    makeFixedPointExpandingConfigOptionHandler();
    testParseOptions_longChainOfConfigs_12long();
    assertThat(eventHandler.getEvents()).isEmpty();
  }

  @Test
  public void testParseOptions_longChain_inPlace() {
    makeInPlaceExpandingConfigOptionHandler();
    testParseOptions_longChainOfConfigs_12long();
    // Expect only one warning, we don't want multiple warnings for the same chain.
    assertThat(eventHandler.getEvents())
        .containsExactly(
            Event.warn(
                "There is a recursive chain of configs 12 configs long: [alpha, beta, gamma, "
                    + "delta, epsilon, zeta, eta, theta, iota, kappa, lambda, mu]. This seems "
                    + "excessive, and might be hiding errors."));
  }

  private void testParseOptions_twoLongChains() {
    ImmutableList<String> args =
        ImmutableList.<String>builder()
            .add("c0")
            .addAll(GREEK_ALPHABET_CHAIN)
            .add("--rc_source=/somewhere/.blazerc")
            .add("--config=alpha")
            .add("--config=gamma")
            .build();

    optionHandler.parseOptions(args, eventHandler);
    assertThat(parser.getResidue()).isEmpty();
  }

  @Test
  public void testParseOptions_2LongChains_FixedPoint() {
    makeFixedPointExpandingConfigOptionHandler();
    testParseOptions_twoLongChains();
    // In fixed point, the repetition --config=gamma does not led to a new expansion, but it does
    // mean that gamma gets expanded in the first round, so the ordering is weird.
    TestOptions options = parser.getOptions(TestOptions.class);
    assertThat(options).isNotNull();
    assertThat(options.testMultipleString)
        .containsExactly(
            "alpha", "gamma", "beta", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa",
            "lambda", "mu")
        .inOrder();
    assertThat(eventHandler.getEvents()).isEmpty();
  }

  @Test
  public void testParseOptions_2LongChains_inPlace() {
    makeInPlaceExpandingConfigOptionHandler();
    testParseOptions_twoLongChains();
    // Expect the second --config=gamma to have started a second chain, and get warnings about both.
    TestOptions options = parser.getOptions(TestOptions.class);
    assertThat(options).isNotNull();
    assertThat(options.testMultipleString)
        .containsExactly(
            "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa",
            "lambda", "mu", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa",
            "lambda", "mu")
        .inOrder();
    assertThat(eventHandler.getEvents())
        .containsExactly(
            Event.warn(
                "There is a recursive chain of configs 12 configs long: [alpha, beta, gamma, "
                    + "delta, epsilon, zeta, eta, theta, iota, kappa, lambda, mu]. This seems "
                    + "excessive, and might be hiding errors."),
            Event.warn(
                "There is a recursive chain of configs 10 configs long: [gamma, delta, epsilon, "
                    + "zeta, eta, theta, iota, kappa, lambda, mu]. This seems excessive, "
                    + "and might be hiding errors."),
            Event.warn(
                "The following configs were expanded more than once: [gamma, delta, epsilon, zeta, "
                    + "eta, theta, iota, kappa, lambda, mu]. For repeatable flags, repeats are "
                    + "counted twice and may lead to unexpected behavior."));
  }

  private void testWarningFlag() {
    optionHandler.parseOptions(
        ImmutableList.of(
            "c0",
            "--unconditional_warning",
            "You are forcing this warning to print for no apparent reason"),
        eventHandler);
    assertThat(eventHandler.getEvents())
        .containsExactly(
            Event.warn("You are forcing this warning to print for no apparent reason"));
    assertThat(parser.getResidue()).isEmpty();
    assertThat(optionHandler.getRcfileNotes()).isEmpty();
  }

  @Test
  public void testWarningFlag_fixedPoint() {
    makeFixedPointExpandingConfigOptionHandler();
    testWarningFlag();
  }

  @Test
  public void testWarningFlag_inPlace() {
    makeInPlaceExpandingConfigOptionHandler();
    testWarningFlag();
  }

  private void testWarningFlag_byConfig_notTriggered() {
    optionHandler.parseOptions(
        ImmutableList.of(
            "c0",
            "--default_override=0:c0:conf=--unconditional_warning="
                + "config \"conf\" is deprecated, please stop using!",
            "--rc_source=/somewhere/.blazerc"),
        eventHandler);
    assertThat(eventHandler.getEvents()).isEmpty();
    assertThat(parser.getResidue()).isEmpty();
    assertThat(optionHandler.getRcfileNotes()).isEmpty();
  }

  @Test
  public void testWarningFlag_byConfig_notTriggered_fixedPoint() {
    makeFixedPointExpandingConfigOptionHandler();
    testWarningFlag_byConfig_notTriggered();
  }

  @Test
  public void testWarningFlag_byConfig_notTriggered_inPlace() {
    makeInPlaceExpandingConfigOptionHandler();
    testWarningFlag_byConfig_notTriggered();
  }

  private void testWarningFlag_byConfig_triggered() {
    optionHandler.parseOptions(
        ImmutableList.of(
            "c0",
            "--config=conf",
            "--default_override=0:c0:conf=--unconditional_warning="
                + "config \"conf\" is deprecated, please stop using!",
            "--rc_source=/somewhere/.blazerc"),
        eventHandler);
    assertThat(eventHandler.getEvents())
        .containsExactly(Event.warn("config \"conf\" is deprecated, please stop using!"));
    assertThat(parser.getResidue()).isEmpty();
    assertThat(optionHandler.getRcfileNotes())
        .containsExactly(
            "Found applicable config definition c0:conf in file /somewhere/.blazerc: "
                + "--unconditional_warning=config \"conf\" is deprecated, please stop using!");
  }

  @Test
  public void testWarningFlag_byConfig_triggered_fixedPoint() {
    makeFixedPointExpandingConfigOptionHandler();
    testWarningFlag_byConfig_triggered();
  }

  @Test
  public void testWarningFlag_byConfig_triggered_inPlace() {
    makeInPlaceExpandingConfigOptionHandler();
    testWarningFlag_byConfig_triggered();
  }

  @Test
  public void testConfigAfterExplicit_fixedPoint() {
    makeFixedPointExpandingConfigOptionHandler();
    optionHandler.parseOptions(
        ImmutableList.of(
            "c0",
            "--test_string=explicitValue",
            "--config=conf",
            "--default_override=0:c0:conf=--test_string=fromConf",
            "--rc_source=/somewhere/.blazerc"),
        eventHandler);
    TestOptions parseResult = parser.getOptions(TestOptions.class);
    assertThat(eventHandler.getEvents()).isEmpty();
    // The fact that --config=conf comes after the explicit value does not matter
    assertThat(parseResult.testString).isEqualTo("explicitValue");
    assertThat(optionHandler.getRcfileNotes())
        .containsExactly(
            "Found applicable config definition c0:conf in file /somewhere/.blazerc: "
                + "--test_string=fromConf");
  }

  @Test
  public void testConfigAfterExplicit_inPlace() {
    makeInPlaceExpandingConfigOptionHandler();
    optionHandler.parseOptions(
        ImmutableList.of(
            "c0",
            "--test_string=explicitValue",
            "--config=conf",
            "--default_override=0:c0:conf=--test_string=fromConf",
            "--rc_source=/somewhere/.blazerc"),
        eventHandler);
    TestOptions parseResult = parser.getOptions(TestOptions.class);
    // In the in-place expansion, the config's expansion has precedence, but issues a warning since
    // users might not know that their explicit value was overridden.
    assertThat(eventHandler.getEvents())
        .containsExactly(
            Event.warn(
                "option '--config=conf' (source command line options) was expanded and now "
                    + "overrides the explicit option --test_string=explicitValue with "
                    + "--test_string=fromConf"));
    assertThat(parseResult.testString).isEqualTo("fromConf");
    assertThat(optionHandler.getRcfileNotes())
        .containsExactly(
            "Found applicable config definition c0:conf in file /somewhere/.blazerc: "
                + "--test_string=fromConf");
  }

  @Test
  public void testExplicitOverridesConfig_fixedPoint() {
    makeFixedPointExpandingConfigOptionHandler();
    optionHandler.parseOptions(
        ImmutableList.of(
            "c0",
            "--config=conf",
            "--test_string=explicitValue",
            "--default_override=0:c0:conf=--test_string=fromConf",
            "--rc_source=/somewhere/.blazerc"),
        eventHandler);
    TestOptions parseResult = parser.getOptions(TestOptions.class);
    assertThat(eventHandler.getEvents()).isEmpty();
    assertThat(parseResult.testString).isEqualTo("explicitValue");
    assertThat(optionHandler.getRcfileNotes())
        .containsExactly(
            "Found applicable config definition c0:conf in file /somewhere/.blazerc: "
                + "--test_string=fromConf");
  }

  @Test
  public void testExplicitOverridesConfig_inPlace() {
    makeInPlaceExpandingConfigOptionHandler();
    optionHandler.parseOptions(
        ImmutableList.of(
            "c0",
            "--config=conf",
            "--test_string=explicitValue",
            "--default_override=0:c0:conf=--test_string=fromConf",
            "--rc_source=/somewhere/.blazerc"),
        eventHandler);
    TestOptions parseResult = parser.getOptions(TestOptions.class);
    assertThat(eventHandler.getEvents()).isEmpty();
    assertThat(parseResult.testString).isEqualTo("explicitValue");
    assertThat(optionHandler.getRcfileNotes())
        .containsExactly(
            "Found applicable config definition c0:conf in file /somewhere/.blazerc: "
                + "--test_string=fromConf");
  }
}
