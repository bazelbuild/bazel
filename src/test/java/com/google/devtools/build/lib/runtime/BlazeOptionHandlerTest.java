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
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ListMultimap;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;
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

  private StoredEventHandler eventHandler;
  private OptionsParser parser;
  private BlazeOptionHandler optionHandler;

  @Before
  public void setUp() throws Exception {
    ImmutableList<Class<? extends OptionsBase>> optionsClasses =
        ImmutableList.of(TestOptions.class, CommonCommandOptions.class, ClientOptions.class);

    BlazeOptionHandlerTestHelper helper =
        new BlazeOptionHandlerTestHelper(optionsClasses, /* allowResidue= */ true);
    eventHandler = helper.getEventHandler();
    parser = helper.getOptionsParser();
    optionHandler = helper.getOptionHandler();
  }

  private static ListMultimap<String, RcChunkOfArgs> structuredArgsFrom2SimpleRcsWithOnlyResidue() {
    ListMultimap<String, RcChunkOfArgs> structuredArgs = ArrayListMultimap.create();
    // first add all lines of rc1, then rc2, to simulate a simple, import free, 2 rc file setup.
    structuredArgs.put("c0", new RcChunkOfArgs("rc1", ImmutableList.of("a")));
    structuredArgs.put("c0:config", new RcChunkOfArgs("rc1", ImmutableList.of("b")));

    structuredArgs.put("common", new RcChunkOfArgs("rc2", ImmutableList.of("c")));
    structuredArgs.put("c0", new RcChunkOfArgs("rc2", ImmutableList.of("d", "e")));
    structuredArgs.put("c1:other", new RcChunkOfArgs("rc2", ImmutableList.of("f", "g")));
    return structuredArgs;
  }

  private static ListMultimap<String, RcChunkOfArgs> structuredArgsFrom2SimpleRcsWithFlags() {
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

  private static ListMultimap<String, RcChunkOfArgs>
      structuredArgsFromImportedRcsWithOnlyResidue() {
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

  private static ListMultimap<String, RcChunkOfArgs> structuredArgsForDifferentPlatforms() {
    ListMultimap<String, RcChunkOfArgs> structuredArgs = ArrayListMultimap.create();
    structuredArgs.put("c0:linux", new RcChunkOfArgs("rc1", ImmutableList.of("command_linux")));
    structuredArgs.put("c0:windows", new RcChunkOfArgs("rc1", ImmutableList.of("command_windows")));
    structuredArgs.put("c0:macos", new RcChunkOfArgs("rc1", ImmutableList.of("command_macos")));
    structuredArgs.put("c0:freebsd", new RcChunkOfArgs("rc1", ImmutableList.of("command_freebsd")));
    structuredArgs.put("c0:openbsd", new RcChunkOfArgs("rc1", ImmutableList.of("command_openbsd")));
    structuredArgs.put(
        "c0:platform_config",
        new RcChunkOfArgs("rc1", ImmutableList.of("--enable_platform_specific_config")));
    return structuredArgs;
  }

  @Test
  public void testStructureRcOptionsAndConfigs_argumentless() throws Exception {
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
  public void testStructureRcOptionsAndConfigs_configOnly() throws Exception {
    BlazeOptionHandler.structureRcOptionsAndConfigs(
        eventHandler,
        Arrays.asList("rc1", "rc2"),
        Arrays.asList(new ClientOptions.OptionOverride(0, "c0:none", "a")),
        ImmutableSet.of("c0"));
    assertThat(eventHandler.isEmpty()).isTrue();
  }

  @Test
  public void testStructureRcOptionsAndConfigs_invalidCommand() throws Exception {
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
  public void testStructureRcOptionsAndConfigs_twoRcs() throws Exception {
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
  public void testStructureRcOptionsAndConfigs_importedRcs() throws Exception {
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
  public void testStructureRcOptionsAndConfigs_badOverrideIndex() throws Exception {
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
        .containsAtLeast(
            Event.warn("inconsistency in generated command line args. Ignoring bogus argument\n"),
            Event.warn("inconsistency in generated command line args. Ignoring bogus argument\n"));
  }

  @Test
  public void testParseRcOptions_empty() throws Exception {
    optionHandler.parseRcOptions(eventHandler, ArrayListMultimap.create());
    assertThat(eventHandler.getEvents()).isEmpty();
    assertThat(parser.getResidue()).isEmpty();
  }

  @Test
  public void testParseRcOptions_flatRcs_residue() throws Exception {
    optionHandler.parseRcOptions(eventHandler, structuredArgsFrom2SimpleRcsWithOnlyResidue());
    assertThat(eventHandler.getEvents()).isEmpty();
    assertThat(parser.getResidue()).containsExactly("c", "a", "d", "e").inOrder();
  }

  @Test
  public void testParseRcOptions_flatRcs_flags() throws Exception {
    optionHandler.parseRcOptions(eventHandler, structuredArgsFrom2SimpleRcsWithFlags());
    assertThat(eventHandler.getEvents()).isEmpty();
    TestOptions options = parser.getOptions(TestOptions.class);
    assertThat(options).isNotNull();
    assertThat(options.testMultipleString).containsExactly("common", "foo", "bar").inOrder();
  }

  @Test
  public void testParseRcOptions_importedRcs_residue() throws Exception {
    optionHandler.parseRcOptions(eventHandler, structuredArgsFromImportedRcsWithOnlyResidue());
    assertThat(eventHandler.getEvents()).isEmpty();
    assertThat(parser.getResidue()).containsExactly("c", "a", "d", "e", "h").inOrder();
  }

  @Test
  public void testExpandConfigOptions_configless() throws Exception {
    optionHandler.expandConfigOptions(eventHandler, structuredArgsFrom2SimpleRcsWithOnlyResidue());
    assertThat(parser.getResidue()).isEmpty();
  }

  @Test
  public void testExpandConfigOptions_withConfig() throws Exception {
    parser.parse("--config=config");
    optionHandler.expandConfigOptions(eventHandler, structuredArgsFrom2SimpleRcsWithOnlyResidue());
    assertThat(parser.getResidue()).containsExactly("b");
    assertThat(optionHandler.getRcfileNotes())
        .containsExactly("Found applicable config definition c0:config in file rc1: b");
  }

  @Test
  public void testExpandConfigOptions_withPlatformSpecificConfigEnabled() throws Exception {
    parser.parse("--enable_platform_specific_config");
    optionHandler.expandConfigOptions(eventHandler, structuredArgsForDifferentPlatforms());
    switch (OS.getCurrent()) {
      case LINUX:
        assertThat(parser.getResidue()).containsExactly("command_linux");
        break;
      case DARWIN:
        assertThat(parser.getResidue()).containsExactly("command_macos");
        break;
      case WINDOWS:
        assertThat(parser.getResidue()).containsExactly("command_windows");
        break;
      case FREEBSD:
        assertThat(parser.getResidue()).containsExactly("command_freebsd");
        break;
      case OPENBSD:
        assertThat(parser.getResidue()).containsExactly("command_openbsd");
        break;
      default:
        assertThat(parser.getResidue()).isEmpty();
    }
  }

  @Test
  public void testExpandConfigOptions_withPlatformSpecificConfigEnabledInConfig() throws Exception {
    // --enable_platform_specific_config itself will affect the selecting of config sections.
    // Because Bazel expands config sections recursively, we want to make sure it's fine to enable
    // --enable_platform_specific_config via another config section.
    parser.parse("--config=platform_config");
    optionHandler.expandConfigOptions(eventHandler, structuredArgsForDifferentPlatforms());
    switch (OS.getCurrent()) {
      case LINUX:
        assertThat(parser.getResidue()).containsExactly("command_linux");
        break;
      case DARWIN:
        assertThat(parser.getResidue()).containsExactly("command_macos");
        break;
      case WINDOWS:
        assertThat(parser.getResidue()).containsExactly("command_windows");
        break;
      case FREEBSD:
        assertThat(parser.getResidue()).containsExactly("command_freebsd");
        break;
      case OPENBSD:
        assertThat(parser.getResidue()).containsExactly("command_openbsd");
        break;
      default:
        assertThat(parser.getResidue()).isEmpty();
    }
  }

  @Test
  public void testExpandConfigOptions_withPlatformSpecificConfigEnabledWhenNothingSpecified()
      throws Exception {
    parser.parse("--enable_platform_specific_config");
    optionHandler.parseRcOptions(eventHandler, ArrayListMultimap.create());
    assertThat(eventHandler.getEvents()).isEmpty();
    assertThat(parser.getResidue()).isEmpty();
  }

  @Test
  public void testExpandConfigOptions_withConfigForUnapplicableCommand() throws Exception {
    parser.parse("--config=other");
    OptionsParsingException e =
        assertThrows(
            OptionsParsingException.class,
            () ->
                optionHandler.expandConfigOptions(
                    eventHandler, structuredArgsFrom2SimpleRcsWithOnlyResidue()));
    assertThat(parser.getResidue()).isEmpty();
    assertThat(optionHandler.getRcfileNotes()).isEmpty();
    assertThat(e).hasMessageThat().contains("Config value 'other' is not defined in any .rc file");
  }

  @Test
  public void testUndefinedConfig() throws Exception {
    parser.parse("--config=invalid");
    OptionsParsingException e =
        assertThrows(
            OptionsParsingException.class,
            () -> optionHandler.expandConfigOptions(eventHandler, ArrayListMultimap.create()));
    assertThat(e)
        .hasMessageThat()
        .contains("Config value 'invalid' is not defined in any .rc file");
  }

  @Test
  public void testParseOptions_argless() {
    optionHandler.parseOptions(ImmutableList.of("c0"), eventHandler);
    assertThat(eventHandler.getEvents()).isEmpty();
    assertThat(parser.getResidue()).isEmpty();
    assertThat(optionHandler.getRcfileNotes()).isEmpty();
  }

  @Test
  public void testParseOptions_residue() {
    optionHandler.parseOptions(ImmutableList.of("c0", "res"), eventHandler);
    assertThat(eventHandler.getEvents()).isEmpty();
    assertThat(parser.getResidue()).contains("res");
    assertThat(optionHandler.getRcfileNotes()).isEmpty();
  }

  @Test
  public void testParseOptions_explicitOption() {
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
  public void testParseOptions_rcOption() {
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
  public void testParseOptions_multipleRcs() {
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
  public void testParseOptions_multipleRcsWithMultipleCommands() {
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
  public void testParseOptions_rcOptionAndExplicit() {
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
  public void testParseOptions_multiCommandRcOptionAndExplicit() {
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
  public void testParseOptions_multipleRcsWithMultipleCommandsPlusExplicitOption() {
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
  public void testParseOptions_explicitConfig() {
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

    // "config" is expanded from --config=conf, which occurs last.
    TestOptions options = parser.getOptions(TestOptions.class);
    assertThat(options).isNotNull();
    assertThat(options.testMultipleString).containsExactly("rc", "explicit", "config").inOrder();
  }

  @Test
  public void testParseOptions_rcSpecifiedConfig() {
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

    // "config" is expanded from --config=conf, which occurs before the explicit mention of "rc".
    TestOptions options = parser.getOptions(TestOptions.class);
    assertThat(options).isNotNull();
    assertThat(options.testMultipleString).containsExactly("config", "rc", "explicit").inOrder();
  }

  @Test
  public void testParseOptions_recursiveConfig() {
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

    // The 2nd config, --config=other, is added by --config=conf after conf adds its own value.
    TestOptions options = parser.getOptions(TestOptions.class);
    assertThat(options).isNotNull();
    assertThat(options.testMultipleString)
        .containsExactly("config1", "othercommon", "other", "rc", "explicit")
        .inOrder();
  }

  @Test
  public void testParseOptions_recursiveConfigWithDifferentTokens() {
    optionHandler.parseOptions(
        ImmutableList.of(
            "c0",
            "--default_override=0:c0=--test_multiple_string=rc",
            "--default_override=0:c0:other=--test_multiple_string=other",
            "--default_override=0:c0:conf=--test_multiple_string=config1",
            "--default_override=0:c0:conf=--config",
            "--default_override=0:c0:conf=other",
            "--rc_source=/somewhere/.blazerc",
            "--config=conf"),
        eventHandler);

    assertThat(eventHandler.getEvents())
        .containsExactly(
            Event.error(
                "In file /somewhere/.blazerc, the definition of config conf expands to another "
                    + "config that either has no value or is not in the form --config=value. For "
                    + "recursive config definitions, please do not provide the value in a "
                    + "separate token, such as in the form '--config value'."));
  }

  @Test
  public void testParseOptions_complexConfigOrder() {
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

  @Test
  public void testParseOptions_repeatSubConfig() {
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

  @Test
  public void testParseOptions_repeatConfig() {
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

  @Test
  public void testParseOptions_configCycleLength1() {
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
    assertThat(eventHandler.getEvents())
        .contains(
            Event.error(
                "Config expansion has a cycle: config value foo expands to itself, see "
                    + "inheritance chain [foo]"));
  }

  @Test
  public void testParseOptions_configCycleLength2() {
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
    assertThat(eventHandler.getEvents())
        .contains(
            Event.error(
                "Config expansion has a cycle: config value foo expands to itself, see "
                    + "inheritance chain [foo, bar]"));
  }

  @Test
  public void testParseOptions_recursiveConfigWasAlreadyPresent() {
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

  @Test
  public void testParseOptions_longChain() {
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
    // Expect only one warning, we don't want multiple warnings for the same chain.
    assertThat(eventHandler.getEvents())
        .containsExactly(
            Event.warn(
                "There is a recursive chain of configs 12 configs long: [alpha, beta, gamma, "
                    + "delta, epsilon, zeta, eta, theta, iota, kappa, lambda, mu]. This seems "
                    + "excessive, and might be hiding errors."));
  }

  @Test
  public void testParseOptions_2LongChains() {
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

  @Test
  public void testWarningFlag() {
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
  public void testWarningFlag_byConfig_notTriggered() {
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
  public void testWarningFlag_byConfig_triggered() {
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
  public void testConfigAfterExplicit() {
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
  public void testExplicitOverridesConfig() {
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
