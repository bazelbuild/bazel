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
    optionHandler =
        BlazeOptionHandler.getHandler(
            runtime,
            runtime.getWorkspace(),
            new C0Command(),
            C0Command.class.getAnnotation(Command.class),
            parser,
            InvocationPolicy.getDefaultInstance());
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
        .containsAllOf(
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
  public void testExpandConfigOptions_withConfigForUnapplicableCommand() throws Exception {
    parser.parse("--config=other");
    optionHandler.expandConfigOptions(eventHandler, structuredArgsFrom2SimpleRcsWithOnlyResidue());
    assertThat(parser.getResidue()).isEmpty();
    assertThat(optionHandler.getRcfileNotes()).isEmpty();
    assertThat(eventHandler.getEvents())
        .contains(Event.warn("Config values are not defined in any .rc file: other"));
  }

  @Test
  public void testAllowUndefinedConfig() throws Exception {
    parser.parse("--config=invalid", "--allow_undefined_configs");
    optionHandler.expandConfigOptions(eventHandler, ArrayListMultimap.create());
    assertThat(parser.getResidue()).isEmpty();
    assertThat(optionHandler.getRcfileNotes()).isEmpty();
    assertThat(eventHandler.getEvents())
        .contains(Event.warn("Config values are not defined in any .rc file: invalid"));
  }

  @Test
  public void testNoAllowUndefinedConfig() {
    try {
      parser.parse("--config=invalid", "--noallow_undefined_configs");
      optionHandler.expandConfigOptions(eventHandler, ArrayListMultimap.create());
      fail();
    } catch (OptionsParsingException e) {
      assertThat(e)
          .hasMessageThat()
          .contains("Config values are not defined in any .rc file: invalid");
    }
  }

  @Test
  public void testParseOptions_argless() {
    optionHandler.parseOptions(ImmutableList.of("c0"), eventHandler);
    assertThat(parser.getResidue()).isEmpty();
    assertThat(optionHandler.getRcfileNotes()).isEmpty();
    assertThat(eventHandler.isEmpty()).isTrue();
  }

  @Test
  public void testParseOptions_residue() {
    optionHandler.parseOptions(ImmutableList.of("c0", "res"), eventHandler);
    assertThat(parser.getResidue()).contains("res");
    assertThat(optionHandler.getRcfileNotes()).isEmpty();
    assertThat(eventHandler.isEmpty()).isTrue();
  }

  @Test
  public void testParseOptions_explicitOption() {
    optionHandler.parseOptions(
        ImmutableList.of("c0", "--test_multiple_string=explicit"), eventHandler);
    assertThat(parser.getResidue()).isEmpty();
    assertThat(optionHandler.getRcfileNotes()).isEmpty();
    assertThat(eventHandler.isEmpty()).isTrue();
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
    assertThat(parser.getResidue()).isEmpty();
    // Check that multiple options in the same rc chunk are collapsed into 1 announce_rc entry.
    assertThat(optionHandler.getRcfileNotes())
        .containsExactly(
            "Reading rc options for 'c0' from /somewhere/.blazerc:\n"
                + "  'c0' options: --test_multiple_string=rc_a --test_multiple_string=rc_b");
    assertThat(eventHandler.getEvents()).isEmpty();
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
    assertThat(parser.getResidue()).isEmpty();
    assertThat(optionHandler.getRcfileNotes())
        .containsExactly(
            "Reading rc options for 'c0' from /somewhere/.blazerc:\n"
                + "  'c0' options: --test_multiple_string=rc1_a",
            "Reading rc options for 'c0' from /some/other/.blazerc:\n"
                + "  'c0' options: --test_multiple_string=rc2",
            "Reading rc options for 'c0' from /somewhere/.blazerc:\n"
                + "  'c0' options: --test_multiple_string=rc1_b");
    assertThat(eventHandler.getEvents()).isEmpty();
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
    assertThat(eventHandler.getEvents()).isEmpty();
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
    assertThat(parser.getResidue()).isEmpty();
    assertThat(optionHandler.getRcfileNotes())
        .containsExactly(
            "Reading rc options for 'c0' from /somewhere/.blazerc:\n"
                + "  'c0' options: --test_multiple_string=rc");
    assertThat(eventHandler.getEvents()).isEmpty();
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
    assertThat(parser.getResidue()).isEmpty();
    assertThat(optionHandler.getRcfileNotes())
        .containsExactly(
            "Reading rc options for 'c0' from /somewhere/.blazerc:\n"
                + "  Inherited 'common' options: --test_multiple_string=rc_common",
            "Reading rc options for 'c0' from /somewhere/.blazerc:\n"
                + "  'c0' options: --test_multiple_string=rc_c0_1 --test_multiple_string=rc_c0_2");
    assertThat(eventHandler.getEvents()).isEmpty();
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
    assertThat(eventHandler.getEvents()).isEmpty();
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
    assertThat(parser.getResidue()).isEmpty();
    assertThat(optionHandler.getRcfileNotes())
        .containsExactly(
            "Reading rc options for 'c0' from /somewhere/.blazerc:\n"
                + "  'c0' options: --test_multiple_string=rc",
            "Found applicable config definition c0:conf in file /somewhere/.blazerc: "
                + "--test_multiple_string=config");
    assertThat(eventHandler.getEvents()).isEmpty();
    TestOptions options = parser.getOptions(TestOptions.class);
    assertThat(options).isNotNull();
    // "config" is lower priority (occurs earlier in the list) than "explicit" in the fix-point
    // expansion, despite --config=conf occurring later.
    assertThat(options.testMultipleString).containsExactly("rc", "config", "explicit").inOrder();
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
    assertThat(parser.getResidue()).isEmpty();
    assertThat(optionHandler.getRcfileNotes())
        .containsExactly(
            "Reading rc options for 'c0' from /somewhere/.blazerc:\n"
                + "  'c0' options: --config=conf --test_multiple_string=rc",
            "Found applicable config definition c0:conf in file /somewhere/.blazerc: "
                + "--test_multiple_string=config");
    assertThat(eventHandler.getEvents()).isEmpty();
    TestOptions options = parser.getOptions(TestOptions.class);
    assertThat(options).isNotNull();
    // "config" is higher priority (occurs later in the list) than "rc" in the fix-point
    // expansion, despite --config=conf occurring before the explicit mention of "rc".
    assertThat(options.testMultipleString).containsExactly("rc", "config", "explicit").inOrder();
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
    assertThat(eventHandler.getEvents()).isEmpty();
    TestOptions options = parser.getOptions(TestOptions.class);
    assertThat(options).isNotNull();
    // The 2nd config, --config=other, is expanded after the config that added it.
    assertThat(options.testMultipleString)
        .containsExactly("rc", "config1", "othercommon", "other", "explicit")
        .inOrder();
  }

  @Test
  public void testParseOptions_recursiveConfigWasAlreadyPresent() {
    optionHandler.parseOptions(
        ImmutableList.of(
            "c0",
            "--default_override=0:c0=--config=conf",
            "--default_override=0:c0=--config=other",
            "--default_override=0:c0=--test_multiple_string=rc",
            "--default_override=0:c0:other=--test_multiple_string=other",
            "--default_override=0:c0:conf=--test_multiple_string=config1",
            "--default_override=0:c0:conf=--config=other",
            "--default_override=0:common:other=--test_multiple_string=othercommon",
            "--rc_source=/somewhere/.blazerc",
            "--test_multiple_string=explicit"),
        eventHandler);
    assertThat(parser.getResidue()).isEmpty();
    assertThat(optionHandler.getRcfileNotes())
        .containsExactly(
            "Reading rc options for 'c0' from /somewhere/.blazerc:\n"
                + "  'c0' options: --config=conf --config=other --test_multiple_string=rc",
            "Found applicable config definition common:other in file /somewhere/.blazerc: "
                + "--test_multiple_string=othercommon",
            "Found applicable config definition c0:conf in file /somewhere/.blazerc: "
                + "--test_multiple_string=config1 --config=other",
            "Found applicable config definition c0:other in file /somewhere/.blazerc: "
                + "--test_multiple_string=other");
    assertThat(eventHandler.getEvents()).isEmpty();
    TestOptions options = parser.getOptions(TestOptions.class);
    assertThat(options).isNotNull();
    // The 2nd config, --config=other, is expanded at the same time as --config=conf, since they are
    // both initially present. The "common" definition is therefore first. other is not reexpanded
    // when it is added by --config=conf, since it was already included.
    assertThat(options.testMultipleString)
        .containsExactly("rc", "othercommon", "config1", "other", "explicit")
        .inOrder();
  }
}
