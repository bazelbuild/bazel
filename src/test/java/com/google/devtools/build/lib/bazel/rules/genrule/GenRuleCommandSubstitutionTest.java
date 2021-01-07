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

package com.google.devtools.build.lib.bazel.rules.genrule;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;

import com.google.common.base.Joiner;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * A unit test of the various kinds of label and "Make"-variable substitutions that are applied to
 * the genrule "cmd" attribute.
 *
 * <p>Some of these tests are similar to tests in LabelExpanderTest and MakeVariableExpanderTest,
 * but this test case exercises the composition of these various transformations.
 */
@RunWith(JUnit4.class)
public final class GenRuleCommandSubstitutionTest extends BuildViewTestCase {

  private static final Pattern SETUP_COMMAND_PATTERN =
      Pattern.compile(".*/genrule-setup.sh;\\s+(?<command>.*)");

  private String getGenruleCommand(String genrule) throws Exception {
    return ((SpawnAction)
            getGeneratingAction(getFilesToBuild(getConfiguredTarget(genrule)).toList().get(0)))
        .getArguments()
        .get(2);
  }

  private void assertExpansionEquals(String expected, String genrule) throws Exception {
    String command = getGenruleCommand(genrule);
    assertCommandEquals(expected, command);
  }

  private static void assertCommandEquals(String expected, String command) {
    // Ensure the command after the genrule setup is correct.
    Matcher m = SETUP_COMMAND_PATTERN.matcher(command);
    if (m.matches()) {
      command = m.group("command");
    }

    assertWithMessage("Expected command to be \"" + expected + "\", but found \"" + command + "\"")
        .that(command)
        .isEqualTo(expected);
  }

  private void assertExpansionFails(String expectedErrorSuffix, String genrule) throws Exception {
    reporter.removeHandler(failFastHandler); // we expect errors
    eventCollector.clear();
    getConfiguredTarget(genrule);
    assertContainsEvent(expectedErrorSuffix);
  }

  // Creates a BUILD file defining a genrule called "//test" with no srcs or
  // deps, one output and the specified command.
  private void genrule(String command) throws Exception {
    scratch.overwriteFile(
        "test/BUILD",
        // This is a horrible workaround for b/147306893:
        // somehow, duplicate events (same location, same message)
        // are being suppressed, so we must vary the location of the
        // genrule by inserting a unique number of newlines.
        new String(new char[seq++]).replace('\0', '\n'),
        "genrule(name = 'test',",
        "        outs = ['out'],",
        "        cmd = '" + command + "')");

    // Since we're probably re-defining "//test":
    invalidatePackages();
  }

  private int seq = 0;

  @Test
  public void testLocationSyntaxErrors() throws Exception {
    genrule("$(location )");
    assertExpansionFails(
        "invalid label in $(location) expression: empty package-relative label", "//test");

    genrule("$(location foo bar");
    assertExpansionFails("unterminated variable reference", "//test");

    genrule("$(location");
    assertExpansionFails("unterminated variable reference", "//test");

    genrule("$(locationz");
    assertExpansionFails("unterminated variable reference", "//test");

    genrule("$(locationz)");
    assertExpansionFails("$(locationz) not defined", "//test");

    genrule("$(locationz )");
    assertExpansionFails("$(locationz) not defined", "//test");

    genrule("$(locationz foo )");
    assertExpansionFails("$(locationz) not defined", "//test");
  }

  @Test
  public void testLocationOfLabelThatIsNotAPrerequsite() throws Exception {
    scratch.file(
        "test/BUILD",
        "exports_files(['exists'])",
        "genrule(name = 'test1',",
        "        outs = ['test1.out'],",
        "        cmd = '$(location :exists)')",
        "genrule(name = 'test2',",
        "        outs = ['test2.out'],",
        "        cmd = '$(location :doesnt_exist)')");

    // $(location) of a non-prerequisite fails, even if the target exists:

    assertExpansionFails(
        "label '//test:exists' in $(location) expression is "
            + "not a declared prerequisite of this rule",
        "//test:test1");

    assertExpansionFails(
        "label '//test:doesnt_exist' in $(location) expression is "
            + "not a declared prerequisite of this rule",
        "//test:test2");
  }

  @Test
  public void testLocationOfMultiFileLabel() throws Exception {
    scratch.file(
        "deuce/BUILD",
        "genrule(name = 'deuce',",
        "        outs = ['out.1', 'out.2'],",
        "        cmd = ':')");
    checkError(
        "test",
        "test1",
        "label '//deuce:deuce' in $(location) expression expands to more than one "
            + "file, please use $(locations //deuce:deuce) instead",
        "genrule(name = 'test1',",
        "        tools = ['//deuce'],",
        "        outs = ['test1.out'],",
        "        cmd = '$(location //deuce)')");
  }

  @Test
  public void testUnknownVariable() throws Exception {
    genrule("$(UNKNOWN)");
    assertExpansionFails("$(UNKNOWN) not defined", "//test");
  }

  @Test
  public void testLocationOfSourceLabel() throws Exception {
    scratch.file(
        "test1/BUILD",
        "genrule(name = 'test1',",
        "        srcs = ['src'],",
        "        outs = ['out'],",
        "        cmd = '$(location //test1:src)')");
    assertExpansionEquals("test1/src", "//test1");

    scratch.file(
        "test2/BUILD",
        "genrule(name = 'test2',",
        "        srcs = ['src'],",
        "        outs = ['out'],",
        "        cmd = '$(location src)')");
    assertExpansionEquals("test2/src", "//test2");

    scratch.file(
        "test3/BUILD",
        "genrule(name = 'test3',",
        "        srcs = ['src'],",
        "        outs = ['out'],",
        "        cmd = '$(location :src)')");
    assertExpansionEquals("test3/src", "//test3");
  }

  @Test
  public void testLocationOfOutputLabel() throws Exception {
    String gendir = targetConfig.getMakeVariableDefault("GENDIR");
    scratch.file(
        "test1/BUILD",
        "genrule(name = 'test1',",
        "        outs = ['out'],",
        "        cmd = '$(location //test1:out)')");
    assertExpansionEquals(gendir + "/test1/out", "//test1");

    scratch.file(
        "test2/BUILD",
        "genrule(name = 'test2',",
        "        outs = ['out'],",
        "        cmd = '$(location out)')");
    assertExpansionEquals(gendir + "/test2/out", "//test2");

    scratch.file(
        "test3/BUILD",
        "genrule(name = 'test3',",
        "        outs = ['out'],",
        "        cmd = '$(location out)')");
    assertExpansionEquals(gendir + "/test3/out", "//test3");
  }

  @Test
  public void testLocationsSyntaxErrors() throws Exception {
    genrule("$(locations )");
    assertExpansionFails(
        "invalid label in $(locations) expression: empty package-relative label", "//test");

    genrule("$(locations foo bar");
    assertExpansionFails("unterminated variable reference", "//test");

    genrule("$(locations");
    assertExpansionFails("unterminated variable reference", "//test");

    genrule("$(locationsz");
    assertExpansionFails("unterminated variable reference", "//test");

    genrule("$(locationsz)");
    assertExpansionFails("$(locationsz) not defined", "//test");

    genrule("$(locationsz )");
    assertExpansionFails("$(locationsz) not defined", "//test");

    genrule("$(locationsz foo )");
    assertExpansionFails("$(locationsz) not defined", "//test");
  }

  @Test
  public void testLocationsOfLabelThatIsNotAPrerequsite() throws Exception {
    scratch.file(
        "test/BUILD",
        "exports_files(['exists'])",
        "genrule(name = 'test1',",
        "        outs = ['test1.out'],",
        "        cmd = '$(locations :exists)')",
        "genrule(name = 'test2',",
        "        outs = ['test2.out'],",
        "        cmd = '$(locations :doesnt_exist)')");

    // $(locations) of a non-prerequisite fails, even if the target exists:

    assertExpansionFails(
        "label '//test:exists' in $(locations) expression is "
            + "not a declared prerequisite of this rule",
        "//test:test1");

    assertExpansionFails(
        "label '//test:doesnt_exist' in $(locations) expression is "
            + "not a declared prerequisite of this rule",
        "//test:test2");
  }

  @Test
  public void testLocationsOfMultiFileLabel() throws Exception {
    String gendir = targetConfig.getMakeVariableDefault("GENDIR");
    scratch.file(
        "test/BUILD",
        "genrule(name = 'x',",
        "        srcs = ['src'],",
        "        outs = ['out1', 'out2'],",
        "        cmd = ':')",
        "genrule(name = 'y',",
        "        srcs = ['x'],",
        "        outs = ['out'],",
        "        cmd = '$(locations x)')");

    assertExpansionEquals(gendir + "/test/out1 " + gendir + "/test/out2", "//test:y");
  }

  @Test
  public void testLocationLocationsAndLabel() throws Exception {
    String gendir = targetConfig.getMakeVariableDefault("GENDIR");
    scratch.file(
        "test/BUILD",
        "genrule(name = 'x',",
        "        srcs = ['src'],",
        "        outs = ['out'],",
        "        cmd = ':')",
        "genrule(name = 'y',",
        "        srcs = ['src'],",
        "        outs = ['out1', 'out2'],",
        "        cmd = ':')",
        "genrule(name = 'r',",
        "        srcs = ['x', 'y', 'z'],",
        "        outs = ['res'],",
        "        cmd = ' _ $(location x) _ $(locations y) _ ')");

    String expected =
        "_ " + gendir + "/test/out _ " + gendir + "/test/out1 " + gendir + "/test/out2 _ ";
    assertExpansionEquals(expected, "//test:r");
  }

  @Test
  public void testLocationsOfSourceLabel() throws Exception {
    scratch.file(
        "test1/BUILD",
        "genrule(name = 'test1',",
        "        srcs = ['src'],",
        "        outs = ['out'],",
        "        cmd = '$(locations //test1:src)')");
    assertExpansionEquals("test1/src", "//test1");

    scratch.file(
        "test2/BUILD",
        "genrule(name = 'test2',",
        "        srcs = ['src'],",
        "        outs = ['out'],",
        "        cmd = '$(locations src)')");
    assertExpansionEquals("test2/src", "//test2");

    scratch.file(
        "test3/BUILD",
        "genrule(name = 'test3',",
        "        srcs = ['src'],",
        "        outs = ['out'],",
        "        cmd = '$(location :src)')");
    assertExpansionEquals("test3/src", "//test3");
  }

  @Test
  public void testLocationsOfOutputLabel() throws Exception {
    String gendir = targetConfig.getMakeVariableDefault("GENDIR");
    scratch.file(
        "test1/BUILD",
        "genrule(name = 'test1',",
        "        outs = ['out'],",
        "        cmd = '$(locations //test1:out)')");
    assertExpansionEquals(gendir + "/test1/out", "//test1");

    scratch.file(
        "test2/BUILD",
        "genrule(name = 'test2',",
        "        outs = ['out'],",
        "        cmd = '$(locations out)')");
    assertExpansionEquals(gendir + "/test2/out", "//test2");

    scratch.file(
        "test3/BUILD",
        "genrule(name = 'test3',",
        "        outs = ['out'],",
        "        cmd = '$(locations out)')");
    assertExpansionEquals(gendir + "/test3/out", "//test3");
  }

  @Test
  public void testOuts() throws Exception {
    String expected = targetConfig.getMakeVariableDefault("GENDIR") + "/test/out";
    scratch.file(
        "test/BUILD",
        "genrule(name = 'test',",
        "        outs = ['out'],",
        "        cmd = '$(OUTS) # $@')");
    assertExpansionEquals(expected + " # " + expected, "//test");
  }

  @Test
  public void testSrcs() throws Exception {
    String expected = "test/src";

    scratch.file(
        "test/BUILD",
        "genrule(name = 'test',",
        "        srcs = ['src'],",
        "        outs = ['out'],",
        "        cmd = '$(SRCS) # $<')");
    assertExpansionEquals(expected + " # " + expected, "//test");
  }

  @Test
  public void testDollarDollar() throws Exception {
    scratch.file(
        "test/BUILD",
        "genrule(name = 'test',",
        "        outs = ['out'],",
        "        cmd = '$$DOLLAR')");
    assertExpansionEquals("$DOLLAR", "//test");
  }

  @Test
  public void testDollarLessThanWithZeroInputs() throws Exception {
    scratch.file(
        "test/BUILD",
        "genrule(name = 'test',",
        "        outs = ['out'],",
        "        cmd  = '$<')");
    assertExpansionFails("variable '$<' : no input file", "//test");
  }

  @Test
  public void testDollarLessThanWithMultipleInputs() throws Exception {
    scratch.file(
        "test/BUILD",
        "genrule(name = 'test',",
        "        srcs = ['src1', 'src2'],",
        "        outs = ['out'],",
        "        cmd  = '$<')");
    assertExpansionFails("variable '$<' : more than one input file", "//test");
  }

  @Test
  public void testDollarAtWithMultipleOutputs() throws Exception {
    scratch.file(
        "test/BUILD",
        "genrule(name = 'test',",
        "        outs = ['out.1', 'out.2'],",
        "        cmd  = '$@')");
    assertExpansionFails("variable '$@' : more than one output file", "//test");
  }

  @Test
  public void testDollarAtWithZeroOutputs() throws Exception {
    scratch.file(
        "test/BUILD",
        "genrule(name = 'test',",
        "        srcs = ['src1', 'src2'],",
        "        outs = [],",
        "        cmd  = '$@')");
    assertExpansionFails("Genrules without outputs don't make sense", "//test");
  }

  @Test
  public void testShellVariables() throws Exception {
    genrule("for file in a b c;do echo $$file;done");
    assertExpansionEquals("for file in a b c;do echo $file;done", "//test");
    assertNoEvents();

    genrule("$${file%:.*8}");
    assertExpansionEquals("${file%:.*8}", "//test");
    assertNoEvents();

    genrule("$$(basename file)");
    assertExpansionEquals("$(basename file)", "//test");
    assertNoEvents();

    genrule("$(basename file)");
    assertExpansionFails("$(basename) not defined", "//test");
    assertContainsEvent("$(basename) not defined");
  }

  @Test
  public void heuristicLabelExpansion_singletonFilegroupInTools_expandsToFile() throws Exception {
    scratch.file(
        "foo/BUILD",
        "filegroup(name = 'fg', srcs = ['fg1.txt'])",
        "genrule(",
        "  name = 'gen',",
        "  outs = ['gen.out'],",
        "  tools = [':fg'],",
        "  heuristic_label_expansion = True,",
        "  cmd = 'cp :fg $@',",
        ")");

    useConfiguration("--experimental_enable_aggregating_middleman");
    assertThat(getGenruleCommand("//foo:gen")).contains("foo/fg1.txt");

    useConfiguration("--noexperimental_enable_aggregating_middleman");
    assertThat(getGenruleCommand("//foo:gen")).contains("foo/fg1.txt");
  }

  @Test
  public void heuristicLabelExpansion_emptyFilegroupInTools_fails() throws Exception {
    scratch.file(
        "foo/BUILD",
        "filegroup(name = 'fg', srcs = [])",
        "genrule(",
        "  name = 'gen',",
        "  outs = ['gen.out'],",
        "  tools = [':fg'],",
        "  heuristic_label_expansion = True,",
        "  cmd = 'cp :fg $@',",
        ")");

    useConfiguration("--experimental_enable_aggregating_middleman");
    assertExpansionFails("expands to 0 files", "//foo:gen");

    useConfiguration("--noexperimental_enable_aggregating_middleman");
    assertExpansionFails("expands to 0 files", "//foo:gen");
  }

  @Test
  public void heuristicLabelExpansion_multiFilegroupInTools_fails() throws Exception {
    scratch.file(
        "foo/BUILD",
        "filegroup(name = 'fg', srcs = ['fg1.txt', 'fg2.txt'])",
        "genrule(",
        "  name = 'gen',",
        "  outs = ['gen.out'],",
        "  tools = [':fg'],",
        "  heuristic_label_expansion = True,",
        "  cmd = 'cp :fg $@',",
        ")");

    useConfiguration("--experimental_enable_aggregating_middleman");
    assertExpansionFails("expands to 2 files", "//foo:gen");

    useConfiguration("--noexperimental_enable_aggregating_middleman");
    assertExpansionFails("expands to 2 files", "//foo:gen");
  }

  @Test
  public void testDollarFileFails() throws Exception {
    checkError(
        "test",
        "test",
        "'$file' syntax is not supported; use '$(file)' ",
        getBuildFileWithCommand("for file in a b c;do echo $file;done"));
  }

  @Test
  public void testDollarFile2Fails() throws Exception {
    checkError(
        "test",
        "test",
        "'${file%:.*8}' syntax is not supported; use '$(file%:.*8)' ",
        getBuildFileWithCommand("${file%:.*8}"));
  }

  private static String getBuildFileWithCommand(String command) {
    return Joiner.on("\n")
        .join(
            "genrule(name = 'test',",
            "        outs = ['out'],",
            "        cmd = '" + command + "')");
  }
}
