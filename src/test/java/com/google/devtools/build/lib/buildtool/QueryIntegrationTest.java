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
package com.google.devtools.build.lib.buildtool;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;

import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.query2.proto.proto2api.Build;
import com.google.devtools.build.lib.query2.proto.proto2api.Build.QueryResult;
import com.google.devtools.build.lib.query2.query.output.QueryOptions;
import com.google.devtools.build.lib.runtime.BlazeCommandResult;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.GotOptionsEvent;
import com.google.devtools.build.lib.runtime.commands.QueryCommand;
import com.google.devtools.build.lib.runtime.proto.InvocationPolicyOuterClass.InvocationPolicy;
import com.google.devtools.build.lib.unix.UnixFileSystem;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.common.options.OptionsParsingResult;
import com.google.protobuf.ExtensionRegistry;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.xpath.XPathConstants;
import javax.xml.xpath.XPathExpression;
import javax.xml.xpath.XPathFactory;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;

/**
 * Integration tests for 'blaze query'.
 */
@RunWith(JUnit4.class)
public class QueryIntegrationTest extends BuildIntegrationTestCase {
  private final CustomFileSystem fs = new CustomFileSystem();
  private BlazeCommandResult lastBlazeCommandResult;
  private final List<String> options = new ArrayList<>();

  private static class CustomFileSystem extends UnixFileSystem {
    final Map<Path, FileStatus> stubbedStats = new HashMap<>();

    CustomFileSystem() {
      super(DigestHashFunction.SHA256, "");
    }

    void stubStat(Path path, @Nullable FileStatus stubbedResult) {
      stubbedStats.put(path, stubbedResult);
    }

    @Override
    public FileStatus statIfFound(Path path, boolean followSymlinks) throws IOException {
      if (stubbedStats.containsKey(path)) {
        return stubbedStats.get(path);
      }
      return super.statIfFound(path, followSymlinks);
    }
  }

  @Override
  protected FileSystem createFileSystem() {
    return fs;
  }

  @Before
  public final void setQueryOptions() {
    runtimeWrapper.addOptionsClass(QueryOptions.class);
  }

  // Number large enough that an unordered collection with this many elements will never happen to
  // iterate over them in their "natural" order.
  private static final int NUM_DEPS = 1000;

  private static void assertSameElementsDifferentOrder(List<String> actual, List<String> expected) {
    assertThat(actual).containsExactlyElementsIn(expected);
    int i = 0;
    for (; i < expected.size(); i++) {
      if (!actual.get(i).equals(expected.get(i))) {
        break;
      }
    }
    assertWithMessage("Lists should not have been in same order")
        .that(i < expected.size())
        .isTrue();
  }

  private static List<String> getTargetNames(QueryResult result) {
    List<String> results = new ArrayList<>();
    for (Build.Target target : result.getTargetList()) {
      results.add(target.getRule().getName());
    }
    return results;
  }

  @Test
  public void testProtoUnorderedAndOrdered() throws Exception {
    List<String> expected = new ArrayList<>(NUM_DEPS + 1);
    String targets = "";
    String depString = "";
    for (int i = 0; i < NUM_DEPS; i++) {
      String dep = Integer.toString(i);
      depString += "'" + dep + "', ";
      expected.add("//foo:" + dep);
      targets += "sh_library(name = '" + dep + "')\n";
    }
    expected.add("//foo:a");
    Collections.sort(expected, Collections.reverseOrder());
    write("foo/BUILD", "sh_library(name = 'a', deps = [" + depString + "])", targets);
    QueryResult result = getProtoQueryResult("deps(//foo:a)");
    assertSameElementsDifferentOrder(getTargetNames(result), expected);
    options.add("--order_output=full");
    result = getProtoQueryResult("deps(//foo:a)");
    assertThat(getTargetNames(result)).containsExactlyElementsIn(expected).inOrder();
  }

  /**
   * Test that {min,max}rank work as expected with ordering. Since minrank and maxrank have special
   * handling for cycles in the graph, we put a cycle in to exercise that code.
   */
  private void assertRankUnorderedAndOrdered(boolean minRank) throws Exception {
    List<String> expected = new ArrayList<>(2 * NUM_DEPS + 1);
    // The build file looks like:
    // sh_library(name = 'a', deps = ['cycle1', '1', '2', ..., ]
    // sh_library(name = '1')
    // ...
    // sh_library(name = 'n')
    // sh_library(name = 'cycle0', deps = ['cyclen'])
    // sh_library(name = 'cycle1', deps = ['cycle0'])
    // ...
    // sh_library(name = 'cyclen', deps = ['cycle{n-1}'])
    String targets = "";
    String depString = "";
    for (int i = 0; i < NUM_DEPS; i++) {
      String dep = Integer.toString(i);
      depString += "'" + dep + "', ";
      expected.add("1 //foo:" + dep);
      expected.add("1 //foo:cycle" + dep);
      targets += "sh_library(name = '" + dep + "')\n";
      targets += "sh_library(name = 'cycle" + dep + "', deps = ['cycle";
      if (i > 0) {
        targets += i - 1;
      } else {
        targets += NUM_DEPS - 1;
      }
      targets += "'])\n";
    }
    Collections.sort(expected);
    expected.add(0, "0 //foo:a");
    options.add("--output=" + (minRank ? "minrank" : "maxrank"));
    options.add("--keep_going");
    write("foo/BUILD", "sh_library(name = 'a', deps = ['cycle0', " + depString + "])", targets);
    List<String> result = getStringQueryResult("deps(//foo:a)");
    assertWithMessage(result.toString()).that(result.get(0)).isEqualTo("0 //foo:a");
    assertSameElementsDifferentOrder(result, expected);
    options.add("--order_output=full");
    result = getStringQueryResult("deps(//foo:a)");
    assertWithMessage(result.toString()).that(result.get(0)).isEqualTo("0 //foo:a");
    assertThat(result).containsExactlyElementsIn(expected).inOrder();
  }

  @Test
  public void testMinrankUnorderedAndOrdered() throws Exception {
    assertRankUnorderedAndOrdered(true);
  }

  @Test
  public void testMaxrankUnorderedAndOrdered() throws Exception {
    assertRankUnorderedAndOrdered(false);
  }

  @Test
  public void testLabelOrderedFullAndDeps() throws Exception {
    List<String> expected = new ArrayList<>(NUM_DEPS + 1);
    String targets = "";
    String depString = "";
    for (int i = 0; i < NUM_DEPS; i++) {
      String dep = Integer.toString(i);
      depString += "'" + dep + "', ";
      expected.add("//foo:" + dep);
      targets += "sh_library(name = '" + dep + "')\n";
    }
    expected.add("//foo:a");
    Collections.sort(expected, Collections.reverseOrder());
    write("foo/BUILD", "sh_library(name = 'a', deps = [" + depString + "])", targets);
    List<String> result = getStringQueryResult("deps(//foo:a)");
    assertThat(result).containsExactlyElementsIn(expected).inOrder();
    options.add("--order_output=deps");
    result = getStringQueryResult("deps(//foo:a)");
    assertSameElementsDifferentOrder(result, expected);
  }

  @Test
  public void testInputFileElementContainsPackageGroups() throws Exception {
    write("fruit/BUILD",
        "package_group(name='coconut', packages=['//fruit/walnut'])",
        "exports_files(['chestnut'], visibility=[':coconut'])");

    Document result = getXmlQueryResult("//fruit:chestnut");
    Element resultNode = getResultNode(result, "//fruit:chestnut");

    assertThat(
            Iterables.getOnlyElement(
                xpathSelect(resultNode, "package-group[@name='//fruit:coconut']")))
        .isNotNull();
  }

  @Test
  public void testNonStrictTests() throws Exception {
    write("donut/BUILD",
        "sh_binary(name = 'thief', srcs = ['thief.sh'])",
        "cc_test(name = 'shop', srcs = ['shop.cc'])",
        "test_suite(name = 'cop', tests = [':thief', ':shop'])");

    // This should not throw an exception, and return 0 targets.
    QueryResult result = getProtoQueryResult("tests(//donut:cop)");
    assertThat(result.getTargetCount()).isEqualTo(1);
    assertThat(result.getTarget(0).getRule().getName()).isEqualTo("//donut:shop");
  }

  @Test
  public void testStrictTests() throws Exception {
    options.add("--strict_test_suite=true");
    write("donut/BUILD",
        "sh_binary(name = 'thief', srcs = ['thief.sh'])",
        "test_suite(name = 'cop', tests = [':thief'])");

    getProtoQueryResult("tests(//donut:cop)");
    assertThat(lastBlazeCommandResult.getExitCode()).isEqualTo(ExitCode.ANALYSIS_FAILURE);
    assertThat(lastBlazeCommandResult.getFailureDetail().getMessage())
        .contains(
            "The label '//donut:thief' in the test_suite "
                + "'//donut:cop' does not refer to a test");
  }

  private void createBadBarBuild() throws IOException {
    Path barBuildFile = write("bar/BUILD", "sh_library(name = 'bar/baz')");
    FileStatus inconsistentFileStatus =
        new FileStatus() {
          @Override
          public boolean isFile() {
            return false;
          }

          @Override
          public boolean isSpecialFile() {
            return false;
          }

          @Override
          public boolean isDirectory() {
            return false;
          }

          @Override
          public boolean isSymbolicLink() {
            return false;
          }

          @Override
          public long getSize() {
            return 0;
          }

          @Override
          public long getLastModifiedTime() {
            return 0;
          }

          @Override
          public long getLastChangeTime() {
            return 0;
          }

          @Override
          public long getNodeId() {
            return 0;
          }
        };
    fs.stubStat(barBuildFile, inconsistentFileStatus);
  }

  // Regression test for b/14248208.
  private void runInconsistentFileSystem(boolean keepGoing) throws Exception {
    createBadBarBuild();
    if (keepGoing) {
      options.add("--keep_going");
    }
    getQueryResult("deps(//bar:baz)");
    assertThat(lastBlazeCommandResult.getExitCode())
        .isEqualTo(keepGoing ? ExitCode.PARTIAL_ANALYSIS_FAILURE : ExitCode.ANALYSIS_FAILURE);
    events.assertContainsError("Inconsistent filesystem operations");
    assertThat(events.errors()).hasSize(1);
  }

  @Test
  public void inconsistentFileSystemKeepGoing() throws Exception {
    runInconsistentFileSystem(/*keepGoing=*/ true);
  }

  @Test
  public void inconsistentFileSystemNoKeepGoing() throws Exception {
    runInconsistentFileSystem(/*keepGoing=*/ false);
  }

  private void runDepInconsistentFileSystem(boolean keepGoing) throws Exception {
    write("foo/BUILD", "sh_library(name = 'foo', deps = ['//bar:baz'])");
    createBadBarBuild();
    if (keepGoing) {
      options.add("--keep_going");
    }
    getQueryResult("deps(//foo:foo)");
    assertThat(lastBlazeCommandResult.getExitCode())
        .isEqualTo(keepGoing ? ExitCode.PARTIAL_ANALYSIS_FAILURE : ExitCode.ANALYSIS_FAILURE);
    events.assertContainsError("Inconsistent filesystem operations");
    events.assertContainsError("and referenced by '//foo:foo'");
    events.assertContainsError("Evaluation of query \"deps(//foo:foo)\" failed: errors were ");
    // TODO(janakr): We emit duplicate events: in the ErrorPrintingTargetEdgeErrorObserver and in
    //  TransitiveTargetFunction. Should be able to remove one of them, most likely
    //  TransitiveTargetFunction.
    assertThat(events.errors()).hasSize(3);
  }

  @Test
  public void depInconsistentFileSystemKeepGoing() throws Exception {
    runDepInconsistentFileSystem(/*keepGoing=*/ true);
  }

  @Test
  public void depInconsistentFileSystemNoKeepGoing() throws Exception {
    runDepInconsistentFileSystem(/*keepGoing=*/ false);
  }

  private byte[] getQueryResult(String queryString) throws Exception {
    runtimeWrapper.resetOptions();
    runtimeWrapper.addOptions(options);
    runtimeWrapper.addOptions(queryString);
    CommandEnvironment env = runtimeWrapper.newCommand(QueryCommand.class);
    OptionsParsingResult options = env.getOptions();
    for (BlazeModule module : getRuntime().getBlazeModules()) {
      module.beforeCommand(env);
    }

    env.getEventBus()
        .post(
            new GotOptionsEvent(
                getRuntime().getStartupOptionsProvider(),
                options,
                InvocationPolicy.getDefaultInstance()));

    for (BlazeModule module : getRuntime().getBlazeModules()) {
      env.getSkyframeExecutor().injectExtraPrecomputedValues(module.getPrecomputedValues());
    }

    // In this test we are allowed to omit the beforeCommand; so force setting of a command
    // id in the CommandEnvironment, as we will need it in a moment even though we deviate from
    // normal calling order.
    try {
      env.getCommandId();
    } catch (IllegalArgumentException e) {
      // Ignored, as we know the test deviates from normal calling order.
    }

    ByteArrayOutputStream stdout = new ByteArrayOutputStream();
    env.getReporter()
        .addHandler(
            event -> {
              if (event.getKind().equals(EventKind.STDOUT)) {
                try {
                  stdout.write(event.getMessageBytes());
                } catch (IOException e) {
                  throw new IllegalStateException(e);
                }
              }
            });
    lastBlazeCommandResult = new QueryCommand().exec(env, options);
    return stdout.toByteArray();
  }

  private Document getXmlQueryResult(String queryString) throws Exception {
    options.add("--output=xml");
    byte[] queryResult = getQueryResult(queryString);
    return DocumentBuilderFactory.newInstance().newDocumentBuilder()
        .parse(new ByteArrayInputStream(queryResult));
  }

  private static List<Node> xpathSelect(Node doc, String expression) throws Exception {
    XPathExpression expr = XPathFactory.newInstance().newXPath().compile(expression);
    NodeList result = (NodeList) expr.evaluate(doc, XPathConstants.NODESET);
    List<Node> list = new ArrayList<>();
    for (int i = 0; i < result.getLength(); i++) {
      list.add(result.item(i));
    }
    return list;
  }

  private List<String> getStringQueryResult(String queryString) throws Exception {
    return Arrays.asList(
        new String(getQueryResult(queryString), Charset.defaultCharset()).split("\n"));
  }

  private QueryResult getProtoQueryResult(String queryString) throws Exception {
    options.add("--output=proto");
    return QueryResult.parseFrom(getQueryResult(queryString), ExtensionRegistry.getEmptyRegistry());
  }

  Element getResultNode(Document xml, String ruleName) throws Exception {
    return (Element) Iterables.getOnlyElement(xpathSelect(xml,
        String.format("/query/*[@name='%s']", ruleName)));
  }
}
