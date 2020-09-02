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
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.query2.common.AbstractBlazeQueryEnvironment;
import com.google.devtools.build.lib.query2.common.UniverseScope;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.Setting;
import com.google.devtools.build.lib.query2.engine.QueryEvalResult;
import com.google.devtools.build.lib.query2.engine.QueryException;
import com.google.devtools.build.lib.query2.engine.QueryExpression;
import com.google.devtools.build.lib.query2.engine.QueryUtil;
import com.google.devtools.build.lib.query2.engine.QueryUtil.AggregateAllOutputFormatterCallback;
import com.google.devtools.build.lib.query2.proto.proto2api.Build;
import com.google.devtools.build.lib.query2.proto.proto2api.Build.QueryResult;
import com.google.devtools.build.lib.query2.query.output.OutputFormatter;
import com.google.devtools.build.lib.query2.query.output.OutputFormatters;
import com.google.devtools.build.lib.query2.query.output.QueryOptions;
import com.google.devtools.build.lib.query2.query.output.QueryOptions.OrderOutput;
import com.google.devtools.build.lib.query2.query.output.QueryOutputUtils;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.GotOptionsEvent;
import com.google.devtools.build.lib.runtime.KeepGoingOption;
import com.google.devtools.build.lib.runtime.commands.QueryCommand;
import com.google.devtools.build.lib.runtime.proto.InvocationPolicyOuterClass.InvocationPolicy;
import com.google.devtools.common.options.Options;
import com.google.devtools.common.options.OptionsParser;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Set;
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
  private QueryOptions queryOptions;
  private boolean keepGoing;

  @Before
  public final void setQueryOptions() throws Exception  {
    queryOptions = Options.getDefaults(QueryOptions.class);
    keepGoing = Options.getDefaults(KeepGoingOption.class).keepGoing;
    queryOptions.universeScope = ImmutableList.of("//...:*");
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
    queryOptions.orderOutput = OrderOutput.FULL;
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
    queryOptions.outputFormat = minRank ? "minrank" : "maxrank";
    keepGoing = true;
    write("foo/BUILD", "sh_library(name = 'a', deps = ['cycle0', " + depString + "])", targets);
    List<String> result = getStringQueryResult("deps(//foo:a)");
    assertWithMessage(result.toString()).that(result.get(0)).isEqualTo("0 //foo:a");
    assertSameElementsDifferentOrder(result, expected);
    queryOptions.orderOutput = OrderOutput.FULL;
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
    queryOptions.orderOutput = OrderOutput.DEPS;
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
    queryOptions.strictTestSuite = true;
    write("donut/BUILD",
        "sh_binary(name = 'thief', srcs = ['thief.sh'])",
        "test_suite(name = 'cop', tests = [':thief'])");

    QueryException e =
        assertThrows(QueryException.class, () -> getProtoQueryResult("tests(//donut:cop)"));
    assertThat(e)
        .hasMessageThat()
        .contains(
            "The label '//donut:thief' in the test_suite "
                + "'//donut:cop' does not refer to a test");
  }

  private byte[] getQueryResult(String queryString) throws Exception {
    // TODO(hanwen): this should probably use BlazeRuntimeWrapper so
    // we don't have to duplicate option/module handling.
    OptionsParser optionsParser = runtimeWrapper.createOptionsParser();
    Command command = QueryCommand.class.getAnnotation(Command.class);
    CommandEnvironment env =
        getBlazeWorkspace().initCommand(command, optionsParser, new ArrayList<>(), 0L, 0L);
    for (BlazeModule module : getRuntime().getBlazeModules()) {
      module.beforeCommand(env);
    }

    env.getEventBus()
        .post(
            new GotOptionsEvent(
                getRuntime().getStartupOptionsProvider(),
                optionsParser,
                InvocationPolicy.getDefaultInstance()));

    for (BlazeModule module : getRuntime().getBlazeModules()) {
      env.getSkyframeExecutor().injectExtraPrecomputedValues(module.getPrecomputedValues());
    }

    // In this test we are allowed to ommit the beforeCommand; so force setting of a command
    // id in the CommandEnvironment, as we will need it in a moment even though we deviate from
    // normal calling order.
    try {
      env.getCommandId();
    } catch (IllegalArgumentException e) {
      // Ignored, as we know the test deviates from normal calling order.
    }

    env.syncPackageLoading(optionsParser);

    OutputFormatter formatter =
        OutputFormatters.getFormatter(
            OutputFormatters.getDefaultFormatters(), queryOptions.outputFormat);
    // TODO(ulfjack): This should run the code in QueryCommand instead.
    Set<Setting> settings = queryOptions.toSettings();
    AbstractBlazeQueryEnvironment<Target> queryEnv =
        QueryCommand.newQueryEnvironment(
            env,
            keepGoing,
            !QueryOutputUtils.shouldStreamResults(queryOptions, formatter),
            UniverseScope.EMPTY,
            /*loadingPhaseThreads=*/ 1,
            settings,
            /*useGraphlessQuery=*/ false);
    AggregateAllOutputFormatterCallback<Target, ?> callback =
        QueryUtil.newOrderedAggregateAllOutputFormatterCallback(queryEnv);
    QueryEvalResult result =
        queryEnv.evaluateQuery(QueryExpression.parse(queryString, queryEnv), callback);

    ByteArrayOutputStream outputStream = new ByteArrayOutputStream();

    Reporter reporter = new Reporter(new EventBus(), events.collector());
    QueryOutputUtils.output(
        queryOptions,
        result,
        callback.getResult(),
        formatter,
        outputStream,
        queryOptions.aspectDeps.createResolver(env.getPackageManager(), reporter),
        reporter);
    return outputStream.toByteArray();
  }

  private Document getXmlQueryResult(String queryString) throws Exception {
    queryOptions.outputFormat = "xml";
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
    queryOptions.outputFormat = "proto";
    return QueryResult.parseFrom(getQueryResult(queryString));
  }

  Element getResultNode(Document xml, String ruleName) throws Exception {
    return (Element) Iterables.getOnlyElement(xpathSelect(xml,
        String.format("/query/*[@name='%s']", ruleName)));
  }
}
