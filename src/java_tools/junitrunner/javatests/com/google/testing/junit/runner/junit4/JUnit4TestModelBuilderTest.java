// Copyright 2010 The Bazel Authors. All Rights Reserved.
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

package com.google.testing.junit.runner.junit4;

import static com.google.common.truth.Truth.assertThat;
import static com.google.testing.junit.runner.sharding.ShardingFilters.DEFAULT_SHARDING_STRATEGY;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.google.common.collect.Iterables;
import com.google.testing.junit.runner.model.AntXmlResultWriter;
import com.google.testing.junit.runner.model.TestNode;
import com.google.testing.junit.runner.model.TestSuiteModel;
import com.google.testing.junit.runner.model.XmlResultWriter;
import com.google.testing.junit.runner.sharding.ShardingEnvironment;
import com.google.testing.junit.runner.sharding.ShardingFilters;
import com.google.testing.junit.runner.sharding.testing.StubShardingEnvironment;
import com.google.testing.junit.runner.util.FakeTestClock;
import com.google.testing.junit.runner.util.TestClock;
import java.util.List;
import org.junit.Ignore;
import org.junit.Test;
import org.junit.runner.Description;
import org.junit.runner.Request;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.junit.runners.Suite;
import org.junit.runners.Suite.SuiteClasses;

/**
 * Tests for {@code JUnit4TestModelBuilder}
 */
@RunWith(JUnit4.class)
public class JUnit4TestModelBuilderTest {
  private final TestClock fakeTestClock = new FakeTestClock();
  private final ShardingEnvironment stubShardingEnvironment = new StubShardingEnvironment();
  private final XmlResultWriter xmlResultWriter = new AntXmlResultWriter();

  private JUnit4TestModelBuilder builder(Request request, String suiteName,
      ShardingEnvironment shardingEnvironment, ShardingFilters shardingFilters,
      XmlResultWriter xmlResultWriter) {
    return new JUnit4TestModelBuilder(
        request,
        suiteName,
        new TestSuiteModel.Builder(
            fakeTestClock, shardingFilters, shardingEnvironment, xmlResultWriter));
  }

  @Test
  public void testTouchesShardFileWhenShardingEnabled() {
    Class<?> testClass = SampleTestCaseWithTwoTests.class;
    Request request = Request.classWithoutSuiteMethod(testClass);
    ShardingEnvironment mockShardingEnvironment = mock(ShardingEnvironment.class);
    ShardingFilters shardingFilters = new ShardingFilters(
        mockShardingEnvironment, DEFAULT_SHARDING_STRATEGY);
    JUnit4TestModelBuilder modelBuilder = builder(
        request, testClass.getCanonicalName(), mockShardingEnvironment, shardingFilters,
        xmlResultWriter);

    when(mockShardingEnvironment.isShardingEnabled()).thenReturn(true);
    when(mockShardingEnvironment.getTotalShards()).thenReturn(2);
    modelBuilder.get();

    verify(mockShardingEnvironment).touchShardFile();
  }

  @Test
  public void testDoesNotTouchShardFileWhenShardingDisabled() {
    Class<?> testClass = SampleTestCaseWithTwoTests.class;
    Request request = Request.classWithoutSuiteMethod(testClass);
    ShardingEnvironment mockShardingEnvironment = mock(ShardingEnvironment.class);
    ShardingFilters shardingFilters = new ShardingFilters(
        mockShardingEnvironment, DEFAULT_SHARDING_STRATEGY);
    JUnit4TestModelBuilder modelBuilder = builder(
        request, testClass.getCanonicalName(), mockShardingEnvironment, shardingFilters,
        xmlResultWriter);

    when(mockShardingEnvironment.isShardingEnabled()).thenReturn(false);
    modelBuilder.get();

    verify(mockShardingEnvironment, never()).touchShardFile();
  }

  @Test
  public void testCreateModel_topLevelIgnore() throws Exception {
    Class<?> testClass = SampleTestCaseWithTopLevelIgnore.class;
    Request request = Request.classWithoutSuiteMethod(testClass);
    String testClassName = testClass.getCanonicalName();
    JUnit4TestModelBuilder modelBuilder =
        builder(request, testClassName, stubShardingEnvironment, null, xmlResultWriter);

    TestSuiteModel testSuiteModel = modelBuilder.get();
    assertThat(testSuiteModel.getNumTestCases()).isEqualTo(0);
  }

  @Test
  public void testCreateModel_singleTestClass() throws Exception {
    Class<?> testClass = SampleTestCaseWithTwoTests.class;
    Request request = Request.classWithoutSuiteMethod(testClass);
    String testClassName = testClass.getCanonicalName();
    JUnit4TestModelBuilder modelBuilder = builder(
        request, testClassName, stubShardingEnvironment, null, xmlResultWriter);

    Description suite = request.getRunner().getDescription();
    Description testOne = suite.getChildren().get(0);
    Description testTwo = suite.getChildren().get(1);

    TestSuiteModel model = modelBuilder.get();
    TestNode suiteNode = Iterables.getOnlyElement(model.getTopLevelTestSuites());
    assertThat(suiteNode.getDescription()).isEqualTo(suite);
    List<TestNode> testCases = suiteNode.getChildren();
    assertThat(testCases).hasSize(2);
    TestNode testOneNode = testCases.get(0);
    TestNode testTwoNode = testCases.get(1);
    assertThat(testOneNode.getDescription()).isEqualTo(testOne);
    assertThat(testTwoNode.getDescription()).isEqualTo(testTwo);
    assertThat(testOneNode.getChildren()).isEmpty();
    assertThat(testTwoNode.getChildren()).isEmpty();
    assertThat(model.getNumTestCases()).isEqualTo(2);
  }

  @Test
  public void testCreateModel_simpleSuite() throws Exception {
    Class<?> suiteClass = SampleSuite.class;
    Request request = Request.classWithoutSuiteMethod(suiteClass);
    String suiteClassName = suiteClass.getCanonicalName();
    JUnit4TestModelBuilder modelBuilder = builder(
        request, suiteClassName, stubShardingEnvironment, null, xmlResultWriter);

    Description topSuite = request.getRunner().getDescription();
    Description innerSuite = topSuite.getChildren().get(0);
    Description testOne = innerSuite.getChildren().get(0);

    TestSuiteModel model = modelBuilder.get();
    TestNode topSuiteNode = Iterables.getOnlyElement(model.getTopLevelTestSuites());
    assertThat(topSuiteNode.getDescription()).isEqualTo(topSuite);
    TestNode innerSuiteNode = Iterables.getOnlyElement(topSuiteNode.getChildren());
    assertThat(innerSuiteNode.getDescription()).isEqualTo(innerSuite);
    TestNode testOneNode = Iterables.getOnlyElement(innerSuiteNode.getChildren());
    assertThat(testOneNode.getDescription()).isEqualTo(testOne);
    assertThat(testOneNode.getChildren()).isEmpty();
    assertThat(model.getNumTestCases()).isEqualTo(1);
  }

  /** Sample test case with two tests. */
  @RunWith(JUnit4.class)
  public static class SampleTestCaseWithTwoTests {
    @Test
    public void testOne() {
    }

    @Test
    public void testTwo() {
    }
  }

  /** Sample test case with top level @Ignore */
  @Ignore
  @RunWith(JUnit4.class)
  public static class SampleTestCaseWithTopLevelIgnore {
    @Test
    public void testOne() {}

    @Test
    public void testTwo() {}
  }

  /** Sample test case with one test. */
  @RunWith(JUnit4.class)
  public static class SampleTestCaseWithOneTest {
    @Test
    public void testOne() {
    }
  }

  /** Sample suite with one test. */
  @RunWith(Suite.class)
  @SuiteClasses(SampleTestCaseWithOneTest.class)
  public static class SampleSuite {
  }
}
