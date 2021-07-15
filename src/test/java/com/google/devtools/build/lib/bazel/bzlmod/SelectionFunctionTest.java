// Copyright 2021 The Bazel Authors. All rights reserved.
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
//

package com.google.devtools.build.lib.bazel.bzlmod;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.bazel.bzlmod.BzlmodTestUtil.createModuleKey;
import static org.junit.Assert.fail;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.testutil.FoundationTestCase;
import com.google.devtools.build.skyframe.EvaluationContext;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.InMemoryMemoizingEvaluator;
import com.google.devtools.build.skyframe.MemoizingEvaluator;
import com.google.devtools.build.skyframe.RecordingDifferencer;
import com.google.devtools.build.skyframe.SequencedRecordingDifferencer;
import com.google.devtools.build.skyframe.SequentialBuildDriver;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link SelectionFunction}. */
@RunWith(JUnit4.class)
public class SelectionFunctionTest extends FoundationTestCase {

  private SequentialBuildDriver driver;
  private RecordingDifferencer differencer;
  private EvaluationContext evaluationContext;

  @Before
  public void setup() throws Exception {
    differencer = new SequencedRecordingDifferencer();
    evaluationContext =
        EvaluationContext.newBuilder().setNumThreads(8).setEventHandler(reporter).build();
  }

  private void setUpSkyFunctions(
      String rootModuleName,
      ImmutableMap<ModuleKey, Module> depGraph,
      ImmutableMap<String, ModuleOverride> overrides)
      throws Exception {
    MemoizingEvaluator evaluator =
        new InMemoryMemoizingEvaluator(
            ImmutableMap.<SkyFunctionName, SkyFunction>builder()
                .put(
                    SkyFunctions.MODULE_FILE,
                    new SkyFunction() {
                      @Override
                      public SkyValue compute(SkyKey skyKey, Environment env) {
                        Preconditions.checkArgument(
                            skyKey.equals(ModuleFileValue.keyForRootModule()));
                        return ModuleFileValue.create(
                            Module.builder()
                                .setName(rootModuleName)
                                .setVersion(Version.EMPTY)
                                .build(),
                            overrides);
                      }

                      @Override
                      public String extractTag(SkyKey skyKey) {
                        return null;
                      }
                    })
                .put(
                    SkyFunctions.DISCOVERY,
                    new SkyFunction() {
                      @Override
                      public SkyValue compute(SkyKey skyKey, Environment env) {
                        return DiscoveryValue.create(rootModuleName, depGraph);
                      }

                      @Override
                      public String extractTag(SkyKey skyKey) {
                        return null;
                      }
                    })
                .put(SkyFunctions.SELECTION, new SelectionFunction())
                .build(),
            differencer);
    driver = new SequentialBuildDriver(evaluator);
  }

  @Test
  public void testSimpleDiamond() throws Exception {
    setUpSkyFunctions(
        "A",
        ImmutableMap.<ModuleKey, Module>builder()
            .put(
                createModuleKey("A", ""),
                Module.builder()
                    .setName("A")
                    .setVersion(Version.EMPTY)
                    .addDep("BfromA", createModuleKey("B", "1.0"))
                    .addDep("CfromA", createModuleKey("C", "2.0"))
                    .build())
            .put(
                createModuleKey("B", "1.0"),
                Module.builder()
                    .setName("B")
                    .setVersion(Version.parse("1.0"))
                    .addDep("DfromB", createModuleKey("D", "1.0"))
                    .build())
            .put(
                createModuleKey("C", "2.0"),
                Module.builder()
                    .setName("C")
                    .setVersion(Version.parse("2.0"))
                    .addDep("DfromC", createModuleKey("D", "2.0"))
                    .build())
            .put(
                createModuleKey("D", "1.0"),
                Module.builder()
                    .setName("D")
                    .setVersion(Version.parse("1.0"))
                    .setCompatibilityLevel(1)
                    .build())
            .put(
                createModuleKey("D", "2.0"),
                Module.builder()
                    .setName("D")
                    .setVersion(Version.parse("2.0"))
                    .setCompatibilityLevel(1)
                    .build())
            .build(),
        ImmutableMap.of());

    EvaluationResult<SelectionValue> result =
        driver.evaluate(ImmutableList.of(SelectionValue.KEY), evaluationContext);
    if (result.hasError()) {
      fail(result.getError().toString());
    }
    SelectionValue selectionValue = result.get(SelectionValue.KEY);
    assertThat(selectionValue.getRootModuleName()).isEqualTo("A");
    assertThat(selectionValue.getDepGraph())
        .containsExactly(
            createModuleKey("A", ""),
            Module.builder()
                .setName("A")
                .setVersion(Version.EMPTY)
                .addDep("BfromA", createModuleKey("B", "1.0"))
                .addDep("CfromA", createModuleKey("C", "2.0"))
                .build(),
            createModuleKey("B", "1.0"),
            Module.builder()
                .setName("B")
                .setVersion(Version.parse("1.0"))
                .addDep("DfromB", createModuleKey("D", "2.0"))
                .build(),
            createModuleKey("C", "2.0"),
            Module.builder()
                .setName("C")
                .setVersion(Version.parse("2.0"))
                .addDep("DfromC", createModuleKey("D", "2.0"))
                .build(),
            createModuleKey("D", "2.0"),
            Module.builder()
                .setName("D")
                .setVersion(Version.parse("2.0"))
                .setCompatibilityLevel(1)
                .build());
  }

  @Test
  public void testDiamondWithFurtherRemoval() throws Exception {
    setUpSkyFunctions(
        "A",
        ImmutableMap.<ModuleKey, Module>builder()
            .put(
                createModuleKey("A", ""),
                Module.builder()
                    .setName("A")
                    .setVersion(Version.EMPTY)
                    .addDep("B", createModuleKey("B", "1.0"))
                    .addDep("C", createModuleKey("C", "2.0"))
                    .build())
            .put(
                createModuleKey("B", "1.0"),
                Module.builder()
                    .setName("B")
                    .setVersion(Version.parse("1.0"))
                    .addDep("D", createModuleKey("D", "1.0"))
                    .build())
            .put(
                createModuleKey("C", "2.0"),
                Module.builder()
                    .setName("C")
                    .setVersion(Version.parse("2.0"))
                    .addDep("D", createModuleKey("D", "2.0"))
                    .build())
            .put(
                createModuleKey("D", "1.0"),
                Module.builder()
                    .setName("D")
                    .setVersion(Version.parse("1.0"))
                    .addDep("E", createModuleKey("E", "1.0"))
                    .build())
            .put(
                createModuleKey("D", "2.0"),
                Module.builder().setName("D").setVersion(Version.parse("2.0")).build())
            // Only D@1.0 needs E. When D@1.0 is removed, E should be gone as well (even though
            // E@1.0 is selected for E).
            .put(
                createModuleKey("E", "1.0"),
                Module.builder().setName("E").setVersion(Version.parse("1.0")).build())
            .build(),
        ImmutableMap.of());

    EvaluationResult<SelectionValue> result =
        driver.evaluate(ImmutableList.of(SelectionValue.KEY), evaluationContext);
    if (result.hasError()) {
      fail(result.getError().toString());
    }
    SelectionValue selectionValue = result.get(SelectionValue.KEY);
    assertThat(selectionValue.getRootModuleName()).isEqualTo("A");
    assertThat(selectionValue.getDepGraph())
        .containsExactly(
            createModuleKey("A", ""),
            Module.builder()
                .setName("A")
                .setVersion(Version.EMPTY)
                .addDep("B", createModuleKey("B", "1.0"))
                .addDep("C", createModuleKey("C", "2.0"))
                .build(),
            createModuleKey("B", "1.0"),
            Module.builder()
                .setName("B")
                .setVersion(Version.parse("1.0"))
                .addDep("D", createModuleKey("D", "2.0"))
                .build(),
            createModuleKey("C", "2.0"),
            Module.builder()
                .setName("C")
                .setVersion(Version.parse("2.0"))
                .addDep("D", createModuleKey("D", "2.0"))
                .build(),
            createModuleKey("D", "2.0"),
            Module.builder().setName("D").setVersion(Version.parse("2.0")).build());
  }

  @Test
  public void testCircularDependencyDueToSelection() throws Exception {
    setUpSkyFunctions(
        "A",
        ImmutableMap.<ModuleKey, Module>builder()
            .put(
                createModuleKey("A", ""),
                Module.builder()
                    .setName("A")
                    .setVersion(Version.EMPTY)
                    .addDep("B", createModuleKey("B", "1.0"))
                    .build())
            .put(
                createModuleKey("B", "1.0"),
                Module.builder()
                    .setName("B")
                    .setVersion(Version.parse("1.0"))
                    .addDep("C", createModuleKey("C", "2.0"))
                    .build())
            .put(
                createModuleKey("C", "2.0"),
                Module.builder()
                    .setName("C")
                    .setVersion(Version.parse("2.0"))
                    .addDep("B", createModuleKey("B", "1.0-pre"))
                    .build())
            .put(
                createModuleKey("B", "1.0-pre"),
                Module.builder()
                    .setName("B")
                    .setVersion(Version.parse("1.0-pre"))
                    .addDep("D", createModuleKey("D", "1.0"))
                    .build())
            .put(
                createModuleKey("D", "1.0"),
                Module.builder().setName("D").setVersion(Version.parse("1.0")).build())
            .build(),
        ImmutableMap.of());

    EvaluationResult<SelectionValue> result =
        driver.evaluate(ImmutableList.of(SelectionValue.KEY), evaluationContext);
    if (result.hasError()) {
      fail(result.getError().toString());
    }
    SelectionValue selectionValue = result.get(SelectionValue.KEY);
    assertThat(selectionValue.getRootModuleName()).isEqualTo("A");
    assertThat(selectionValue.getDepGraph())
        .containsExactly(
            createModuleKey("A", ""),
            Module.builder()
                .setName("A")
                .setVersion(Version.EMPTY)
                .addDep("B", createModuleKey("B", "1.0"))
                .build(),
            createModuleKey("B", "1.0"),
            Module.builder()
                .setName("B")
                .setVersion(Version.parse("1.0"))
                .addDep("C", createModuleKey("C", "2.0"))
                .build(),
            createModuleKey("C", "2.0"),
            Module.builder()
                .setName("C")
                .setVersion(Version.parse("2.0"))
                .addDep("B", createModuleKey("B", "1.0"))
                .build());
    // D is completely gone.
  }

  @Test
  public void differentCompatibilityLevelIsRejected() throws Exception {
    setUpSkyFunctions(
        "A",
        ImmutableMap.<ModuleKey, Module>builder()
            .put(
                createModuleKey("A", ""),
                Module.builder()
                    .setName("A")
                    .setVersion(Version.EMPTY)
                    .addDep("BfromA", createModuleKey("B", "1.0"))
                    .addDep("CfromA", createModuleKey("C", "2.0"))
                    .build())
            .put(
                createModuleKey("B", "1.0"),
                Module.builder()
                    .setName("B")
                    .setVersion(Version.parse("1.0"))
                    .addDep("DfromB", createModuleKey("D", "1.0"))
                    .build())
            .put(
                createModuleKey("C", "2.0"),
                Module.builder()
                    .setName("C")
                    .setVersion(Version.parse("2.0"))
                    .addDep("DfromC", createModuleKey("D", "2.0"))
                    .build())
            .put(
                createModuleKey("D", "1.0"),
                Module.builder()
                    .setName("D")
                    .setVersion(Version.parse("1.0"))
                    .setCompatibilityLevel(1)
                    .build())
            .put(
                createModuleKey("D", "2.0"),
                Module.builder()
                    .setName("D")
                    .setVersion(Version.parse("2.0"))
                    .setCompatibilityLevel(2)
                    .build())
            .build(),
        ImmutableMap.of());

    EvaluationResult<SelectionValue> result =
        driver.evaluate(ImmutableList.of(SelectionValue.KEY), evaluationContext);
    assertThat(result.hasError()).isTrue();
    String error = result.getError().toString();
    assertThat(error).contains("B@1.0 depends on D@1.0 with compatibility level 1");
    assertThat(error).contains("C@2.0 depends on D@2.0 with compatibility level 2");
    assertThat(error).contains("which is different");
  }

  @Test
  public void differentCompatibilityLevelIsOkIfUnreferenced() throws Exception {
    // A 1.0 -> B 1.0 -> C 2.0
    //       \-> C 1.0
    //        \-> D 1.0 -> B 1.1
    //         \-> E 1.0 -> C 1.1
    setUpSkyFunctions(
        "A",
        ImmutableMap.<ModuleKey, Module>builder()
            .put(
                createModuleKey("A", ""),
                Module.builder()
                    .setName("A")
                    .setVersion(Version.parse("1.0"))
                    .addDep("B", createModuleKey("B", "1.0"))
                    .addDep("C", createModuleKey("C", "1.0"))
                    .addDep("D", createModuleKey("D", "1.0"))
                    .addDep("E", createModuleKey("E", "1.0"))
                    .build())
            .put(
                createModuleKey("B", "1.0"),
                Module.builder()
                    .setName("B")
                    .setVersion(Version.parse("1.0"))
                    .addDep("C", createModuleKey("C", "2.0"))
                    .build())
            .put(
                createModuleKey("C", "2.0"),
                Module.builder()
                    .setName("C")
                    .setVersion(Version.parse("2.0"))
                    .setCompatibilityLevel(2)
                    .build())
            .put(
                createModuleKey("C", "1.0"),
                Module.builder()
                    .setName("C")
                    .setVersion(Version.parse("1.0"))
                    .setCompatibilityLevel(1)
                    .build())
            .put(
                createModuleKey("D", "1.0"),
                Module.builder()
                    .setName("D")
                    .setVersion(Version.parse("1.0"))
                    .addDep("B", createModuleKey("B", "1.1"))
                    .build())
            .put(
                createModuleKey("B", "1.1"),
                Module.builder().setName("B").setVersion(Version.parse("1.1")).build())
            .put(
                createModuleKey("E", "1.0"),
                Module.builder()
                    .setName("E")
                    .setVersion(Version.parse("1.0"))
                    .addDep("C", createModuleKey("C", "1.1"))
                    .build())
            .put(
                createModuleKey("C", "1.1"),
                Module.builder()
                    .setName("C")
                    .setVersion(Version.parse("1.1"))
                    .setCompatibilityLevel(1)
                    .build())
            .build(),
        ImmutableMap.of());

    EvaluationResult<SelectionValue> result =
        driver.evaluate(ImmutableList.of(SelectionValue.KEY), evaluationContext);
    if (result.hasError()) {
      fail(result.getError().toString());
    }
    // After selection, C 2.0 is gone, so we're okay.
    // A 1.0 -> B 1.1
    //       \-> C 1.1
    //        \-> D 1.0 -> B 1.1
    //         \-> E 1.0 -> C 1.1
    SelectionValue selectionValue = result.get(SelectionValue.KEY);
    assertThat(selectionValue.getRootModuleName()).isEqualTo("A");
    assertThat(selectionValue.getDepGraph())
        .containsExactly(
            createModuleKey("A", ""),
            Module.builder()
                .setName("A")
                .setVersion(Version.parse("1.0"))
                .addDep("B", createModuleKey("B", "1.1"))
                .addDep("C", createModuleKey("C", "1.1"))
                .addDep("D", createModuleKey("D", "1.0"))
                .addDep("E", createModuleKey("E", "1.0"))
                .build(),
            createModuleKey("B", "1.1"),
            Module.builder().setName("B").setVersion(Version.parse("1.1")).build(),
            createModuleKey("C", "1.1"),
            Module.builder()
                .setName("C")
                .setVersion(Version.parse("1.1"))
                .setCompatibilityLevel(1)
                .build(),
            createModuleKey("D", "1.0"),
            Module.builder()
                .setName("D")
                .setVersion(Version.parse("1.0"))
                .addDep("B", createModuleKey("B", "1.1"))
                .build(),
            createModuleKey("E", "1.0"),
            Module.builder()
                .setName("E")
                .setVersion(Version.parse("1.0"))
                .addDep("C", createModuleKey("C", "1.1"))
                .build());
  }

  @Test
  public void multipleVersionOverride_fork_allowedVersionMissingInDepGraph() throws Exception {
    setUpSkyFunctions(
        "A",
        ImmutableMap.<ModuleKey, Module>builder()
            .put(
                createModuleKey("A", ""),
                Module.builder()
                    .setName("A")
                    .setVersion(Version.EMPTY)
                    .addDep("B1", createModuleKey("B", "1.0"))
                    .addDep("B2", createModuleKey("B", "2.0"))
                    .build())
            .put(
                createModuleKey("B", "1.0"),
                Module.builder().setName("B").setVersion(Version.parse("1.0")).build())
            .put(
                createModuleKey("B", "2.0"),
                Module.builder().setName("B").setVersion(Version.parse("2.0")).build())
            .build(),
        ImmutableMap.of(
            "B",
            MultipleVersionOverride.create(
                ImmutableList.of(Version.parse("1.0"), Version.parse("2.0"), Version.parse("3.0")),
                "")));

    EvaluationResult<SelectionValue> result =
        driver.evaluate(ImmutableList.of(SelectionValue.KEY), evaluationContext);
    assertThat(result.hasError()).isTrue();
    String error = result.getError().toString();
    assertThat(error)
        .contains(
            "multiple_version_override for module B contains version 3.0, but it doesn't exist in"
                + " the dependency graph");
  }

  @Test
  public void multipleVersionOverride_fork_goodCase() throws Exception {
    // For more complex good cases, see the "diamond" test cases below.
    setUpSkyFunctions(
        "A",
        ImmutableMap.<ModuleKey, Module>builder()
            .put(
                createModuleKey("A", ""),
                Module.builder()
                    .setName("A")
                    .setVersion(Version.EMPTY)
                    .addDep("B1", createModuleKey("B", "1.0"))
                    .addDep("B2", createModuleKey("B", "2.0"))
                    .build())
            .put(
                createModuleKey("B", "1.0"),
                Module.builder().setName("B").setVersion(Version.parse("1.0")).build())
            .put(
                createModuleKey("B", "2.0"),
                Module.builder().setName("B").setVersion(Version.parse("2.0")).build())
            .build(),
        ImmutableMap.of(
            "B",
            MultipleVersionOverride.create(
                ImmutableList.of(Version.parse("1.0"), Version.parse("2.0")), "")));

    EvaluationResult<SelectionValue> result =
        driver.evaluate(ImmutableList.of(SelectionValue.KEY), evaluationContext);
    if (result.hasError()) {
      fail(result.getError().toString());
    }
    SelectionValue selectionValue = result.get(SelectionValue.KEY);
    assertThat(selectionValue.getRootModuleName()).isEqualTo("A");
    assertThat(selectionValue.getDepGraph())
        .containsExactly(
            createModuleKey("A", ""),
            Module.builder()
                .setName("A")
                .setVersion(Version.EMPTY)
                .addDep("B1", createModuleKey("B", "1.0"))
                .addDep("B2", createModuleKey("B", "2.0"))
                .build(),
            createModuleKey("B", "1.0"),
            Module.builder().setName("B").setVersion(Version.parse("1.0")).build(),
            createModuleKey("B", "2.0"),
            Module.builder().setName("B").setVersion(Version.parse("2.0")).build());
  }

  @Test
  public void multipleVersionOverride_fork_sameVersionUsedTwice() throws Exception {
    setUpSkyFunctions(
        "A",
        ImmutableMap.<ModuleKey, Module>builder()
            .put(
                createModuleKey("A", ""),
                Module.builder()
                    .setName("A")
                    .setVersion(Version.EMPTY)
                    .addDep("B1", createModuleKey("B", "1.0"))
                    .addDep("B2", createModuleKey("B", "1.3"))
                    .addDep("B3", createModuleKey("B", "1.5"))
                    .build())
            .put(
                createModuleKey("B", "1.0"),
                Module.builder().setName("B").setVersion(Version.parse("1.0")).build())
            .put(
                createModuleKey("B", "1.3"),
                Module.builder().setName("B").setVersion(Version.parse("1.3")).build())
            .put(
                createModuleKey("B", "1.5"),
                Module.builder().setName("B").setVersion(Version.parse("1.5")).build())
            .build(),
        ImmutableMap.of(
            "B",
            MultipleVersionOverride.create(
                ImmutableList.of(Version.parse("1.0"), Version.parse("1.5")), "")));

    EvaluationResult<SelectionValue> result =
        driver.evaluate(ImmutableList.of(SelectionValue.KEY), evaluationContext);
    assertThat(result.hasError()).isTrue();
    String error = result.getError().toString();
    assertThat(error)
        .containsMatch(
            "A@_ depends on B@1.5 at least twice \\(with repo names (B2 and B3)|(B3 and B2)\\)");
  }

  @Test
  public void multipleVersionOverride_diamond_differentCompatibilityLevels() throws Exception {
    setUpSkyFunctions(
        "A",
        ImmutableMap.<ModuleKey, Module>builder()
            .put(
                createModuleKey("A", ""),
                Module.builder()
                    .setName("A")
                    .setVersion(Version.EMPTY)
                    .addDep("BfromA", createModuleKey("B", "1.0"))
                    .addDep("CfromA", createModuleKey("C", "2.0"))
                    .build())
            .put(
                createModuleKey("B", "1.0"),
                Module.builder()
                    .setName("B")
                    .setVersion(Version.parse("1.0"))
                    .addDep("DfromB", createModuleKey("D", "1.0"))
                    .build())
            .put(
                createModuleKey("C", "2.0"),
                Module.builder()
                    .setName("C")
                    .setVersion(Version.parse("2.0"))
                    .addDep("DfromC", createModuleKey("D", "2.0"))
                    .build())
            .put(
                createModuleKey("D", "1.0"),
                Module.builder()
                    .setName("D")
                    .setVersion(Version.parse("1.0"))
                    .setCompatibilityLevel(1)
                    .build())
            .put(
                createModuleKey("D", "2.0"),
                Module.builder()
                    .setName("D")
                    .setVersion(Version.parse("2.0"))
                    .setCompatibilityLevel(2)
                    .build())
            .build(),
        ImmutableMap.of(
            "D",
            MultipleVersionOverride.create(
                ImmutableList.of(Version.parse("1.0"), Version.parse("2.0")), "")));

    EvaluationResult<SelectionValue> result =
        driver.evaluate(ImmutableList.of(SelectionValue.KEY), evaluationContext);
    if (result.hasError()) {
      fail(result.getError().toString());
    }
    SelectionValue selectionValue = result.get(SelectionValue.KEY);
    assertThat(selectionValue.getRootModuleName()).isEqualTo("A");
    assertThat(selectionValue.getDepGraph())
        .containsExactly(
            createModuleKey("A", ""),
            Module.builder()
                .setName("A")
                .setVersion(Version.EMPTY)
                .addDep("BfromA", createModuleKey("B", "1.0"))
                .addDep("CfromA", createModuleKey("C", "2.0"))
                .build(),
            createModuleKey("B", "1.0"),
            Module.builder()
                .setName("B")
                .setVersion(Version.parse("1.0"))
                .addDep("DfromB", createModuleKey("D", "1.0"))
                .build(),
            createModuleKey("C", "2.0"),
            Module.builder()
                .setName("C")
                .setVersion(Version.parse("2.0"))
                .addDep("DfromC", createModuleKey("D", "2.0"))
                .build(),
            createModuleKey("D", "1.0"),
            Module.builder()
                .setName("D")
                .setVersion(Version.parse("1.0"))
                .setCompatibilityLevel(1)
                .build(),
            createModuleKey("D", "2.0"),
            Module.builder()
                .setName("D")
                .setVersion(Version.parse("2.0"))
                .setCompatibilityLevel(2)
                .build());
  }

  @Test
  public void multipleVersionOverride_diamond_sameCompatibilityLevel() throws Exception {
    setUpSkyFunctions(
        "A",
        ImmutableMap.<ModuleKey, Module>builder()
            .put(
                createModuleKey("A", ""),
                Module.builder()
                    .setName("A")
                    .setVersion(Version.EMPTY)
                    .addDep("BfromA", createModuleKey("B", "1.0"))
                    .addDep("CfromA", createModuleKey("C", "2.0"))
                    .build())
            .put(
                createModuleKey("B", "1.0"),
                Module.builder()
                    .setName("B")
                    .setVersion(Version.parse("1.0"))
                    .addDep("DfromB", createModuleKey("D", "1.0"))
                    .build())
            .put(
                createModuleKey("C", "2.0"),
                Module.builder()
                    .setName("C")
                    .setVersion(Version.parse("2.0"))
                    .addDep("DfromC", createModuleKey("D", "2.0"))
                    .build())
            .put(
                createModuleKey("D", "1.0"),
                Module.builder().setName("D").setVersion(Version.parse("1.0")).build())
            .put(
                createModuleKey("D", "2.0"),
                Module.builder().setName("D").setVersion(Version.parse("2.0")).build())
            .build(),
        ImmutableMap.of(
            "D",
            MultipleVersionOverride.create(
                ImmutableList.of(Version.parse("1.0"), Version.parse("2.0")), "")));

    EvaluationResult<SelectionValue> result =
        driver.evaluate(ImmutableList.of(SelectionValue.KEY), evaluationContext);
    if (result.hasError()) {
      fail(result.getError().toString());
    }
    SelectionValue selectionValue = result.get(SelectionValue.KEY);
    assertThat(selectionValue.getRootModuleName()).isEqualTo("A");
    assertThat(selectionValue.getDepGraph())
        .containsExactly(
            createModuleKey("A", ""),
            Module.builder()
                .setName("A")
                .setVersion(Version.EMPTY)
                .addDep("BfromA", createModuleKey("B", "1.0"))
                .addDep("CfromA", createModuleKey("C", "2.0"))
                .build(),
            createModuleKey("B", "1.0"),
            Module.builder()
                .setName("B")
                .setVersion(Version.parse("1.0"))
                .addDep("DfromB", createModuleKey("D", "1.0"))
                .build(),
            createModuleKey("C", "2.0"),
            Module.builder()
                .setName("C")
                .setVersion(Version.parse("2.0"))
                .addDep("DfromC", createModuleKey("D", "2.0"))
                .build(),
            createModuleKey("D", "1.0"),
            Module.builder().setName("D").setVersion(Version.parse("1.0")).build(),
            createModuleKey("D", "2.0"),
            Module.builder().setName("D").setVersion(Version.parse("2.0")).build());
  }

  @Test
  public void multipleVersionOverride_diamond_snappingToNextHighestVersion() throws Exception {
    // A --> B1@1.0 -> C@1.0
    //   \-> B2@1.0 -> C@1.3  [allowed]
    //   \-> B3@1.0 -> C@1.5
    //   \-> B4@1.0 -> C@1.7  [allowed]
    //   \-> B5@1.0 -> C@2.0  [allowed]
    setUpSkyFunctions(
        "A",
        ImmutableMap.<ModuleKey, Module>builder()
            .put(
                createModuleKey("A", ""),
                Module.builder()
                    .setName("A")
                    .setVersion(Version.EMPTY)
                    .addDep("B1", createModuleKey("B1", "1.0"))
                    .addDep("B2", createModuleKey("B2", "1.0"))
                    .addDep("B3", createModuleKey("B3", "1.0"))
                    .addDep("B4", createModuleKey("B4", "1.0"))
                    .addDep("B5", createModuleKey("B5", "1.0"))
                    .build())
            .put(
                createModuleKey("B1", "1.0"),
                Module.builder()
                    .setName("B1")
                    .setVersion(Version.parse("1.0"))
                    .addDep("C", createModuleKey("C", "1.0"))
                    .build())
            .put(
                createModuleKey("B2", "1.0"),
                Module.builder()
                    .setName("B2")
                    .setVersion(Version.parse("1.0"))
                    .addDep("C", createModuleKey("C", "1.3"))
                    .build())
            .put(
                createModuleKey("B3", "1.0"),
                Module.builder()
                    .setName("B3")
                    .setVersion(Version.parse("1.0"))
                    .addDep("C", createModuleKey("C", "1.5"))
                    .build())
            .put(
                createModuleKey("B4", "1.0"),
                Module.builder()
                    .setName("B4")
                    .setVersion(Version.parse("1.0"))
                    .addDep("C", createModuleKey("C", "1.7"))
                    .build())
            .put(
                createModuleKey("B5", "1.0"),
                Module.builder()
                    .setName("B5")
                    .setVersion(Version.parse("1.0"))
                    .addDep("C", createModuleKey("C", "2.0"))
                    .build())
            .put(
                createModuleKey("C", "1.0"),
                Module.builder()
                    .setName("C")
                    .setVersion(Version.parse("1.0"))
                    .setCompatibilityLevel(1)
                    .build())
            .put(
                createModuleKey("C", "1.3"),
                Module.builder()
                    .setName("C")
                    .setVersion(Version.parse("1.3"))
                    .setCompatibilityLevel(1)
                    .build())
            .put(
                createModuleKey("C", "1.5"),
                Module.builder()
                    .setName("C")
                    .setVersion(Version.parse("1.5"))
                    .setCompatibilityLevel(1)
                    .build())
            .put(
                createModuleKey("C", "1.7"),
                Module.builder()
                    .setName("C")
                    .setVersion(Version.parse("1.7"))
                    .setCompatibilityLevel(1)
                    .build())
            .put(
                createModuleKey("C", "2.0"),
                Module.builder()
                    .setName("C")
                    .setVersion(Version.parse("2.0"))
                    .setCompatibilityLevel(2)
                    .build())
            .build(),
        ImmutableMap.of(
            "C",
            MultipleVersionOverride.create(
                ImmutableList.of(Version.parse("1.3"), Version.parse("1.7"), Version.parse("2.0")),
                "")));

    EvaluationResult<SelectionValue> result =
        driver.evaluate(ImmutableList.of(SelectionValue.KEY), evaluationContext);
    if (result.hasError()) {
      fail(result.getError().toString());
    }
    SelectionValue selectionValue = result.get(SelectionValue.KEY);
    assertThat(selectionValue.getRootModuleName()).isEqualTo("A");
    // A --> B1@1.0 -> C@1.3  [originally C@1.0]
    //   \-> B2@1.0 -> C@1.3  [allowed]
    //   \-> B3@1.0 -> C@1.7  [originally C@1.5]
    //   \-> B4@1.0 -> C@1.7  [allowed]
    //   \-> B5@1.0 -> C@2.0  [allowed]
    assertThat(selectionValue.getDepGraph())
        .containsExactly(
            createModuleKey("A", ""),
            Module.builder()
                .setName("A")
                .setVersion(Version.EMPTY)
                .addDep("B1", createModuleKey("B1", "1.0"))
                .addDep("B2", createModuleKey("B2", "1.0"))
                .addDep("B3", createModuleKey("B3", "1.0"))
                .addDep("B4", createModuleKey("B4", "1.0"))
                .addDep("B5", createModuleKey("B5", "1.0"))
                .build(),
            createModuleKey("B1", "1.0"),
            Module.builder()
                .setName("B1")
                .setVersion(Version.parse("1.0"))
                .addDep("C", createModuleKey("C", "1.3"))
                .build(),
            createModuleKey("B2", "1.0"),
            Module.builder()
                .setName("B2")
                .setVersion(Version.parse("1.0"))
                .addDep("C", createModuleKey("C", "1.3"))
                .build(),
            createModuleKey("B3", "1.0"),
            Module.builder()
                .setName("B3")
                .setVersion(Version.parse("1.0"))
                .addDep("C", createModuleKey("C", "1.7"))
                .build(),
            createModuleKey("B4", "1.0"),
            Module.builder()
                .setName("B4")
                .setVersion(Version.parse("1.0"))
                .addDep("C", createModuleKey("C", "1.7"))
                .build(),
            createModuleKey("B5", "1.0"),
            Module.builder()
                .setName("B5")
                .setVersion(Version.parse("1.0"))
                .addDep("C", createModuleKey("C", "2.0"))
                .build(),
            createModuleKey("C", "1.3"),
            Module.builder()
                .setName("C")
                .setVersion(Version.parse("1.3"))
                .setCompatibilityLevel(1)
                .build(),
            createModuleKey("C", "1.7"),
            Module.builder()
                .setName("C")
                .setVersion(Version.parse("1.7"))
                .setCompatibilityLevel(1)
                .build(),
            createModuleKey("C", "2.0"),
            Module.builder()
                .setName("C")
                .setVersion(Version.parse("2.0"))
                .setCompatibilityLevel(2)
                .build());
  }

  @Test
  public void multipleVersionOverride_diamond_dontSnapToDifferentCompatibility() throws Exception {
    // A --> B1@1.0 -> C@1.0  [allowed]
    //   \-> B2@1.0 -> C@1.7
    //   \-> B3@1.0 -> C@2.0  [allowed]
    setUpSkyFunctions(
        "A",
        ImmutableMap.<ModuleKey, Module>builder()
            .put(
                createModuleKey("A", ""),
                Module.builder()
                    .setName("A")
                    .setVersion(Version.EMPTY)
                    .addDep("B1", createModuleKey("B1", "1.0"))
                    .addDep("B2", createModuleKey("B2", "1.0"))
                    .addDep("B3", createModuleKey("B3", "1.0"))
                    .build())
            .put(
                createModuleKey("B1", "1.0"),
                Module.builder()
                    .setName("B1")
                    .setVersion(Version.parse("1.0"))
                    .addDep("C", createModuleKey("C", "1.0"))
                    .build())
            .put(
                createModuleKey("B2", "1.0"),
                Module.builder()
                    .setName("B2")
                    .setVersion(Version.parse("1.0"))
                    .addDep("C", createModuleKey("C", "1.7"))
                    .build())
            .put(
                createModuleKey("B3", "1.0"),
                Module.builder()
                    .setName("B3")
                    .setVersion(Version.parse("1.0"))
                    .addDep("C", createModuleKey("C", "2.0"))
                    .build())
            .put(
                createModuleKey("C", "1.0"),
                Module.builder()
                    .setName("C")
                    .setVersion(Version.parse("1.0"))
                    .setCompatibilityLevel(1)
                    .build())
            .put(
                createModuleKey("C", "1.7"),
                Module.builder()
                    .setName("C")
                    .setVersion(Version.parse("1.7"))
                    .setCompatibilityLevel(1)
                    .build())
            .put(
                createModuleKey("C", "2.0"),
                Module.builder()
                    .setName("C")
                    .setVersion(Version.parse("2.0"))
                    .setCompatibilityLevel(2)
                    .build())
            .build(),
        ImmutableMap.of(
            "C",
            MultipleVersionOverride.create(
                ImmutableList.of(Version.parse("1.0"), Version.parse("2.0")), "")));

    EvaluationResult<SelectionValue> result =
        driver.evaluate(ImmutableList.of(SelectionValue.KEY), evaluationContext);
    assertThat(result.hasError()).isTrue();
    String error = result.getError().toString();
    assertThat(error)
        .contains(
            "B2@1.0 depends on C@1.7 which is not allowed by the multiple_version_override on C,"
                + " which allows only [1.0, 2.0]");
  }

  @Test
  public void multipleVersionOverride_diamond_unknownCompatibility() throws Exception {
    // A --> B1@1.0 -> C@1.0  [allowed]
    //   \-> B2@1.0 -> C@2.0  [allowed]
    //   \-> B3@1.0 -> C@3.0
    setUpSkyFunctions(
        "A",
        ImmutableMap.<ModuleKey, Module>builder()
            .put(
                createModuleKey("A", ""),
                Module.builder()
                    .setName("A")
                    .setVersion(Version.EMPTY)
                    .addDep("B1", createModuleKey("B1", "1.0"))
                    .addDep("B2", createModuleKey("B2", "1.0"))
                    .addDep("B3", createModuleKey("B3", "1.0"))
                    .build())
            .put(
                createModuleKey("B1", "1.0"),
                Module.builder()
                    .setName("B1")
                    .setVersion(Version.parse("1.0"))
                    .addDep("C", createModuleKey("C", "1.0"))
                    .build())
            .put(
                createModuleKey("B2", "1.0"),
                Module.builder()
                    .setName("B2")
                    .setVersion(Version.parse("1.0"))
                    .addDep("C", createModuleKey("C", "2.0"))
                    .build())
            .put(
                createModuleKey("B3", "1.0"),
                Module.builder()
                    .setName("B3")
                    .setVersion(Version.parse("1.0"))
                    .addDep("C", createModuleKey("C", "3.0"))
                    .build())
            .put(
                createModuleKey("C", "1.0"),
                Module.builder()
                    .setName("C")
                    .setVersion(Version.parse("1.0"))
                    .setCompatibilityLevel(1)
                    .build())
            .put(
                createModuleKey("C", "2.0"),
                Module.builder()
                    .setName("C")
                    .setVersion(Version.parse("2.0"))
                    .setCompatibilityLevel(2)
                    .build())
            .put(
                createModuleKey("C", "3.0"),
                Module.builder()
                    .setName("C")
                    .setVersion(Version.parse("3.0"))
                    .setCompatibilityLevel(3)
                    .build())
            .build(),
        ImmutableMap.of(
            "C",
            MultipleVersionOverride.create(
                ImmutableList.of(Version.parse("1.0"), Version.parse("2.0")), "")));

    EvaluationResult<SelectionValue> result =
        driver.evaluate(ImmutableList.of(SelectionValue.KEY), evaluationContext);
    assertThat(result.hasError()).isTrue();
    String error = result.getError().toString();
    assertThat(error)
        .contains(
            "B3@1.0 depends on C@3.0 which is not allowed by the multiple_version_override on C,"
                + " which allows only [1.0, 2.0]");
  }

  @Test
  public void multipleVersionOverride_diamond_badVersionsAreOkayIfUnreferenced() throws Exception {
    // A --> B1@1.0 --> C@1.0  [allowed]
    //   \          \-> B2@1.1
    //   \-> B2@1.0 --> C@1.5
    //   \-> B3@1.0 --> C@2.0  [allowed]
    //   \          \-> B4@1.1
    //   \-> B4@1.0 --> C@3.0
    setUpSkyFunctions(
        "A",
        ImmutableMap.<ModuleKey, Module>builder()
            .put(
                createModuleKey("A", ""),
                Module.builder()
                    .setName("A")
                    .setVersion(Version.EMPTY)
                    .addDep("B1", createModuleKey("B1", "1.0"))
                    .addDep("B2", createModuleKey("B2", "1.0"))
                    .addDep("B3", createModuleKey("B3", "1.0"))
                    .addDep("B4", createModuleKey("B4", "1.0"))
                    .build())
            .put(
                createModuleKey("B1", "1.0"),
                Module.builder()
                    .setName("B1")
                    .setVersion(Version.parse("1.0"))
                    .addDep("C", createModuleKey("C", "1.0"))
                    .addDep("B2", createModuleKey("B2", "1.1"))
                    .build())
            .put(
                createModuleKey("B2", "1.0"),
                Module.builder()
                    .setName("B2")
                    .setVersion(Version.parse("1.0"))
                    .addDep("C", createModuleKey("C", "1.5"))
                    .build())
            .put(
                createModuleKey("B2", "1.1"),
                Module.builder().setName("B2").setVersion(Version.parse("1.1")).build())
            .put(
                createModuleKey("B3", "1.0"),
                Module.builder()
                    .setName("B3")
                    .setVersion(Version.parse("1.0"))
                    .addDep("C", createModuleKey("C", "2.0"))
                    .addDep("B4", createModuleKey("B4", "1.1"))
                    .build())
            .put(
                createModuleKey("B4", "1.0"),
                Module.builder()
                    .setName("B4")
                    .setVersion(Version.parse("1.0"))
                    .addDep("C", createModuleKey("C", "3.0"))
                    .build())
            .put(
                createModuleKey("B4", "1.1"),
                Module.builder().setName("B4").setVersion(Version.parse("1.1")).build())
            .put(
                createModuleKey("C", "1.0"),
                Module.builder()
                    .setName("C")
                    .setVersion(Version.parse("1.0"))
                    .setCompatibilityLevel(1)
                    .build())
            .put(
                createModuleKey("C", "1.5"),
                Module.builder()
                    .setName("C")
                    .setVersion(Version.parse("1.5"))
                    .setCompatibilityLevel(1)
                    .build())
            .put(
                createModuleKey("C", "2.0"),
                Module.builder()
                    .setName("C")
                    .setVersion(Version.parse("2.0"))
                    .setCompatibilityLevel(2)
                    .build())
            .put(
                createModuleKey("C", "3.0"),
                Module.builder()
                    .setName("C")
                    .setVersion(Version.parse("3.0"))
                    .setCompatibilityLevel(3)
                    .build())
            .build(),
        ImmutableMap.of(
            "C",
            MultipleVersionOverride.create(
                ImmutableList.of(Version.parse("1.0"), Version.parse("2.0")), "")));

    EvaluationResult<SelectionValue> result =
        driver.evaluate(ImmutableList.of(SelectionValue.KEY), evaluationContext);
    if (result.hasError()) {
      fail(result.getError().toString());
    }
    SelectionValue selectionValue = result.get(SelectionValue.KEY);
    assertThat(selectionValue.getRootModuleName()).isEqualTo("A");
    // A --> B1@1.0 --> C@1.0  [allowed]
    //   \          \-> B2@1.1
    //   \-> B2@1.1
    //   \-> B3@1.0 --> C@2.0  [allowed]
    //   \          \-> B4@1.1
    //   \-> B4@1.1
    // C@1.5 and C@3.0, the versions violating the allowlist, are gone.
    assertThat(selectionValue.getDepGraph())
        .containsExactly(
            createModuleKey("A", ""),
            Module.builder()
                .setName("A")
                .setVersion(Version.EMPTY)
                .addDep("B1", createModuleKey("B1", "1.0"))
                .addDep("B2", createModuleKey("B2", "1.1"))
                .addDep("B3", createModuleKey("B3", "1.0"))
                .addDep("B4", createModuleKey("B4", "1.1"))
                .build(),
            createModuleKey("B1", "1.0"),
            Module.builder()
                .setName("B1")
                .setVersion(Version.parse("1.0"))
                .addDep("C", createModuleKey("C", "1.0"))
                .addDep("B2", createModuleKey("B2", "1.1"))
                .build(),
            createModuleKey("B2", "1.1"),
            Module.builder().setName("B2").setVersion(Version.parse("1.1")).build(),
            createModuleKey("B3", "1.0"),
            Module.builder()
                .setName("B3")
                .setVersion(Version.parse("1.0"))
                .addDep("C", createModuleKey("C", "2.0"))
                .addDep("B4", createModuleKey("B4", "1.1"))
                .build(),
            createModuleKey("B4", "1.1"),
            Module.builder().setName("B4").setVersion(Version.parse("1.1")).build(),
            createModuleKey("C", "1.0"),
            Module.builder()
                .setName("C")
                .setVersion(Version.parse("1.0"))
                .setCompatibilityLevel(1)
                .build(),
            createModuleKey("C", "2.0"),
            Module.builder()
                .setName("C")
                .setVersion(Version.parse("2.0"))
                .setCompatibilityLevel(2)
                .build());
  }
}
