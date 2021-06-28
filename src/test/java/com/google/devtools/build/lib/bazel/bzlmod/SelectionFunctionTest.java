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
import static org.junit.Assert.fail;

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

  private void setUpDiscoveryResult(String rootModuleName, ImmutableMap<ModuleKey, Module> depGraph)
      throws Exception {
    MemoizingEvaluator evaluator =
        new InMemoryMemoizingEvaluator(
            ImmutableMap.<SkyFunctionName, SkyFunction>builder()
                .put(
                    SkyFunctions.DISCOVERY,
                    new SkyFunction() {
                      @Override
                      public SkyValue compute(SkyKey skyKey, Environment env) {
                        return DiscoveryValue.create(rootModuleName, depGraph, ImmutableMap.of());
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
    setUpDiscoveryResult(
        "A",
        ImmutableMap.<ModuleKey, Module>builder()
            .put(
                ModuleKey.create("A", ""),
                Module.builder()
                    .setName("A")
                    .setVersion("")
                    .addDep("BfromA", ModuleKey.create("B", "1.0"))
                    .addDep("CfromA", ModuleKey.create("C", "2.0"))
                    .build())
            .put(
                ModuleKey.create("B", "1.0"),
                Module.builder()
                    .setName("B")
                    .setVersion("1.0")
                    .addDep("DfromB", ModuleKey.create("D", "1.0"))
                    .build())
            .put(
                ModuleKey.create("C", "2.0"),
                Module.builder()
                    .setName("C")
                    .setVersion("2.0")
                    .addDep("DfromC", ModuleKey.create("D", "2.0"))
                    .build())
            .put(
                ModuleKey.create("D", "1.0"),
                Module.builder().setName("D").setVersion("1.0").build())
            .put(
                ModuleKey.create("D", "2.0"),
                Module.builder().setName("D").setVersion("2.0").build())
            .build());

    EvaluationResult<SelectionValue> result =
        driver.evaluate(ImmutableList.of(SelectionValue.KEY), evaluationContext);
    if (result.hasError()) {
      fail(result.getError().toString());
    }
    SelectionValue selectionValue = result.get(SelectionValue.KEY);
    assertThat(selectionValue.getRootModuleName()).isEqualTo("A");
    assertThat(selectionValue.getDepGraph())
        .containsExactly(
            ModuleKey.create("A", ""),
            Module.builder()
                .setName("A")
                .setVersion("")
                .addDep("BfromA", ModuleKey.create("B", "1.0"))
                .addDep("CfromA", ModuleKey.create("C", "2.0"))
                .build(),
            ModuleKey.create("B", "1.0"),
            Module.builder()
                .setName("B")
                .setVersion("1.0")
                .addDep("DfromB", ModuleKey.create("D", "2.0"))
                .build(),
            ModuleKey.create("C", "2.0"),
            Module.builder()
                .setName("C")
                .setVersion("2.0")
                .addDep("DfromC", ModuleKey.create("D", "2.0"))
                .build(),
            ModuleKey.create("D", "2.0"),
            Module.builder().setName("D").setVersion("2.0").build());
  }

  @Test
  public void testDiamondWithFurtherRemoval() throws Exception {
    setUpDiscoveryResult(
        "A",
        ImmutableMap.<ModuleKey, Module>builder()
            .put(
                ModuleKey.create("A", ""),
                Module.builder()
                    .setName("A")
                    .setVersion("")
                    .addDep("B", ModuleKey.create("B", "1.0"))
                    .addDep("C", ModuleKey.create("C", "2.0"))
                    .build())
            .put(
                ModuleKey.create("B", "1.0"),
                Module.builder()
                    .setName("B")
                    .setVersion("1.0")
                    .addDep("D", ModuleKey.create("D", "1.0"))
                    .build())
            .put(
                ModuleKey.create("C", "2.0"),
                Module.builder()
                    .setName("C")
                    .setVersion("2.0")
                    .addDep("D", ModuleKey.create("D", "2.0"))
                    .build())
            .put(
                ModuleKey.create("D", "1.0"),
                Module.builder()
                    .setName("D")
                    .setVersion("1.0")
                    .addDep("E", ModuleKey.create("E", "1.0"))
                    .build())
            .put(
                ModuleKey.create("D", "2.0"),
                Module.builder().setName("D").setVersion("2.0").build())
            // Only D@1.0 needs E. When D@1.0 is removed, E should be gone as well (even though
            // E@1.0 is selected for E).
            .put(
                ModuleKey.create("E", "1.0"),
                Module.builder().setName("E").setVersion("1.0").build())
            .build());

    EvaluationResult<SelectionValue> result =
        driver.evaluate(ImmutableList.of(SelectionValue.KEY), evaluationContext);
    if (result.hasError()) {
      fail(result.getError().toString());
    }
    SelectionValue selectionValue = result.get(SelectionValue.KEY);
    assertThat(selectionValue.getRootModuleName()).isEqualTo("A");
    assertThat(selectionValue.getDepGraph())
        .containsExactly(
            ModuleKey.create("A", ""),
            Module.builder()
                .setName("A")
                .setVersion("")
                .addDep("B", ModuleKey.create("B", "1.0"))
                .addDep("C", ModuleKey.create("C", "2.0"))
                .build(),
            ModuleKey.create("B", "1.0"),
            Module.builder()
                .setName("B")
                .setVersion("1.0")
                .addDep("D", ModuleKey.create("D", "2.0"))
                .build(),
            ModuleKey.create("C", "2.0"),
            Module.builder()
                .setName("C")
                .setVersion("2.0")
                .addDep("D", ModuleKey.create("D", "2.0"))
                .build(),
            ModuleKey.create("D", "2.0"),
            Module.builder().setName("D").setVersion("2.0").build());
  }

  @Test
  public void testCircularDependencyDueToSelection() throws Exception {
    setUpDiscoveryResult(
        "A",
        ImmutableMap.<ModuleKey, Module>builder()
            .put(
                ModuleKey.create("A", ""),
                Module.builder()
                    .setName("A")
                    .setVersion("")
                    .addDep("B", ModuleKey.create("B", "1.0"))
                    .build())
            .put(
                ModuleKey.create("B", "1.0"),
                Module.builder()
                    .setName("B")
                    .setVersion("1.0")
                    .addDep("C", ModuleKey.create("C", "2.0"))
                    .build())
            .put(
                ModuleKey.create("C", "2.0"),
                Module.builder()
                    .setName("C")
                    .setVersion("2.0")
                    .addDep("B", ModuleKey.create("B", "1.0-pre"))
                    .build())
            .put(
                ModuleKey.create("B", "1.0-pre"),
                Module.builder()
                    .setName("B")
                    .setVersion("1.0-pre")
                    .addDep("D", ModuleKey.create("D", "1.0"))
                    .build())
            .put(
                ModuleKey.create("D", "1.0"),
                Module.builder().setName("D").setVersion("1.0").build())
            .build());

    EvaluationResult<SelectionValue> result =
        driver.evaluate(ImmutableList.of(SelectionValue.KEY), evaluationContext);
    if (result.hasError()) {
      fail(result.getError().toString());
    }
    SelectionValue selectionValue = result.get(SelectionValue.KEY);
    assertThat(selectionValue.getRootModuleName()).isEqualTo("A");
    assertThat(selectionValue.getDepGraph())
        .containsExactly(
            ModuleKey.create("A", ""),
            Module.builder()
                .setName("A")
                .setVersion("")
                .addDep("B", ModuleKey.create("B", "1.0"))
                .build(),
            ModuleKey.create("B", "1.0"),
            Module.builder()
                .setName("B")
                .setVersion("1.0")
                .addDep("C", ModuleKey.create("C", "2.0"))
                .build(),
            ModuleKey.create("C", "2.0"),
            Module.builder()
                .setName("C")
                .setVersion("2.0")
                .addDep("B", ModuleKey.create("B", "1.0"))
                .build());
    // D is completely gone.
  }
}
