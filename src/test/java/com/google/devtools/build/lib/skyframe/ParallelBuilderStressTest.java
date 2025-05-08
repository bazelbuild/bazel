// Copyright 2015 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skyframe;


import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Stress tests for the parallel builder.
 */
@RunWith(JUnit4.class)
public class ParallelBuilderStressTest extends ParallelBuilderTest {

  /**
   * A larger set of tests using randomly-generated complex dependency graphs.
   */
  @Test
  public void testRandomStressTest1() throws Exception {
    final int numTrials = 2;
    final int numArtifacts = 100;
    final int randomSeed = 43;
    StressTest test = new StressTest(numArtifacts, numTrials, randomSeed);
    test.runStressTest();
  }

  @Test
  public void testRandomStressTest2() throws Exception {
    final int numTrials = 10;
    final int numArtifacts = 10;
    final int randomSeed = 44;
    StressTest test = new StressTest(numArtifacts, numTrials, randomSeed);
    test.runStressTest();
  }
}
