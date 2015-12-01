// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.collect.nestedset;

import static org.junit.Assert.assertEquals;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Random;

/**
 * Tests for {@link RecordingUniqueifier}.
 */
@RunWith(JUnit4.class)
public class RecordingUniqueifierTest {

  private static final Random RANDOM = new Random();
  
  private static final int VERY_SMALL = 3; // one byte
  private static final int SMALL = 11;     // two bytes
  private static final int MEDIUM = 18;    // three bytes -- unmemoed
  // For this one, the "* 8" is a bytes to bits (1 memo is 1 bit)
  private static final int LARGE = (RecordingUniqueifier.LENGTH_THRESHOLD * 8) + 3;

  private static final int[] SIZES = new int[] {VERY_SMALL, SMALL, MEDIUM, LARGE};
  
  private void doTest(int uniqueInputs, int deterministicHeadSize) throws Exception {
    Preconditions.checkArgument(deterministicHeadSize <= uniqueInputs,
        "deterministicHeadSize must be smaller than uniqueInputs");

      // Setup

      List<Integer> inputList = new ArrayList<>(uniqueInputs);
      Collection<Integer> inputsDeduped = new LinkedHashSet<>(uniqueInputs);

      for (int i = 0; i < deterministicHeadSize; i++) { // deterministic head
        inputList.add(i);
        inputsDeduped.add(i);
      }

      while (inputsDeduped.size() < uniqueInputs) { // random selectees
        Integer i = RANDOM.nextInt(uniqueInputs);
        inputList.add(i);
        inputsDeduped.add(i);
      }

      // Unmemoed run

      List<Integer> firstList = new ArrayList<>(uniqueInputs);
      RecordingUniqueifier recordingUniqueifier = new RecordingUniqueifier();
      for (Integer i : inputList) {
        if (recordingUniqueifier.isUnique(i)) {
          firstList.add(i);
        }
      }

      // Potentially memo'ed run

      List<Integer> secondList = new ArrayList<>(uniqueInputs);
      Object memo = recordingUniqueifier.getMemo();
      Uniqueifier uniqueifier = RecordingUniqueifier.createReplayUniqueifier(memo);
      for (Integer i : inputList) {
        if (uniqueifier.isUnique(i)) {
          secondList.add(i);
        }
      }

      // Evaluate results

      inputsDeduped = ImmutableList.copyOf(inputsDeduped);
      assertEquals("Unmemo'ed run has unexpected contents", inputsDeduped, firstList);
      assertEquals("Memo'ed run has unexpected contents", inputsDeduped, secondList);
  }

  private void doTestWithLucidException(int uniqueInputs, int deterministicHeadSize)
      throws Exception {
    try {
      doTest(uniqueInputs, deterministicHeadSize);
    } catch (Exception e) {
      throw new Exception("Failure in size: " + uniqueInputs, e);
    }
  }

  @Test
  public void noInputs() throws Exception {
    doTestWithLucidException(0, 0);
  }
  
  @Test
  public void allUnique() throws Exception {
    for (int size : SIZES) {
      doTestWithLucidException(size, size);
    }
  }

  @Test
  public void fuzzedWithDeterministic2() throws Exception {
    // The way that it is used, we know that the first two additions are not equal.
    // Optimizations were made for this case in small memos.
    for (int size : SIZES) {
      doTestWithLucidException(size, 2);
    }
  }

  @Test
  public void fuzzedWithDeterministic2_otherSizes() throws Exception {
    for (int i = 0; i < 100; i++) {
      int size = RANDOM.nextInt(10000) + 2;
      doTestWithLucidException(size, 2);
    }
  }
}
