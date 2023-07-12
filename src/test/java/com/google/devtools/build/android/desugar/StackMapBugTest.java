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
package com.google.devtools.build.android.desugar;

import static com.google.common.truth.Truth.assertThat;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import test.util.TestClassForStackMapFrame;

/** This test case is for testing the fix for b/36654936. */
@RunWith(JUnit4.class)
public class StackMapBugTest {

  /** This is a regression test for b/36654936 (external ASM bug 317785) */
  @Test
  public void testAsmBug317785() {
    int result = TestClassForStackMapFrame.testInputForAsmBug317785();
    assertThat(result).isEqualTo(20);
  }

  /**
   * This is a regression test for b/36654936 (external ASM bug 317785). The first attempted fix
   * cl/152199391 caused stack map frame corruption, which caused the following test to fail.
   */
  @Test
  public void testStackMapFrameCorrectness() {
    TestClassForStackMapFrame testObject = new TestClassForStackMapFrame();
    assertThat(testObject.joinIntegers(0)).isEmpty();
    assertThat(testObject.joinIntegers(1)).isEqualTo("0=Even");
    assertThat(testObject.joinIntegers(2)).isEqualTo("0=Even,1=Odd");
  }
}
