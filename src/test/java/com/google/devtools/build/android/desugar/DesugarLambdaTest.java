// Copyright 2016 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.android.desugar.testdata.ConstantArgumentsInLambda;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests uncommon lambda scenarios. */
@RunWith(JUnit4.class)
public class DesugarLambdaTest {

  /**
   * Test for b/62060793. Verifies constant lambda arguments that were pushed using *CONST_0
   * instructions.
   */
  @Test
  public void testCallLambdaWithConstants() throws Exception {
    assertThat(ConstantArgumentsInLambda.lambdaWithConstantArguments().call("test"))
        .isEqualTo("testfalse\00120.00.0049nulltrue");
  }
}
