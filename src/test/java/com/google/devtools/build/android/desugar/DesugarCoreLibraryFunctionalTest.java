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

/**
 * Test that exercises classes in the {@code testdata} package. This is meant to be run against a
 * desugared version of those classes, which in turn exercise various desugaring features.
 */
@RunWith(JUnit4.class)
public class DesugarCoreLibraryFunctionalTest {

  @Test
  public void testAutoboxedTypeLambda() {
    AutoboxedTypes.Lambda lambdaUse = AutoboxedTypes.autoboxedTypeLambda(1);
    assertThat(lambdaUse.charAt("Karen")).isEqualTo("a");
  }
}
