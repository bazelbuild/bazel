// Copyright 2022 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android.r8;

import static com.google.common.truth.Truth.assertThat;

import com.android.tools.r8.CompilationMode;
import com.google.devtools.build.android.r8.CompatDexBuilder.DexingKeyR8;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link DexingKeyR8Test}. */
@RunWith(JUnit4.class)
public class DexingKeyR8Test {

  @Test
  public void testOrderMatters() {
    DexingKeyR8 key = DexingKeyR8.create(CompilationMode.RELEASE, 21, new byte[] {0});
    assertThat(key.compilationMode()).isEqualTo(CompilationMode.RELEASE);
    assertThat(key.minSdkVersion()).isEqualTo(21);
  }

  /**
   * Makes sure that arrays are compared by content. Auto-value promises that but I want to be
   * really sure as we'd never get any cache hits if arrays were compared by reference.
   */
  @Test
  public void testContentMatters() {
    // Exact same input
    assertThat(DexingKeyR8.create(CompilationMode.RELEASE, 21, new byte[] {1, 2, 3}))
        .isEqualTo(DexingKeyR8.create(CompilationMode.RELEASE, 21, new byte[] {1, 2, 3}));
    // Bytecode differs
    assertThat(DexingKeyR8.create(CompilationMode.RELEASE, 21, new byte[] {1, 2, 3}))
        .isNotEqualTo(DexingKeyR8.create(CompilationMode.RELEASE, 21, new byte[] {1, 3, 3}));
    // compilationMode differs
    assertThat(DexingKeyR8.create(CompilationMode.RELEASE, 21, new byte[] {1, 2, 3}))
        .isNotEqualTo(DexingKeyR8.create(CompilationMode.DEBUG, 21, new byte[] {1, 2, 3}));
  }

  // TODO(rules-android): Write a test that asserts the DexingKeyR8 class considers all relevant
  // fields from CompatDexBuilder (see DX's tests for {@link DexingOptions} with @{link
  // DexingKeytest}#testFieldsCoverDexingOptions).
}
