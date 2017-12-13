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
package com.google.devtools.build.android.dexer;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.android.dexer.Dexing.DexingKey;
import com.google.devtools.build.android.dexer.Dexing.DexingOptions;
import java.lang.reflect.Member;
import java.util.HashSet;
import java.util.Set;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link DexingKeyTest}. */
@RunWith(JUnit4.class)
public class DexingKeyTest {

  @Test
  public void testOrderMatters() {
    DexingKey key = DexingKey.create(false, true, 2, new byte[0]);
    assertThat(key.localInfo()).isFalse();
    assertThat(key.optimize()).isTrue();
    assertThat(key.positionInfo()).isEqualTo(2);
  }

  /**
   * Makes sure that arrays are compared by content.  Auto-value promises that but I want to be
   * really sure as we'd never get any cache hits if arrays were compared by reference.
   */
  @Test
  public void testContentMatters() {
    assertThat(DexingKey.create(false, false, 1, new byte[] { 1, 2, 3 }))
        .isEqualTo(DexingKey.create(false, false, 1, new byte[] { 1, 2, 3 }));
    assertThat(DexingKey.create(false, false, 1, new byte[] { 1, 2, 3 }))
        .isNotEqualTo(DexingKey.create(false, false, 1, new byte[] { 1, 3, 3 }));
  }

  /**
   * Makes sure that all {@link DexingOptions} (that can affect resulting {@code .dex} files) are
   * reflected in {@link DexingKey}.  Forgetting to reflect new options in {@link DexingKey} could
   * result in spurious cache hits when {@link DexingKey} is used as cache key for dexing results
   * (a {@code .dex} file created with different options would be used where it shouldn't), so we
   * have this test to make sure that doesn't happen.
   */
  @Test
  public void testFieldsCoverDexingOptions() {
    Set<String> keyMethods = names(DexingKey.class.getDeclaredMethods());
    Set<String> optionsFields = names(DexingOptions.class.getDeclaredFields());
    keyMethods.remove("create"); // Ignore factory method (we just want accessors)
    keyMethods.remove("classfileContent"); // Ignore classfile content (we just want options)
    keyMethods.remove("$jacocoInit"); // Ignore extra method generated in coverage builds
    optionsFields.remove("printWarnings"); // Doesn't affect resulting dex files
    optionsFields.remove("$jacocoData"); // Ignore extra field generated in coverage builds
    assertThat(keyMethods).containsExactlyElementsIn(optionsFields);
  }

  private static Set<String> names(Member... members) {
    HashSet<String> result = new HashSet<>();
    for (Member member : members) {
      result.add(member.getName());
    }
    return result;
  }
}

