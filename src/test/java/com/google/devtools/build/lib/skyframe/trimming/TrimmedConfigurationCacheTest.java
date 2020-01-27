// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skyframe.trimming;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth8.assertThat;

import com.google.common.collect.ImmutableMap;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for the TrimmedConfigurationCache. */
@RunWith(JUnit4.class)
public final class TrimmedConfigurationCacheTest {

  private TrimmedConfigurationCache<TestKey, String, ImmutableMap<String, String>> cache;

  @Before
  public void initializeCache() {
    cache = TestKey.newCache();
  }

  @Test
  public void get_onFreshCache_returnsEmpty() throws Exception {
    assertThat(cache.get(TestKey.parse("<A: 1> //foo"))).isEmpty();
  }

  @Test
  public void get_afterAddingSubsetCacheEntry_returnsMatchingValue() throws Exception {
    TestKey canonicalKey = TestKey.parse("<A: 1, B: 1> //foo");
    cache.putIfAbsent(canonicalKey, TestKey.parseConfiguration("A: 1"));

    assertThat(cache.get(TestKey.parse("<A: 1, C: 1> //foo"))).hasValue(canonicalKey);
  }

  @Test
  public void get_afterRemovingMatchingCacheEntry_returnsEmpty() throws Exception {
    TestKey canonicalKey = TestKey.parse("<A: 1, B: 1> //foo");
    cache.putIfAbsent(canonicalKey, TestKey.parseConfiguration("A: 1"));
    cache.remove(canonicalKey);

    assertThat(cache.get(TestKey.parse("<A: 1, B: 2> //foo"))).isEmpty();
  }

  @Test
  public void get_afterRemovingCacheEntryWithDifferentConfig_returnsOriginalKey() throws Exception {
    TestKey canonicalKey = TestKey.parse("<A: 1, B: 1> //foo");
    cache.putIfAbsent(canonicalKey, TestKey.parseConfiguration("A: 1"));
    TestKey removedOtherKey = TestKey.parse("<A: 1, B: 2> //foo");
    cache.remove(removedOtherKey);

    assertThat(cache.get(canonicalKey)).hasValue(canonicalKey);
  }

  @Test
  public void get_afterRemovingCacheEntryWithDifferentDescriptor_returnsOriginalKey()
      throws Exception {
    TestKey canonicalKey = TestKey.parse("<A: 1, B: 1> //foo");
    cache.putIfAbsent(canonicalKey, TestKey.parseConfiguration("A: 1"));
    TestKey removedOtherKey = TestKey.parse("<A: 1, B: 1> //bar");
    cache.remove(removedOtherKey);

    assertThat(cache.get(canonicalKey)).hasValue(canonicalKey);
  }

  @Test
  public void get_afterClearingMatchingCacheEntry_returnsEmpty() throws Exception {
    cache.putIfAbsent(TestKey.parse("<A: 1, B: 1> //foo"), TestKey.parseConfiguration("A: 1"));
    cache.clear();

    assertThat(cache.get(TestKey.parse("<A: 1, B: 2> //foo"))).isEmpty();
  }

  @Test
  public void get_afterRemovingAndReAddingCacheEntry_returnsNewValue() throws Exception {
    TestKey oldKey = TestKey.parse("<A: 1, B: 1> //foo");
    ImmutableMap<String, String> trimmedConfiguration = TestKey.parseConfiguration("A: 1");
    cache.putIfAbsent(oldKey, trimmedConfiguration);
    cache.remove(oldKey);
    TestKey newKey = TestKey.parse("<A: 1, C: 1> //foo");
    cache.putIfAbsent(newKey, trimmedConfiguration);

    assertThat(cache.get(TestKey.parse("<A: 1, D: 1> //foo"))).hasValue(newKey);
  }

  @Test
  public void get_afterAddingMatchingConfigurationForDifferentDescriptor_returnsEmpty()
      throws Exception {
    cache.putIfAbsent(TestKey.parse("<A: 1, B: 1> //bar"), TestKey.parseConfiguration("A: 1"));

    assertThat(cache.get(TestKey.parse("<A: 1, B: 1> //foo"))).isEmpty();
  }

  @Test
  public void get_afterAddingNonMatchingConfigurationForSameDescriptor_returnsEmpty()
      throws Exception {
    cache.putIfAbsent(TestKey.parse("<A: 1, B: 1> //foo"), TestKey.parseConfiguration("A: 1"));

    assertThat(cache.get(TestKey.parse("<A: 2, B: 1> //foo"))).isEmpty();
  }

  @Test
  public void get_afterAddingAndInvalidatingMatchingCacheEntry_returnsEmpty() throws Exception {
    TestKey canonicalKey = TestKey.parse("<A: 1, B: 1> //foo");
    cache.putIfAbsent(canonicalKey, TestKey.parseConfiguration("A: 1"));
    cache.invalidate(canonicalKey);

    assertThat(cache.get(TestKey.parse("<A: 1, C: 1> //foo"))).isEmpty();
  }

  @Test
  public void get_afterAddingAndInvalidatingAndReAddingMatchingCacheEntry_returnsOriginalValue()
      throws Exception {
    TestKey oldKey = TestKey.parse("<A: 1, B: 1> //foo");
    ImmutableMap<String, String> trimmedConfiguration = TestKey.parseConfiguration("A: 1");
    cache.putIfAbsent(oldKey, trimmedConfiguration);
    cache.invalidate(oldKey);
    TestKey newKey = TestKey.parse("<A: 1, C: 1> //foo");
    cache.putIfAbsent(newKey, trimmedConfiguration);

    assertThat(cache.get(TestKey.parse("<A: 1, D: 1> //foo"))).hasValue(oldKey);
  }

  @Test
  public void get_afterAddingAndInvalidatingAndRevalidatingMatchingCacheEntry_returnsOriginalValue()
      throws Exception {
    TestKey oldKey = TestKey.parse("<A: 1, B: 1> //foo");
    ImmutableMap<String, String> trimmedConfiguration = TestKey.parseConfiguration("A: 1");
    cache.putIfAbsent(oldKey, trimmedConfiguration);
    cache.invalidate(oldKey);
    cache.revalidate(oldKey);

    assertThat(cache.get(TestKey.parse("<A: 1, D: 1> //foo"))).hasValue(oldKey);
  }

  @Test
  public void get_afterMovingKeyToDifferentTrimming_returnsEmpty() throws Exception {
    TestKey oldKey = TestKey.parse("<A: 1, B: 1> //foo");
    cache.putIfAbsent(oldKey, TestKey.parseConfiguration("A: 1"));
    cache.invalidate(oldKey);
    cache.putIfAbsent(oldKey, TestKey.parseConfiguration("B: 1"));

    assertThat(cache.get(TestKey.parse("<A: 1, B: 2> //foo"))).isEmpty();
  }

  @Test(expected = IllegalArgumentException.class)
  public void putIfAbsent_forNonSubsetConfiguration_throwsIllegalArgumentException()
      throws Exception {
    cache.putIfAbsent(TestKey.parse("<A: 1> //foo"), TestKey.parseConfiguration("A: 2"));
  }

  @Test
  public void putIfAbsent_onFreshCache_returnsInputKey() throws Exception {
    TestKey inputKey = TestKey.parse("<A: 1, B: 1> //foo");

    assertThat(cache.putIfAbsent(inputKey, TestKey.parseConfiguration("A: 1"))).isEqualTo(inputKey);
  }

  @Test
  public void putIfAbsent_afterRemoving_returnsNewKey() throws Exception {
    TestKey oldKey = TestKey.parse("<A: 1, B: 1> //foo");
    ImmutableMap<String, String> trimmedConfiguration = TestKey.parseConfiguration("A: 1");
    cache.putIfAbsent(oldKey, trimmedConfiguration);
    cache.remove(oldKey);
    TestKey newKey = TestKey.parse("<A: 1, B: 2> //foo");

    assertThat(cache.putIfAbsent(newKey, trimmedConfiguration)).isEqualTo(newKey);
  }

  @Test
  public void putIfAbsent_afterAddingEqualConfigurationForDifferentDescriptor_returnsInputKey()
      throws Exception {
    TestKey oldKey = TestKey.parse("<A: 1, B: 1> //foo");
    ImmutableMap<String, String> trimmedConfiguration = TestKey.parseConfiguration("A: 1");
    cache.putIfAbsent(oldKey, trimmedConfiguration);
    TestKey newKey = TestKey.parse("<A: 1, B: 1> //bar");

    assertThat(cache.putIfAbsent(newKey, trimmedConfiguration)).isEqualTo(newKey);
  }

  @Test
  public void putIfAbsent_afterAddingNonEqualConfigurationForSameDescriptor_returnsInputKey()
      throws Exception {
    cache.putIfAbsent(TestKey.parse("<A: 1, B: 1> //foo"), TestKey.parseConfiguration("A: 1"));
    TestKey newKey = TestKey.parse("<A: 2, B: 2> //foo");

    assertThat(cache.putIfAbsent(newKey, TestKey.parseConfiguration("A: 2"))).isEqualTo(newKey);
  }

  @Test
  public void putIfAbsent_afterAddingEqualConfigurationForSameDescriptor_returnsOriginalKey()
      throws Exception {
    TestKey oldKey = TestKey.parse("<A: 1, B: 1> //foo");
    ImmutableMap<String, String> trimmedConfiguration = TestKey.parseConfiguration("A: 1");
    cache.putIfAbsent(oldKey, trimmedConfiguration);

    assertThat(cache.putIfAbsent(TestKey.parse("<A: 1, B: 2> //foo"), trimmedConfiguration))
        .isEqualTo(oldKey);
  }

  @Test
  public void putIfAbsent_afterInvalidatingEqualEntry_returnsOriginalKey() throws Exception {
    TestKey oldKey = TestKey.parse("<A: 1, B: 1> //foo");
    ImmutableMap<String, String> trimmedConfiguration = TestKey.parseConfiguration("A: 1");
    cache.putIfAbsent(oldKey, trimmedConfiguration);
    cache.invalidate(oldKey);

    assertThat(cache.putIfAbsent(TestKey.parse("<A: 1, B: 2> //foo"), trimmedConfiguration))
        .isEqualTo(oldKey);
  }

  @Test
  public void putIfAbsent_forKeyAssociatedWithDifferentTrimming_returnsOldKey() throws Exception {
    TestKey oldKey = TestKey.parse("<A: 1, B: 1> //foo");
    cache.putIfAbsent(oldKey, TestKey.parseConfiguration("A: 1"));
    cache.invalidate(oldKey);

    assertThat(cache.putIfAbsent(oldKey, TestKey.parseConfiguration("B: 1"))).isEqualTo(oldKey);
  }

  @Test
  public void putIfAbsent_afterMovingPreviousAssociatedKeyToNewTrimming_returnsNewKey()
      throws Exception {
    TestKey oldKey = TestKey.parse("<A: 1, B: 1> //foo");
    ImmutableMap<String, String> trimmedA1 = TestKey.parseConfiguration("A: 1");
    ImmutableMap<String, String> trimmedB1 = TestKey.parseConfiguration("B: 1");
    cache.putIfAbsent(oldKey, trimmedA1);
    cache.invalidate(oldKey);
    cache.putIfAbsent(oldKey, trimmedB1);
    cache.invalidate(oldKey);
    TestKey newKey = TestKey.parse("<A: 1, B: 2> //foo");

    // This is testing that oldKey is not still associated with trimmedA1, because now it's
    // associated with trimmedB1 instead.
    assertThat(cache.putIfAbsent(newKey, trimmedA1)).isEqualTo(newKey);
  }

  @Test
  public void putIfAbsent_afterInvalidatingAndReAddingEqualEntry_returnsOriginalKey()
      throws Exception {
    TestKey oldKey = TestKey.parse("<A: 1, B: 1> //foo");
    ImmutableMap<String, String> trimmedConfiguration = TestKey.parseConfiguration("A: 1");
    cache.putIfAbsent(oldKey, trimmedConfiguration);
    cache.invalidate(oldKey);
    cache.putIfAbsent(TestKey.parse("<A: 1, B: 2> //foo"), trimmedConfiguration);

    assertThat(cache.putIfAbsent(TestKey.parse("<A: 1, B: 3> //foo"), trimmedConfiguration))
        .isEqualTo(oldKey);
  }

  @Test
  public void putIfAbsent_afterInvalidatingAndRevalidatingEqualEntry_returnsOriginalKey()
      throws Exception {
    TestKey oldKey = TestKey.parse("<A: 1, B: 1> //foo");
    ImmutableMap<String, String> trimmedConfiguration = TestKey.parseConfiguration("A: 1");
    cache.putIfAbsent(oldKey, trimmedConfiguration);
    cache.invalidate(oldKey);
    cache.revalidate(oldKey);

    assertThat(cache.putIfAbsent(TestKey.parse("<A: 1, B: 2> //foo"), trimmedConfiguration))
        .isEqualTo(oldKey);
  }

  @Test
  public void putIfAbsent_afterAddingAndInvalidatingSubsetConfiguration_returnsInputKey()
      throws Exception {
    TestKey oldKey = TestKey.parse("<A: 1, B: 1> //foo");
    ImmutableMap<String, String> trimmedConfiguration = TestKey.parseConfiguration("A: 1");
    cache.putIfAbsent(oldKey, trimmedConfiguration);
    cache.invalidate(oldKey);
    TestKey newKey = TestKey.parse("<A: 1, B: 2> //foo");

    assertThat(cache.putIfAbsent(newKey, TestKey.parseConfiguration("A: 1, B: 2")))
        .isEqualTo(newKey);
  }

  @Test
  public void putIfAbsent_afterAddingAndInvalidatingSupersetConfiguration_returnsInputKey()
      throws Exception {
    TestKey oldKey = TestKey.parse("<A: 1, B: 1, C: 1> //foo");
    cache.putIfAbsent(oldKey, TestKey.parseConfiguration("A: 1, B: 1"));
    cache.invalidate(oldKey);
    TestKey newKey = TestKey.parse("<A: 1, B: 1, C: 2> //foo");

    assertThat(cache.putIfAbsent(newKey, TestKey.parseConfiguration("A: 1"))).isEqualTo(newKey);
  }

  @Test
  public void invalidate_onEntryNotInCache_doesNotThrow() throws Exception {
    // Expect no exception here.
    cache.invalidate(TestKey.parse("<A: 1> //foo"));
  }

  @Test
  public void invalidate_onAlreadyInvalidatedEntry_doesNotThrow() throws Exception {
    TestKey oldKey = TestKey.parse("<A: 1, B: 1> //foo");
    cache.putIfAbsent(oldKey, TestKey.parseConfiguration("A: 1"));
    cache.invalidate(oldKey);

    // Expect no exception here.
    cache.invalidate(oldKey);
  }

  @Test
  public void invalidate_onRemovedEntry_doesNotThrow() throws Exception {
    TestKey oldKey = TestKey.parse("<A: 1, B: 1> //foo");
    cache.putIfAbsent(oldKey, TestKey.parseConfiguration("A: 1"));
    cache.remove(oldKey);

    // Expect no exception here.
    cache.invalidate(oldKey);
  }

  @Test
  public void revalidate_onEntryNotInCache_doesNotThrow() throws Exception {
    // Expect no exception here.
    cache.revalidate(TestKey.parse("<A: 1> //foo"));
  }

  @Test
  public void revalidate_onAlreadyInvalidatedAndRevalidatedEntry_doesNotThrow() throws Exception {
    TestKey oldKey = TestKey.parse("<A: 1, B: 1> //foo");
    cache.putIfAbsent(oldKey, TestKey.parseConfiguration("A: 1"));
    cache.invalidate(oldKey);
    cache.revalidate(oldKey);

    // Expect no exception here.
    cache.revalidate(oldKey);
  }

  @Test
  public void revalidate_onRemovedEntry_doesNotThrow() throws Exception {
    TestKey oldKey = TestKey.parse("<A: 1, B: 1> //foo");
    cache.putIfAbsent(oldKey, TestKey.parseConfiguration("A: 1"));
    cache.remove(oldKey);

    // Expect no exception here.
    cache.revalidate(oldKey);
  }

  @Test
  public void remove_onEntryNotInCache_doesNotThrow() throws Exception {
    // Expect no exception here.
    cache.remove(TestKey.parse("<A: 1> //foo"));
  }

  @Test
  public void remove_onAlreadyInvalidatedEntry_doesNotThrow() throws Exception {
    TestKey oldKey = TestKey.parse("<A: 1, B: 1> //foo");
    cache.putIfAbsent(oldKey, TestKey.parseConfiguration("A: 1"));
    cache.invalidate(oldKey);

    // Expect no exception here.
    cache.remove(oldKey);
  }

  @Test
  public void remove_onEntryWithDifferentConfiguration_doesNotThrow() throws Exception {
    cache.putIfAbsent(TestKey.parse("<A: 1, B: 1> //foo"), TestKey.parseConfiguration("A: 1"));

    // Expect no exception here.
    cache.remove(TestKey.parse("<A: 1, B: 2> //foo"));
  }

  @Test
  public void remove_onEntryWithDifferentDescriptor_doesNotThrow() throws Exception {
    cache.putIfAbsent(TestKey.parse("<A: 1, B: 1> //foo"), TestKey.parseConfiguration("A: 1"));

    // Expect no exception here.
    cache.remove(TestKey.parse("<A: 1, B: 1> //bar"));
  }

  @Test
  public void remove_onRemovedEntry_doesNotThrow() throws Exception {
    TestKey oldKey = TestKey.parse("<A: 1, B: 1> //foo");
    cache.putIfAbsent(oldKey, TestKey.parseConfiguration("A: 1"));
    cache.remove(oldKey);

    // Expect no exception here.
    cache.remove(oldKey);
  }

  @Test
  public void clear_onEmptyCache_doesNotThrow() throws Exception {
    cache.clear();
  }
}
