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

package com.google.devtools.build.lib.skyframe;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.fail;

import com.google.common.base.VerifyException;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.HashMultimap;
import com.google.common.collect.ListMultimap;
import com.google.common.collect.Multimap;
import com.google.common.collect.SetMultimap;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.testutil.FoundationTestCase;
import com.google.devtools.build.lib.testutil.Suite;
import com.google.devtools.build.lib.testutil.TestSpec;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests {@link ConfiguredTargetFunction}'s logic for determining each target's
 * {@link BuildConfiguration}.
 */
@TestSpec(size = Suite.SMALL_TESTS)
@RunWith(JUnit4.class)
public class ConfigurationsForTargetsTest extends FoundationTestCase {

  // TODO(gregce): add thorough unit testing for getDynamicConfigurations.

  @Test
  public void testPutOnlyEntryWithSetMultimap() throws Exception {
    internalTestPutOnlyEntry(HashMultimap.create());
  }

  /**
   * Unlike {@link SetMultimap}, {@link ListMultimap} allows duplicate <Key, value> pairs. Make
   * sure that doesn't fool {@link ConfiguredTargetFunction#putOnlyEntry}.
   */
  @Test
  public void testPutOnlyEntryWithListMultimap() throws Exception {
    internalTestPutOnlyEntry(ArrayListMultimap.create());
  }

  private void internalTestPutOnlyEntry(Multimap<String, String> map) throws Exception {
    ConfiguredTargetFunction.putOnlyEntry(map, "foo", "bar");
    ConfiguredTargetFunction.putOnlyEntry(map, "baz", "bar");
    try {
      ConfiguredTargetFunction.putOnlyEntry(map, "foo", "baz");
      fail("Expected an exception when trying to add a new value to an existing key");
    } catch (VerifyException e) {
      assertThat(e).hasMessage("couldn't insert baz: map already has key foo");
    }
    try {
      ConfiguredTargetFunction.putOnlyEntry(map, "foo", "bar");
      fail("Expected an exception when trying to add a pre-existing <key, value> pair");
    } catch (VerifyException e) {
      assertThat(e).hasMessage("couldn't insert bar: map already has key foo");
    }
  }
}
