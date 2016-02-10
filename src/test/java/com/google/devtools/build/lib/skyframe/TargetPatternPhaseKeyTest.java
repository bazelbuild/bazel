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

import static com.google.devtools.build.lib.skyframe.TargetPatternPhaseKeyTest.Flag.BUILD_TESTS_ONLY;
import static com.google.devtools.build.lib.skyframe.TargetPatternPhaseKeyTest.Flag.COMPILE_ONE_DEPENDENCY;
import static com.google.devtools.build.lib.skyframe.TargetPatternPhaseKeyTest.Flag.DETERMINE_TESTS;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.testing.EqualsTester;
import com.google.common.testing.NullPointerTester;
import com.google.devtools.build.lib.events.NullEventHandler;
import com.google.devtools.build.lib.pkgcache.LoadingOptions;
import com.google.devtools.build.lib.pkgcache.TestFilter;
import com.google.devtools.build.lib.skyframe.TargetPatternPhaseValue.TargetPatternList;
import com.google.devtools.common.options.Options;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import javax.annotation.Nullable;

/** Tests for {@link TargetPatternList}. */
@RunWith(JUnit4.class)
public class TargetPatternPhaseKeyTest {
  static enum Flag {
    COMPILE_ONE_DEPENDENCY,
    BUILD_TESTS_ONLY,
    DETERMINE_TESTS
  }

  @Test
  public void testEquality() throws Exception {
    new EqualsTester()
       .addEqualityGroup(of(ImmutableList.of("a"), "offset"))
       .addEqualityGroup(of(ImmutableList.of("b"), "offset"))
       .addEqualityGroup(of(ImmutableList.of("b"), ""))
       .addEqualityGroup(of(ImmutableList.of("c"), ""))
       .addEqualityGroup(of(ImmutableList.<String>of(), ""))
       .addEqualityGroup(of(ImmutableList.<String>of(), "", null, COMPILE_ONE_DEPENDENCY))
       .addEqualityGroup(of(ImmutableList.<String>of(), "", emptyTestFilter(), BUILD_TESTS_ONLY))
       .addEqualityGroup(of(ImmutableList.<String>of(), "", emptyTestFilter(), DETERMINE_TESTS))
       .testEquals();
  }

  private TargetPatternList of(ImmutableList<String> targetPatterns, String offset,
      @Nullable TestFilter testFilter, Flag... flags) {
    ImmutableSet<Flag> set = ImmutableSet.copyOf(flags);
    boolean compileOneDependency = set.contains(Flag.COMPILE_ONE_DEPENDENCY);
    boolean buildTestsOnly = set.contains(Flag.BUILD_TESTS_ONLY);
    boolean determineTests = set.contains(Flag.DETERMINE_TESTS);
    return new TargetPatternList(targetPatterns, offset, compileOneDependency, buildTestsOnly,
        determineTests, testFilter);
  }

  private TargetPatternList of(ImmutableList<String> targetPatterns, String offset) {
    return of(targetPatterns, offset, null);
  }

  private TestFilter emptyTestFilter() {
    LoadingOptions options = Options.getDefaults(LoadingOptions.class);
    return TestFilter.forOptions(options, NullEventHandler.INSTANCE, ImmutableSet.<String>of());
  }

  @Test
  public void testNull() throws Exception {
    new NullPointerTester()
        .testAllPublicConstructors(TargetPatternList.class);
  }
}
