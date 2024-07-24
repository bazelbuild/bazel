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
package com.google.devtools.build.lib.analysis;

import com.google.devtools.build.lib.analysis.util.OptionsTestCase;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class PlatformOptionsTest extends OptionsTestCase<PlatformOptions> {

  private static final String EXTRA_PLATFORMS_PREFIX = "--extra_execution_platforms=";
  private static final String EXTRA_TOOLCHAINS_PREFIX = "--extra_toolchains=";
  private static final String PLATFORMS_PREFIX = "--platforms=";

  @Override
  protected Class<PlatformOptions> getOptionsClass() {
    return PlatformOptions.class;
  }

  @Test
  public void testExtraPlatforms_orderMatters() throws Exception {
    // It seems that the platforms are considered in order. Picking the wrong one results in broken
    // builds. So add test asserting that order matters.
    PlatformOptions one = createWithPrefix(EXTRA_PLATFORMS_PREFIX, "platform1", "platform2");
    PlatformOptions two = createWithPrefix(EXTRA_PLATFORMS_PREFIX, "platform2", "platform1");
    assertDifferent(one, two);
  }

  @Test
  public void testExtraToolchains_ordering() throws Exception {
    // The ordering matters for tool chains, but the last one in the list has highest priority.
    PlatformOptions one = createWithPrefix(EXTRA_TOOLCHAINS_PREFIX, "one", "two");
    PlatformOptions two = createWithPrefix(EXTRA_TOOLCHAINS_PREFIX, "two", "one");
    assertDifferent(one, two);
  }

  @Test
  public void testExtraToolchains_duplicates() throws Exception {
    // Specifying the same tool chain multiple times is a no-op.
    PlatformOptions one = createWithPrefix(EXTRA_TOOLCHAINS_PREFIX, "one", "one");
    PlatformOptions two = createWithPrefix(EXTRA_TOOLCHAINS_PREFIX, "one");
    assertSame(one, two);
  }

  @Test
  public void testExtraToolchains_duplicates_keepLast() throws Exception {
    // The last toolchain in the list has highest priority, so keep the last of any duplicates.
    PlatformOptions one = createWithPrefix(EXTRA_TOOLCHAINS_PREFIX, "one", "two", "one");
    PlatformOptions two = createWithPrefix(EXTRA_TOOLCHAINS_PREFIX, "two", "one");
    assertSame(one, two);
  }

  @Test
  public void testPlatforms_duplicates() throws Exception {
    PlatformOptions one = createWithPrefix(PLATFORMS_PREFIX, "//p:one,//p:one");
    PlatformOptions two = createWithPrefix(PLATFORMS_PREFIX, "//p:one");
    assertSame(one, two);
  }

  @Test
  public void testPlatforms_extraValues() throws Exception {
    // Only the first value matters.
    PlatformOptions one = createWithPrefix(PLATFORMS_PREFIX, "//one,//two");
    PlatformOptions two = createWithPrefix(PLATFORMS_PREFIX, "//one");
    assertSame(one, two);
  }

  @Test
  public void testPlatforms_orderMatters() throws Exception {
    // Changing the order changes the semantics.
    PlatformOptions foo = createWithPrefix(PLATFORMS_PREFIX, "//one,//two");
    PlatformOptions bar = createWithPrefix(PLATFORMS_PREFIX, "//two,//one");
    assertDifferent(foo, bar);
  }
}
