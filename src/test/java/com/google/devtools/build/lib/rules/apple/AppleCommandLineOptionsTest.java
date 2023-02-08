// Copyright 2023 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.apple;

import com.google.devtools.build.lib.analysis.util.OptionsTestCase;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class AppleCommandLineOptionsTest extends OptionsTestCase<AppleCommandLineOptions> {

  private static final String IOS_CPUS_PREFIX = "--ios_multi_cpus=";
  private static final String WATCHOS_CPUS_PREFIX = "--watchos_cpus=";
  private static final String MACOS_CPUS_PREFIX = "--macos_cpus=";
  private static final String TVOS_CPUS_PREFIX = "--tvos_cpus=";
  private static final String CATALYST_CPUS_PREFIX = "--catalyst_cpus=";

  @Override
  protected Class<AppleCommandLineOptions> getOptionsClass() {
    return AppleCommandLineOptions.class;
  }

  @Test
  public void testIosCpus_ordering() throws Exception {
    AppleCommandLineOptions one = createWithPrefix(IOS_CPUS_PREFIX, "foo", "bar");
    AppleCommandLineOptions two = createWithPrefix(IOS_CPUS_PREFIX, "bar", "foo");
    assertSame(one, two);
  }

  @Test
  public void testIosCpus_duplicates() throws Exception {
    AppleCommandLineOptions one = createWithPrefix(IOS_CPUS_PREFIX, "foo", "foo");
    AppleCommandLineOptions two = createWithPrefix(IOS_CPUS_PREFIX, "foo");
    assertSame(one, two);
  }

  @Test
  public void testWatchosCpus_ordering() throws Exception {
    AppleCommandLineOptions one = createWithPrefix(WATCHOS_CPUS_PREFIX, "foo", "bar");
    AppleCommandLineOptions two = createWithPrefix(WATCHOS_CPUS_PREFIX, "bar", "foo");
    assertSame(one, two);
  }

  @Test
  public void testWatchosCpus_duplicates() throws Exception {
    AppleCommandLineOptions one = createWithPrefix(WATCHOS_CPUS_PREFIX, "foo", "foo");
    AppleCommandLineOptions two = createWithPrefix(WATCHOS_CPUS_PREFIX, "foo");
    assertSame(one, two);
  }

  @Test
  public void testMacosCpus_ordering() throws Exception {
    AppleCommandLineOptions one = createWithPrefix(MACOS_CPUS_PREFIX, "foo", "bar");
    AppleCommandLineOptions two = createWithPrefix(MACOS_CPUS_PREFIX, "bar", "foo");
    assertSame(one, two);
  }

  @Test
  public void testMacosCpus_duplicates() throws Exception {
    AppleCommandLineOptions one = createWithPrefix(MACOS_CPUS_PREFIX, "foo", "foo");
    AppleCommandLineOptions two = createWithPrefix(MACOS_CPUS_PREFIX, "foo");
    assertSame(one, two);
  }

  @Test
  public void testTvosCpus_ordering() throws Exception {
    AppleCommandLineOptions one = createWithPrefix(TVOS_CPUS_PREFIX, "foo", "bar");
    AppleCommandLineOptions two = createWithPrefix(TVOS_CPUS_PREFIX, "bar", "foo");
    assertSame(one, two);
  }

  @Test
  public void testTvosCpus_duplicates() throws Exception {
    AppleCommandLineOptions one = createWithPrefix(TVOS_CPUS_PREFIX, "foo", "foo");
    AppleCommandLineOptions two = createWithPrefix(TVOS_CPUS_PREFIX, "foo");
    assertSame(one, two);
  }

  @Test
  public void testCatalystCpus_ordering() throws Exception {
    AppleCommandLineOptions one = createWithPrefix(CATALYST_CPUS_PREFIX, "foo", "bar");
    AppleCommandLineOptions two = createWithPrefix(CATALYST_CPUS_PREFIX, "bar", "foo");
    assertSame(one, two);
  }

  @Test
  public void testCatalystCpus_duplicates() throws Exception {
    AppleCommandLineOptions one = createWithPrefix(CATALYST_CPUS_PREFIX, "foo", "foo");
    AppleCommandLineOptions two = createWithPrefix(CATALYST_CPUS_PREFIX, "foo");
    assertSame(one, two);
  }
}
