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
package com.google.devtools.build.lib.analysis.config;

import com.google.devtools.build.lib.analysis.util.OptionsTestCase;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** A test for {@link CoreOptions}. */
@RunWith(JUnit4.class)
public final class CoreOptionsTest extends OptionsTestCase<CoreOptions> {

  private static final String FEATURES_PREFIX = "--features=";
  private static final String DEFINE_PREFIX = "--define=";
  private static final String FLAG_ALIAS_PREFIX = "--flag_alias=";

  @Test
  public void testFeatures_orderingOfPositiveFeatures() throws Exception {
    CoreOptions one = createWithPrefix(FEATURES_PREFIX, "foo", "bar");
    CoreOptions two = createWithPrefix(FEATURES_PREFIX, "bar", "foo");
    assertSame(one, two);
  }

  @Test
  public void testFeatures_duplicateFeatures() throws Exception {
    CoreOptions one = createWithPrefix(FEATURES_PREFIX, "foo", "bar");
    CoreOptions two = createWithPrefix(FEATURES_PREFIX, "bar", "foo", "bar");
    assertSame(one, two);
  }

  @Test
  public void testFeatures_disablingWins() throws Exception {
    CoreOptions one = createWithPrefix(FEATURES_PREFIX, "foo", "-foo", "bar");
    CoreOptions two = createWithPrefix(FEATURES_PREFIX, "-foo", "bar");
    assertSame(one, two);
  }

  @Test
  public void testDefines_duplicateKey() throws Exception {
    // Last one should win
    CoreOptions one = createWithPrefix(DEFINE_PREFIX, "a=1", "a=2");
    CoreOptions two = createWithPrefix(DEFINE_PREFIX, "a=2");
    assertSame(one, two);
  }

  @Test
  public void testDefines_orderOfKeys() throws Exception {
    CoreOptions one = createWithPrefix(DEFINE_PREFIX, "a=1", "c=3", "b=2");
    CoreOptions two = createWithPrefix(DEFINE_PREFIX, "b=2", "a=1", "c=3");
    assertSame(one, two);
  }

  @Test
  public void testflagAlias_duplicateKey() throws Exception {
    // Last one should win
    CoreOptions one = createWithPrefix(FLAG_ALIAS_PREFIX, "a=//one", "a=//two");
    CoreOptions two = createWithPrefix(FLAG_ALIAS_PREFIX, "a=//two");
    assertSame(one, two);
  }

  @Test
  public void testFlagAlias_orderOfKeys() throws Exception {
    CoreOptions one = createWithPrefix(FLAG_ALIAS_PREFIX, "a=//one", "c=//three", "b=//two");
    CoreOptions two = createWithPrefix(FLAG_ALIAS_PREFIX, "b=//two", "a=//one", "c=//three");
    assertSame(one, two);
  }

  @Override
  protected Class<CoreOptions> getOptionsClass() {
    return CoreOptions.class;
  }
}
