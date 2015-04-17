// Copyright 2009-2015 Google Inc. All rights reserved.
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

import static org.junit.Assert.assertEquals;

import com.google.common.collect.ImmutableList;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * A test for {@link BuildOptions}.
 */
@RunWith(JUnit4.class)
public class BuildOptionsTest {
  private static final ImmutableList<Class<? extends FragmentOptions>> TEST_OPTIONS =
      ImmutableList.<Class<? extends FragmentOptions>>of(BuildConfiguration.Options.class);

  @Test
  public void testOptionSetCaching() throws Exception {
    BuildOptions a = BuildOptions.createDefaults(TEST_OPTIONS);
    BuildOptions b = BuildOptions.createDefaults(TEST_OPTIONS);
    // The cache keys of the OptionSets must be equal even if these are
    // different objects, if they were created with the same options (no options in this case).
    assertEquals(a.toString(), b.toString());
    assertEquals(a.computeCacheKey(), b.computeCacheKey());
  }

  @Test
  public void testCachingSpecialCases() throws Exception {
    // You can add options here to test that their string representations are good.
    String[] options = new String[] { "--run_under=//run_under" };
    BuildOptions a = BuildOptions.of(TEST_OPTIONS, options);
    BuildOptions b = BuildOptions.of(TEST_OPTIONS, options);
    assertEquals(a.toString(), b.toString());
  }
}
