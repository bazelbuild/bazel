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

package com.google.devtools.build.lib.rules.android;

import com.google.devtools.build.lib.analysis.util.OptionsTestCase;
import com.google.devtools.build.lib.rules.android.AndroidConfiguration.Options;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class AndroidConfigurationTest extends OptionsTestCase<Options> {

  private static final String ANDROID_PLATFORMS_PREFIX = "--android_platforms=";

  @Override
  protected Class<Options> getOptionsClass() {
    return Options.class;
  }

  @Test
  public void testPlatforms_ordering() throws Exception {
    // Order matters.
    Options one = createWithPrefix(ANDROID_PLATFORMS_PREFIX, "//a:one,//b");
    Options two = createWithPrefix(ANDROID_PLATFORMS_PREFIX, "//b,//a:one");
    assertDifferent(one, two);
  }

  @Test
  public void testPlatforms_duplicates() throws Exception {
    // If there are two copies, only the first one is kept.
    Options one = createWithPrefix(ANDROID_PLATFORMS_PREFIX, "//a:a,//b,//a");
    Options two = createWithPrefix(ANDROID_PLATFORMS_PREFIX, "//a,//b");
    assertSame(one, two);
  }
}
