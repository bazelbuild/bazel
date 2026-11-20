// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.util;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.util.OptionsUtils.PathFragmentConverter;
import com.google.devtools.build.lib.vfs.PathFragment;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link OptionsUtils} on Windows. */
@RunWith(JUnit4.class)
public class OptionsUtilsWindowsTest {

  private String oldHome;

  @Before
  public void setHome() {
    oldHome = System.getProperty("user.home");
    System.setProperty(
        "user.home",
        StringEncoding.internalToPlatform(StringEncoding.unicodeToInternal("C:\\Users\\jörg")));
  }

  @After
  public void restoreHome() {
    System.setProperty("user.home", oldHome);
  }

  @Test
  public void tildeExpanded() throws Exception {
    assertThat(new PathFragmentConverter().convert("~\\foo\\~bar"))
        .isEqualTo(PathFragment.create(StringEncoding.unicodeToInternal("C:/Users/jörg/foo/~bar")));
    assertThat(new PathFragmentConverter().convert("~/foo/~bar"))
        .isEqualTo(PathFragment.create(StringEncoding.unicodeToInternal("C:/Users/jörg/foo/~bar")));
  }

  @Test
  public void tildeNotExpanded() throws Exception {
    assertThat(new PathFragmentConverter().convert("foo\\~\\bar"))
        .isEqualTo(PathFragment.create("foo/~/bar"));
    assertThat(new PathFragmentConverter().convert("~foo")).isEqualTo(PathFragment.create("~foo"));
  }
}
