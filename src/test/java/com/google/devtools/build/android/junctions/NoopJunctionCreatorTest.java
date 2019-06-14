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
package com.google.devtools.build.android.junctions;

import static com.google.common.truth.Truth.assertThat;

import java.nio.file.FileSystems;
import java.nio.file.Path;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for NoopJunctionCreator. */
@RunWith(JUnit4.class)
public class NoopJunctionCreatorTest {
  private Path tmproot = null;

  @Before
  public void acquireTmpRoot() {
    String tmpEnv = System.getenv("TEST_TMPDIR");
    assertThat(tmpEnv).isNotNull();
    tmproot = FileSystems.getDefault().getPath(tmpEnv);
    // Cast Path to Object to disambiguate which assertThat-overload to use.
    assertThat((Object) tmproot).isNotNull();
  }

  @Test
  public void testNoopJunctionCreator() throws Exception {
    JunctionCreator jc = new NoopJunctionCreator();
    // Cast Path to Object to disambiguate which assertThat-overload to use.
    assertThat((Object) jc.create(null)).isNull();

    Path p = tmproot.resolve("foo");
    // Cast Path to Object to disambiguate which assertThat-overload to use.
    assertThat((Object) jc.create(p)).isSameInstanceAs(p);
  }
}
