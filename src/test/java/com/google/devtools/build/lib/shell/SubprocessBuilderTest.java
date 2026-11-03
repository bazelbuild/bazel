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

package com.google.devtools.build.lib.shell;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.util.StringEncoding.internalToPlatform;
import static com.google.devtools.build.lib.util.StringEncoding.platformToInternal;
import static com.google.devtools.build.lib.util.StringEncoding.unicodeToInternal;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.testutil.TestUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.File;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link SubprocessBuilder}. */
@RunWith(JUnit4.class)
public class SubprocessBuilderTest {

  @Test
  public void testMakeArgv0AbsoluteWithNonAsciiWorkingDirectory() {
    PathFragment workingDirectory =
        PathFragment.create(platformToInternal(TestUtils.tmpDir()))
            .getRelative(unicodeToInternal("jörg"));

    ImmutableList<String> argv =
        SubprocessBuilder.makeArgv0Absolute(
            ImmutableList.of("bin/tool", "arg"),
            new File(internalToPlatform(workingDirectory.getPathString())));

    assertThat(PathFragment.create(argv.get(0)))
        .isEqualTo(workingDirectory.getRelative("bin/tool"));
    assertThat(argv.get(1)).isEqualTo("arg");
  }
}
