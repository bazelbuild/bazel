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
//

package com.google.devtools.build.lib.bazel.bzlmod;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;
import static org.junit.Assume.assumeTrue;

import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.net.URISyntaxException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link RegistryFunction}. */
@RunWith(JUnit4.class)
public class RegistryFunctionTest {

  @Test
  public void getWatchedRegistryPath_absoluteFileUrl() throws Exception {
    assumeTrue(OS.getCurrent() != OS.WINDOWS);
    assertThat(RegistryFunction.getWatchedRegistryPath("file:///path/to/registry"))
        .isEqualTo(PathFragment.create("/path/to/registry"));
  }

  @Test
  public void getWatchedRegistryPath_rejectsNonFileScheme() {
    var e =
        assertThrows(
            URISyntaxException.class,
            () -> RegistryFunction.getWatchedRegistryPath("https://bcr.bazel.build"));
    assertThat(e).hasMessageThat().contains("Only file:// registries can be watched");
  }
}
