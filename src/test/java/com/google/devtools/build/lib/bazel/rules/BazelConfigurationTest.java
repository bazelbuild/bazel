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

package com.google.devtools.build.lib.bazel.rules;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.bazel.rules.BazelConfiguration.pathOrDefault;

import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.common.options.Options;
import java.util.HashMap;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link BazelConfiguration}.
 */
@RunWith(JUnit4.class)
public class BazelConfigurationTest {
  @Test
  public void getShellExecutableUnset() {
    BazelConfiguration.Options o = Options.getDefaults(BazelConfiguration.Options.class);
    assertThat(new BazelConfiguration(OS.LINUX, o).getShellExecutable())
        .isEqualTo(PathFragment.create("/bin/bash"));
    assertThat(new BazelConfiguration(OS.FREEBSD, o).getShellExecutable())
        .isEqualTo(PathFragment.create("/usr/local/bin/bash"));
    assertThat(new BazelConfiguration(OS.WINDOWS, o).getShellExecutable())
        .isEqualTo(PathFragment.create("c:/tools/msys64/usr/bin/bash.exe"));
  }

  @Test
  public void getShellExecutableIfSet() {
    BazelConfiguration.Options o = Options.getDefaults(BazelConfiguration.Options.class);
    o.shellExecutable = PathFragment.create("/bin/bash");
    assertThat(new BazelConfiguration(OS.LINUX, o).getShellExecutable())
        .isEqualTo(PathFragment.create("/bin/bash"));
    assertThat(new BazelConfiguration(OS.FREEBSD, o).getShellExecutable())
        .isEqualTo(PathFragment.create("/bin/bash"));
    assertThat(new BazelConfiguration(OS.WINDOWS, o).getShellExecutable())
        .isEqualTo(PathFragment.create("/bin/bash"));
  }

  @Test
  public void strictActionEnv() {
    BazelConfiguration.Options options = Options.getDefaults(BazelConfiguration.Options.class);
    options.useStrictActionEnv = true;
    BazelConfiguration config = new BazelConfiguration(OS.LINUX, options);
    Map<String, String> env = new HashMap<>();
    config.setupActionEnvironment(env);
    assertThat(env).containsEntry("PATH", "/bin:/usr/bin");
  }

  @Test
  public void pathOrDefaultOnLinux() {
    assertThat(pathOrDefault(OS.LINUX, null, null)).isEqualTo("/bin:/usr/bin");
    assertThat(pathOrDefault(OS.LINUX, "/not/bin", null)).isEqualTo("/not/bin");
  }

  @Test
  public void pathOrDefaultOnWindows() {
    assertThat(pathOrDefault(OS.WINDOWS, null, null)).isEqualTo("");
    assertThat(pathOrDefault(OS.WINDOWS, "C:/mypath", null))
        .isEqualTo("C:/mypath");
    assertThat(pathOrDefault(OS.WINDOWS, "C:/mypath", PathFragment.create("D:/foo/shell")))
        .isEqualTo("D:\\foo;C:/mypath");
  }

  @Test
  public void optionsAlsoApplyToHost() {
    BazelConfiguration.Options o = Options.getDefaults(BazelConfiguration.Options.class);
    o.shellExecutable = PathFragment.create("/my/shell/binary");
    o.useStrictActionEnv = true;
    BazelConfiguration.Options h = o.getHost();
    assertThat(h.shellExecutable).isEqualTo(PathFragment.create("/my/shell/binary"));
    assertThat(h.useStrictActionEnv).isTrue();
  }
}
