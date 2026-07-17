// Copyright 2024 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.sandbox.cgroups;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.vfs.util.FsApparatus;
import java.io.IOException;
import java.nio.file.Path;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class MountTest {
  private final FsApparatus scratch = FsApparatus.newNative();

  @Test
  public void testParse() throws IOException {
    ImmutableList<Mount> mounts =
        Mount.parse(
            scratch
                .file(
                    "proc/self/mounts",
                    "sysfs /sys sysfs rw,nosuid,nodev,noexec,relatime 0 0",
                    "cgroup /dev/cgroup/cpu cgroup rw,cpu,cpuacct 0 0",
                    "cgroup /dev/cgroup/unified cgroup2 ro 0 0",
                    "proc /proc proc rw,nosuid,nodev,noexec,relatime 0 0")
                .getPathFile());

    assertThat(mounts).hasSize(2);
    assertThat(mounts.get(0).isV2()).isFalse();
    assertThat(mounts.get(1).isV2()).isTrue();
    assertThat(mounts.get(0).opts()).contains("cpu");
    assertThat(mounts.get(0).path()).isEqualTo(Path.of("/dev/cgroup/cpu"));
    assertThat(mounts.get(1).path()).isEqualTo(Path.of("/dev/cgroup/unified"));
  }
}
