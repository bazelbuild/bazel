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
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.io.Files;
import com.google.devtools.build.lib.sandbox.cgroups.controller.Controller.Memory;
import com.google.devtools.build.lib.sandbox.cgroups.controller.v1.LegacyMemory;
import com.google.devtools.build.lib.sandbox.cgroups.controller.v2.UnifiedMemory;
import com.google.devtools.build.lib.vfs.util.FsApparatus;
import java.io.File;
import java.io.IOException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class MemoryTest {
  private final FsApparatus scratch = FsApparatus.newNative();

  @Test
  public void setMemoryLimit_v1() throws IOException {
    File limit = scratch.file("cgroup/memory/memory.limit_in_bytes", "0").getPathFile();
    Memory memory = new LegacyMemory(scratch.path("cgroup/memory").getPathFile().toPath());
    memory.setMaxBytes(1000);
    assertThat(Files.asCharSource(limit, UTF_8).read()).isEqualTo("1000");
  }

  @Test
  public void getMemoryLimit_v1() throws IOException {
    scratch.file("cgroup/memory/memory.limit_in_bytes", "100");
    Memory memory = new LegacyMemory(scratch.path("cgroup/memory").getPathFile().toPath());
    assertThat(memory.getMaxBytes()).isEqualTo(100);
  }

  @Test
  public void getMemoryUsage_v1() throws IOException {
    scratch.file("cgroup/memory/memory.usage_in_bytes", "2000");
    Memory memory = new LegacyMemory(scratch.path("cgroup/memory").getPathFile().toPath());
    assertThat(memory.getUsageInBytes()).isEqualTo(2000);
  }

  @Test
  public void setMemoryLimit_v2() throws IOException {
    File limit = scratch.file("cgroup/memory/memory.max", "0").getPathFile();
    Memory memory = new UnifiedMemory(scratch.path("cgroup/memory").getPathFile().toPath());
    memory.setMaxBytes(1000);
    assertThat(Files.asCharSource(limit, UTF_8).read()).isEqualTo("1000");
  }

  @Test
  public void getMemoryLimit_v2() throws IOException {
    scratch.file("cgroup/memory/memory.max", "100");
    Memory memory = new UnifiedMemory(scratch.path("cgroup/memory").getPathFile().toPath());
    assertThat(memory.getMaxBytes()).isEqualTo(100);
  }

  @Test
  public void getMemoryUsage_v2() throws IOException {
    scratch.file("cgroup/memory/memory.current", "2000");
    Memory memory = new UnifiedMemory(scratch.path("cgroup/memory").getPathFile().toPath());
    assertThat(memory.getUsageInBytes()).isEqualTo(2000);
  }
}
