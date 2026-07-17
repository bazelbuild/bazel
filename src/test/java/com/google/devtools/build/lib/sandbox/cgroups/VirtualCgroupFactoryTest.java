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
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.vfs.util.FsApparatus;
import java.io.IOException;
import java.nio.file.Path;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class VirtualCgroupFactoryTest {
  private final FsApparatus scratch = FsApparatus.newNative();

  private VirtualCgroup root;

  @Before
  public void setup() throws Exception {
    scratch.dir("cpu/cpu");
    scratch.dir("mem/mem");
    ImmutableList<Mount> mounts =
        ImmutableList.of(
            Mount.create(
                scratch.path("cpu").getPathFile().toPath(), "cgroup", ImmutableList.of("cpu")),
            Mount.create(
                scratch.path("mem").getPathFile().toPath(), "cgroup", ImmutableList.of("memory")));
    ImmutableList<Hierarchy> hierarchies =
        ImmutableList.of(
            Hierarchy.create(
                1, ImmutableList.of("cpu"), scratch.path("/cpu").getPathFile().toPath()),
            Hierarchy.create(
                2, ImmutableList.of("memory"), scratch.path("/mem").getPathFile().toPath()));

    root = VirtualCgroup.createRoot(mounts, hierarchies);
    assertThat(root.cpu()).isNotNull();
    assertThat(root.memory()).isNotNull();
  }

  @Test
  public void testCreateNoLimits() throws IOException {
    ImmutableMap<String, Double> defaults = ImmutableMap.of();
    VirtualCgroupFactory factory = new VirtualCgroupFactory("nolimits", root, defaults, false);

    VirtualCgroup vcg = factory.create(1, ImmutableMap.of());

    assertThat(vcg.paths()).isEmpty();
  }

  @Test
  public void testForceCreateNoLimits() throws IOException {
    ImmutableMap<String, Double> defaults = ImmutableMap.of();
    VirtualCgroupFactory factory = new VirtualCgroupFactory("nolimits", root, defaults, true);

    VirtualCgroup vcg = factory.create(1, ImmutableMap.of());

    assertThat(vcg.paths()).isNotEmpty();
    assertThat(vcg.cpu()).isNotNull();
    assertThat(vcg.memory()).isNotNull();
  }

  @Test
  public void testCreateWithDefaultLimits() throws IOException {
    ImmutableMap<String, Double> defaults = ImmutableMap.of("memory", 100.0);
    VirtualCgroupFactory factory = new VirtualCgroupFactory("defaults", root, defaults, false);

    VirtualCgroup vcg = factory.create(1, ImmutableMap.of());

    assertThat(vcg.paths()).isNotEmpty();
    assertThat(vcg.cpu()).isNotNull();
    assertThat(vcg.memory()).isNotNull();
    assertThat(vcg.memory().getMaxBytes()).isEqualTo(100 * 1024 * 1024);
  }

  @Test
  public void testCreateWithCustomLimits() throws IOException {
    scratch.file("cpu/cpu/custom1.scope/cpu.cfs_period_us", "1000");
    ImmutableMap<String, Double> defaults = ImmutableMap.of("memory", 100.0, "cpu", 1.0);
    VirtualCgroupFactory factory = new VirtualCgroupFactory("custom", root, defaults, false);

    VirtualCgroup vcg = factory.create(1, ImmutableMap.of("memory", 200.0));

    assertThat(vcg.paths()).isNotEmpty();
    assertThat(vcg.cpu()).isNotNull();
    assertThat(vcg.memory()).isNotNull();
    assertThat(vcg.cpu().getCpus()).isEqualTo(1);
    assertThat(vcg.memory().getMaxBytes()).isEqualTo(200 * 1024 * 1024);
  }

  @Test
  public void testCreateNull() throws IOException {
    ImmutableMap<String, Double> defaults = ImmutableMap.of("memory", 100.0, "cpu", 1.0);
    VirtualCgroupFactory factory =
        new VirtualCgroupFactory("null", VirtualCgroup.NULL, defaults, false);

    VirtualCgroup vcg = factory.create(1, ImmutableMap.of());

    assertThat(vcg.paths()).isEmpty();
    assertThat(vcg.cpu()).isNull();
    assertThat(vcg.memory()).isNull();
  }

  @Test
  public void testGet() throws IOException {
    ImmutableMap<String, Double> defaults = ImmutableMap.of("memory", 100.0);
    VirtualCgroupFactory factory = new VirtualCgroupFactory("get", root, defaults, false);

    VirtualCgroup vcg = factory.create(1, ImmutableMap.of());

    assertThat(factory.get(1)).isEqualTo(vcg);
  }

  @Test
  public void testRemove() throws IOException {
    ImmutableMap<String, Double> defaults = ImmutableMap.of();
    VirtualCgroupFactory factory = new VirtualCgroupFactory("get", root, defaults, true);

    VirtualCgroup vcg = factory.create(1, ImmutableMap.of());

    assertThat(factory.remove(1)).isEqualTo(vcg);
    for (Path p : vcg.paths()) {
      assertThat(p.toFile().exists()).isFalse();
    }
  }
}
