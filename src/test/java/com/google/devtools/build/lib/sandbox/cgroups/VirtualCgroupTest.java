// Copyright 2022 The Bazel Authors. All rights reserved.
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
import com.google.devtools.build.lib.vfs.util.FsApparatus;
import java.io.File;
import java.io.IOException;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class VirtualCgroupTest {
  private final FsApparatus scratch = FsApparatus.newNative();

  private void updateV1(String path) throws IOException {
    scratch.file(path + "/cgroup.procs");
  }

  private VirtualCGroup createV1() throws IOException {
    File mounts = scratch.file(
        "proc/self/mounts",
        "sysfs /sys sysfs rw,nosuid,nodev,noexec,relatime 0 0",
        String.format("cgroup %s/dev/cgroup/memory cgroup rw,memory,hugetlb 0 0", scratch.path("")),
        String.format("cgroup %s/dev/cgroup/cpuset cgroup rw,cpuset 0 0", scratch.path("")),
        String.format("cgroup %s/dev/cgroup/cpu cgroup rw,cpu,cpuacct 0 0", scratch.path("")),
        String.format("cgroup %s/dev/cgroup/blkio cgroup rw,blkio 0 0", scratch.path("")),
        "proc /proc proc rw,nosuid,nodev,noexec,relatime 0 0").getPathFile();
    File hierarchies = scratch.file(
        "proc/self/cgroup",
        "4:memory,hugetlb:/user.slice",
        "3:cpuset:/",
        "2:cpu,cpuacct:/user.slice",
        "1:blkio:/user.slice").getPathFile();
    updateV1("dev/cgroup/memory/user.slice");
    updateV1("dev/cgroup/cpuset");
    updateV1("dev/cgroup/cpu/user.slice");
    updateV1("dev/cgroup/blkio/user.slice");
    return VirtualCGroup.create(mounts, hierarchies);
  }

  private void updateV2(String path, String controllers) throws IOException {
    scratch.file(path + "/cgroup.procs");
    scratch.file(path + "/cgroup.controllers", controllers);
    scratch.file(path + "/cgroup.subtree_control");
  }

  private VirtualCGroup createV2() throws IOException {
    File mounts = scratch.file(
        "proc/self/mounts",
        "sysfs /sys sysfs rw,nosuid,nodev,noexec,relatime 0 0",
        String.format("cgroup %s/dev/cgroup/unified cgroup2 ro 0 0", scratch.path("")),
        "proc /proc proc rw,nosuid,nodev,noexec,relatime 0 0").getPathFile();
    File hierarchies = scratch.file(
        "proc/self/cgroup",
        "0::/user.slice").getPathFile();
    updateV2("dev/cgroup/unified", "memory cpu");
    updateV2("dev/cgroup/unified/user.slice", "memory cpu");
    return VirtualCGroup.create(mounts, hierarchies);
  }

  private VirtualCGroup createHybrid() throws IOException {
    File mounts = scratch.file(
        "proc/self/mounts",
        "sysfs /sys sysfs rw,nosuid,nodev,noexec,relatime 0 0",
        String.format("cgroup %s/dev/cgroup/cpuset cgroup rw,cpuset 0 0", scratch.path("")),
        String.format("cgroup %s/dev/cgroup/cpu cgroup rw,cpu,cpuacct 0 0", scratch.path("")),
        String.format("cgroup %s/dev/cgroup/blkio cgroup rw,blkio 0 0", scratch.path("")),
        String.format("cgroup %s/dev/cgroup/unified cgroup2 ro 0 0", scratch.path("")),
        "proc /proc proc rw,nosuid,nodev,noexec,relatime 0 0").getPathFile();
    File hierarchies = scratch.file(
        "proc/self/cgroup",
        "3:cpuset:/",
        "2:cpu,cpuacct:/user.slice",
        "1:blkio:/user.slice",
        "0::/user.slice").getPathFile();
    updateV1("dev/cgroup/cpuset");
    updateV1("dev/cgroup/cpu/user.slice");
    updateV1("dev/cgroup/blkio/user.slice");

    updateV2("dev/cgroup/unified", "memory pids");
    updateV2("dev/cgroup/unified/user.slice", "memory pids");
    return VirtualCGroup.create(mounts, hierarchies);
  }

  @Test
  public void testGetRootCgroup_v1() throws IOException {
    VirtualCGroup vcg = createV1();
    assertThat(vcg.cpu()).isNotNull();
    assertThat(vcg.memory()).isNotNull();
    assertThat(vcg.cpu().isLegacy()).isTrue();
    assertThat(vcg.memory().isLegacy()).isTrue();
    assertThat(vcg.cpu().getPath())
        .isEqualTo(scratch.path("dev/cgroup/cpu/user.slice").getPathFile().toPath());
    assertThat(vcg.memory().getPath())
        .isEqualTo(scratch.path("dev/cgroup/memory/user.slice").getPathFile().toPath());
  }

  @Test
  public void testGetRootCgroup_v2() throws IOException {
    VirtualCGroup vcg = createV2();
    assertThat(vcg.cpu()).isNotNull();
    assertThat(vcg.memory()).isNotNull();
    assertThat(vcg.cpu().isLegacy()).isFalse();
    assertThat(vcg.memory().isLegacy()).isFalse();
    assertThat(vcg.cpu().getPath())
        .isEqualTo(scratch.path("dev/cgroup/unified").getPathFile().toPath());
    assertThat(vcg.memory().getPath())
        .isEqualTo(scratch.path("dev/cgroup/unified").getPathFile().toPath());
  }

  @Test
  public void testGetRootCgroup_mixed() throws IOException {
    VirtualCGroup vcg = createHybrid();
    assertThat(vcg.cpu()).isNotNull();
    assertThat(vcg.memory()).isNotNull();
    assertThat(vcg.cpu().isLegacy()).isTrue();
    assertThat(vcg.memory().isLegacy()).isFalse();
    assertThat(vcg.cpu().getPath())
        .isEqualTo(scratch.path("dev/cgroup/cpu/user.slice").getPathFile().toPath());
    assertThat(vcg.memory().getPath())
            .isEqualTo(scratch.path("dev/cgroup/unified").getPathFile().toPath());
  }

  @Test
  public void testCreateChild() throws IOException {
    VirtualCGroup vcg = createHybrid();
    VirtualCGroup child = vcg.child("foo");
    assertThat(child.cpu()).isNotNull();
    assertThat(child.memory()).isNotNull();
    assertThat(child.memory().getPath())
        .isEqualTo(scratch.path("dev/cgroup/unified/foo").getPathFile().toPath());
    assertThat(child.cpu().getPath())
        .isEqualTo(scratch.path("dev/cgroup/cpu/user.slice/foo").getPathFile().toPath());
    File subtree = vcg.memory().getPath().resolve("cgroup.subtree_control").toFile();
    assertThat(Files.asCharSource(subtree, UTF_8).read())
        .isEqualTo("+memory +pids ");
  }

  @Test
  public void testNullCgroupCreatesNullChild() throws IOException {
    VirtualCGroup child = VirtualCGroup.NULL.child("foo");
    assertThat(child.cpu()).isNull();
    assertThat(child.cpu()).isNull();
  }

  @Test
  public void testAddProcess_v1() throws IOException {
    VirtualCGroup vcg = createV1();
    vcg.addProcess(1234);
    assertThat(Files.asCharSource(vcg.cpu().getPath().resolve("cgroup.procs").toFile(), UTF_8).read())
        .isEqualTo("1234");
    assertThat(Files.asCharSource(vcg.memory().getPath().resolve("cgroup.procs").toFile(), UTF_8).read())
        .isEqualTo("1234");
  }

  @Test
  public void testAddProcess_v2() throws IOException {
    VirtualCGroup vcg = createV2();
    vcg.addProcess(1234);
    assertThat(Files.asCharSource(vcg.cpu().getPath().resolve("cgroup.procs").toFile(), UTF_8).read())
        .isEqualTo("1234");
    assertThat(Files.asCharSource(vcg.memory().getPath().resolve("cgroup.procs").toFile(), UTF_8).read())
        .isEqualTo("1234");
  }

  @Test
  public void testCgroupInvalidMounts() throws IOException {
    File mounts = scratch.file("proc/self/mounts").getPathFile();
    File hierarchies = scratch.file("proc/self/cgroup", "0::/user.slice").getPathFile();

    VirtualCGroup vcg = VirtualCGroup.create(mounts, hierarchies);
    assertThat(vcg.cpu()).isNull();
    assertThat(vcg.memory()).isNull();
  }

  @Test
  public void testCgroupInvalidHierarchies() throws IOException {
    File mounts = scratch.file(
        "proc/self/mounts",
        String.format("cgroup %s/dev/cgroup/unified cgroup2 ro 0 0", scratch.path(""))).getPathFile();
    File hierarchies = scratch.file("proc/self/cgroup").getPathFile();

    VirtualCGroup vcg = VirtualCGroup.create(mounts, hierarchies);
    assertThat(vcg.cpu()).isNull();
    assertThat(vcg.memory()).isNull();
  }

  @Test
  public void testCgroupOnlyMemory() throws IOException {
    File mounts = scratch.file(
        "proc/self/mounts",
        String.format("cgroup %s/dev/cgroup/memory cgroup rw,memory,hugetlb 0 0", scratch.path("")),
        String.format("cgroup %s/dev/cgroup/cpuset cgroup rw,cpuset 0 0", scratch.path(""))).getPathFile();
    File hierarchies = scratch.file(
        "proc/self/cgroup",
        "2:cpuset:/user.slice",
        "1:memory,hugetlb:/user.slice").getPathFile();
    scratch.file("dev/cgroup/memory/user.slice/cgroup.procs");
    scratch.file("dev/cgroup/cpuset/user.slice/cgroup.procs");

    VirtualCGroup vcg = VirtualCGroup.create(mounts, hierarchies);
    assertThat(vcg.cpu()).isNull();
    assertThat(vcg.memory()).isNotNull();
  }

  @Test
  public void testCgroupOnlyCpu() throws IOException {
    File mounts = scratch.file(
        "proc/self/mounts",
        String.format("cgroup %s/dev/cgroup/cpu cgroup rw,cpu 0 0", scratch.path("")),
        String.format("cgroup %s/dev/cgroup/cpuset cgroup rw,cpuset 0 0", scratch.path(""))).getPathFile();
    File hierarchies = scratch.file(
        "proc/self/cgroup",
        "2:cpuset:/user.slice",
        "1:cpu:/user.slice").getPathFile();
    scratch.file("dev/cgroup/cpu/user.slice/cgroup.procs");
    scratch.file("dev/cgroup/cpuset/user.slice/cgroup.procs");

    VirtualCGroup vcg = VirtualCGroup.create(mounts, hierarchies);
    assertThat(vcg.cpu()).isNotNull();
    assertThat(vcg.memory()).isNull();
  }
}
