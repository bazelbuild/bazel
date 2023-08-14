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
package com.google.devtools.build.lib.sandbox;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.io.Files;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.util.FsApparatus;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link CgroupsInfo}. */
@RunWith(JUnit4.class)
public class CgroupsInfoTest {
  private final FsApparatus scratch = FsApparatus.newNative();
  /** We use this pseudo-root to get around not being able to replace absolute paths. */
  private String root;

  @Before
  public void setUp() throws IOException {
    root = scratch.dir("fake_root").getPathString();
  }

  @Test
  public void testGetCgroupMountInfo_v1() throws IOException {
    String pathString =
        createFakeAbsoluteFile(
            "/proc/self/mounts",
            "sysfs /sys sysfs rw,nosuid,nodev,noexec,relatime 0 0",
            "cgroup /dev/cgroup/cpu cgroup rw,cpu,cpuacct 0 0",
            "cgroup /dev/cgroup/io cgroup rw,io 0 0",
            "cgroup /dev/cgroup/job cgroup rw,job 0 0",
            "cgroup /dev/cgroup/memory cgroup rw,memory,hugetlb 0 0",
            "proc /proc proc rw,nosuid,nodev,noexec,relatime 0 0");
    Pair<File, Boolean> cgroupsMountInfo = CgroupsInfo.getMemoryCgroupInfo(new File(pathString));
    assertThat(cgroupsMountInfo.second).isFalse();
    assertThat(cgroupsMountInfo.first.toString()).isEqualTo("/dev/cgroup/memory");
  }

  @Test
  public void testGetCgroupMountInfo_v2() throws IOException {
    String pathString =
        createFakeAbsoluteFile(
            "/proc/self/mounts",
            "sysfs /sys sysfs rw,nosuid,nodev,noexec,relatime 0 0",
            "cgroup2 /sys/fs/cgroup cgroup2 rw,memory_recursiveprot 0 0",
            "proc /proc proc rw,nosuid,nodev,noexec,relatime 0 0");
    Pair<File, Boolean> cgroupsMountInfo = CgroupsInfo.getMemoryCgroupInfo(new File(pathString));
    assertThat(cgroupsMountInfo.second).isTrue();
    assertThat(cgroupsMountInfo.first.toString()).isEqualTo("/sys/fs/cgroup");
  }

  @Test
  public void testGetCgroupMountInfo_mixed_v1_has_memory() throws IOException {
    String pathString =
        createFakeAbsoluteFile(
            "/proc/self/mounts",
            "sysfs /sys sysfs rw,nosuid,nodev,noexec,relatime 0 0",
            "cgroup2 /sys/fs/cgroup cgroup2 rw,memory_recursiveprot 0 0",
            "cgroup /dev/cgroup/job cgroup rw,job 0 0",
            "cgroup /dev/cgroup/memory cgroup rw,memory,hugetlb 0 0",
            "proc /proc proc rw,nosuid,nodev,noexec,relatime 0 0");
    Pair<File, Boolean> cgroupsMountInfo = CgroupsInfo.getMemoryCgroupInfo(new File(pathString));
    assertThat(cgroupsMountInfo.second).isFalse();
    assertThat(cgroupsMountInfo.first.toString()).isEqualTo("/dev/cgroup/memory");
  }

  @Test
  public void testGetCgroupMountInfo_mixed_v2_has_memory() throws IOException {
    String pathString =
        createFakeAbsoluteFile(
            "/proc/self/mount",
            "sysfs /sys sysfs rw,nosuid,nodev,noexec,relatime 0 0",
            "cgroup2 /sys/fs/cgroup cgroup2 rw,memory_recursiveprot 0 0",
            "cgroup /dev/cgroup/job cgroup rw,job 0 0",
            "cgroup /dev/cgroup/io cgroup rw,io 0 0",
            "proc /proc proc rw,nosuid,nodev,noexec,relatime 0 0");
    Pair<File, Boolean> cgroupsMountInfo = CgroupsInfo.getMemoryCgroupInfo(new File(pathString));
    assertThat(cgroupsMountInfo.second).isTrue();
    assertThat(cgroupsMountInfo.first.toString()).isEqualTo("/sys/fs/cgroup");
  }

  @Test
  public void testGetCgroupsHierarchy_v1() throws IOException {
    String pathString =
        createFakeAbsoluteFile(
            "/proc/self/cgroup",
            "8:net:/netdir/action-6",
            "7:memory,hugetlb:/memdir/action-6",
            "6:job:/jobdir/action-16",
            "5:io:/iodir/action-1");
    int hierarchy = CgroupsInfo.getMemoryHierarchy(new File(pathString));
    assertThat(hierarchy).isEqualTo(7);
  }

  @Test
  public void testGetCgroupsNode_v1() throws IOException {
    String mountPath = root + "/dev/cgroup/memory";
    String pathString =
        createFakeAbsoluteFile(
            "/proc/self/cgroup",
            "8:net:/netdir/action-6",
            "7:memory,hugetlb:/memdir/action-6",
            "6:job:/jobdir/action-16",
            "5:io:/iodir/action-1");
    scratch.dir(root + "/dev/cgroup/memory/memdir/action-6").createDirectoryAndParents();
    File cgroupsMountInfo = CgroupsInfo.getBlazeMemoryCgroup(new File(mountPath), 7, pathString);
    assertThat(cgroupsMountInfo.getAbsolutePath()).isEqualTo(mountPath + "/memdir/action-6");
  }

  @Test
  public void testGetCgroupsNode_v2() throws IOException {
    String mountPath = root + "/sys/fs/cgroup";
    String procSelfCgroup =
        createFakeAbsoluteFile("/proc/self/cgroup", "0::/user.slice/session.scope");
    scratch.dir(mountPath + "/user.slice/session.scope").createDirectoryAndParents();
    File cgroupsMountInfo =
        CgroupsInfo.getBlazeMemoryCgroup(new File(mountPath), 0, procSelfCgroup);
    assertThat(cgroupsMountInfo.getAbsolutePath())
        .isEqualTo(mountPath + "/user.slice/session.scope");
  }

  @Test
  public void testCreate_v1() throws IOException {
    // We actually use the paths from /proc/mount here, so they must include the fake root.
    String procMount =
        createFakeAbsoluteFile(
            "/proc/self/mounts",
            "sysfs " + root + "/sys sysfs rw,nosuid,nodev,noexec,relatime 0 0",
            "cgroup " + root + "/dev/cgroup/cpu cgroup rw,cpu,cpuacct 0 0",
            "cgroup " + root + "/dev/cgroup/io cgroup rw,io 0 0",
            "cgroup " + root + "/dev/cgroup/job cgroup rw,job 0 0",
            "cgroup " + root + "/dev/cgroup/memory cgroup rw,memory,hugetlb 0 0",
            "proc " + root + "/proc proc rw,nosuid,nodev,noexec,relatime 0 0");
    String procSelfCgroup =
        createFakeAbsoluteFile(
            "/proc/self/cgroup",
            "8:net:/netdir/action-6",
            "7:memory,hugetlb:/memdir/action-6",
            "6:job:/jobdir/action-16",
            "5:io:/iodir/action-1");
    scratch.dir(root + "/dev/cgroup/memory/memdir/action-6").createDirectoryAndParents();
    CgroupsInfo cgroupsInfo = CgroupsInfo.create(procMount, procSelfCgroup);
    assertThat(cgroupsInfo.isCgroupsV2()).isFalse();
    assertThat(cgroupsInfo.getBlazeDir().getAbsolutePath())
        .matches(root + "/dev/cgroup/memory/memdir/action-6/blaze_\\d+_spawns");
    assertThat(cgroupsInfo.getBlazeDir().exists()).isTrue();
  }

  @Test
  public void testCreate_v2() throws IOException {
    String cgroupsRoot = root + "/sys/fs/cgroup";
    // We actually use the paths from /proc/mount here, so they must include the fake root.
    String procMount =
        createFakeAbsoluteFile(
            "/proc/mount",
            "sysfs " + root + "/sys sysfs rw,nosuid,nodev,noexec,relatime 0 0",
            "cgroup2 " + cgroupsRoot + " cgroup2 rw,memory_recursiveprot 0 0",
            "proc " + root + "/proc proc rw,nosuid,nodev,noexec,relatime 0 0");
    String procSelfCgroup =
        createFakeAbsoluteFile("/proc/self/cgroup", "0::/user.slice/session.scope");

    // We base the new subtree off the parent of the current scope.
    String userSlice = root + "/sys/fs/cgroup/user.slice";
    // Even though we create a separate directory off `user.slice`, we must check that the
    // scope for the current process is writable.
    scratch.dir(userSlice + "/session.scope").createDirectoryAndParents();

    String blazeSlice = userSlice + "/blaze_" + ProcessHandle.current().pid() + "_spawns.slice";
    scratch.dir(blazeSlice).createDirectoryAndParents();
    scratch.file(blazeSlice + "/cgroup.controllers", "memory pids");
    // Since this controllers file is missing `pids`, we expect that to be written to it.
    scratch.file(blazeSlice + "/cgroup.subtree_control", "memory");

    CgroupsInfo cgroupsInfo = CgroupsInfo.create(procMount, procSelfCgroup);

    assertThat(cgroupsInfo.isCgroupsV2()).isTrue();
    assertThat(cgroupsInfo.getBlazeDir().getAbsolutePath())
        .matches(root + "/sys/fs/cgroup/user.slice/blaze_\\d+_spawns.slice");

    // This is not what an actual cgroupsv2 file would contain, but it's what we expect to write to
    // it to enable subtree control.
    assertThat(
            Files.readLines(
                new File(blazeSlice + "/cgroup.subtree_control"), StandardCharsets.UTF_8))
        .containsExactly("+memory +pids");
  }

  private String createFakeAbsoluteFile(String fileName, String... contents) throws IOException {
    return scratch.file(root + fileName, contents).getPathString();
  }
}
