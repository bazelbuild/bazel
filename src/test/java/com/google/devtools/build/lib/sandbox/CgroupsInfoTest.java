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
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.io.Files;
import com.google.devtools.build.lib.sandbox.CgroupsInfo.InvalidCgroupsInfo;
import com.google.devtools.build.lib.vfs.util.FsApparatus;
import java.io.File;
import java.io.IOException;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link CgroupsInfo}, {@link CgroupsInfoV1}, {@link CgroupsInfoV2}. */
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
  public void testGetRootCgroup_v1() throws IOException {
    String pathString =
        createFakeAbsoluteFile(
            "/proc/self/mounts",
            "sysfs /sys sysfs rw,nosuid,nodev,noexec,relatime 0 0",
            "cgroup /dev/cgroup/cpu cgroup rw,cpu,cpuacct 0 0",
            "cgroup /dev/cgroup/io cgroup rw,io 0 0",
            "cgroup /dev/cgroup/job cgroup rw,job 0 0",
            "cgroup /dev/cgroup/memory cgroup rw,memory,hugetlb 0 0",
            "proc /proc proc rw,nosuid,nodev,noexec,relatime 0 0");

    CgroupsInfo cgroup = CgroupsInfo.getRootCgroup(new File(pathString));

    assertThat(cgroup.getType()).isEqualTo(CgroupsInfo.Type.ROOT);
    assertThat(cgroup.getVersion()).isEqualTo(CgroupsInfo.Version.V1);
    assertThat(cgroup.getCgroupDir().getPath()).isEqualTo("/dev/cgroup/memory");
  }

  @Test
  public void testGetRootCgroup_v2() throws IOException {
    String pathString =
        createFakeAbsoluteFile(
            "/proc/self/mounts",
            "sysfs /sys sysfs rw,nosuid,nodev,noexec,relatime 0 0",
            "cgroup2 /sys/fs/cgroup cgroup2 rw,memory_recursiveprot 0 0",
            "proc /proc proc rw,nosuid,nodev,noexec,relatime 0 0");

    CgroupsInfo cgroup = CgroupsInfo.getRootCgroup(new File(pathString));

    assertThat(cgroup.getType()).isEqualTo(CgroupsInfo.Type.ROOT);
    assertThat(cgroup.getVersion()).isEqualTo(CgroupsInfo.Version.V2);
    assertThat(cgroup.getCgroupDir().getPath()).isEqualTo("/sys/fs/cgroup");
  }

  @Test
  public void testGetRootCgroup_mixed_v1_has_memory() throws IOException {
    String pathString =
        createFakeAbsoluteFile(
            "/proc/self/mounts",
            "sysfs /sys sysfs rw,nosuid,nodev,noexec,relatime 0 0",
            "cgroup2 /sys/fs/cgroup cgroup2 rw,memory_recursiveprot 0 0",
            "cgroup /dev/cgroup/job cgroup rw,job 0 0",
            "cgroup /dev/cgroup/memory cgroup rw,memory,hugetlb 0 0",
            "proc /proc proc rw,nosuid,nodev,noexec,relatime 0 0");

    CgroupsInfo cgroup = CgroupsInfo.getRootCgroup(new File(pathString));

    assertThat(cgroup.getType()).isEqualTo(CgroupsInfo.Type.ROOT);
    assertThat(cgroup.getVersion()).isEqualTo(CgroupsInfo.Version.V1);
    assertThat(cgroup.getCgroupDir().getPath()).isEqualTo("/dev/cgroup/memory");
  }

  @Test
  public void testGetRootCgroup_mixed_v2_has_memory() throws IOException {
    String pathString =
        createFakeAbsoluteFile(
            "/proc/self/mount",
            "sysfs /sys sysfs rw,nosuid,nodev,noexec,relatime 0 0",
            "cgroup2 /sys/fs/cgroup cgroup2 rw,memory_recursiveprot 0 0",
            "cgroup /dev/cgroup/job cgroup rw,job 0 0",
            "cgroup /dev/cgroup/io cgroup rw,io 0 0",
            "proc /proc proc rw,nosuid,nodev,noexec,relatime 0 0");

    CgroupsInfo cgroup = CgroupsInfo.getRootCgroup(new File(pathString));

    assertThat(cgroup.getType()).isEqualTo(CgroupsInfo.Type.ROOT);
    assertThat(cgroup.getVersion()).isEqualTo(CgroupsInfo.Version.V2);
    assertThat(cgroup.getCgroupDir().getPath()).isEqualTo("/sys/fs/cgroup");
  }

  @Test
  public void testCreateBlazeSpawnsCgroup_v1() throws IOException {
    String mountPath = root + "/dev/cgroup/memory";
    CgroupsInfo rootCgroup =
        new CgroupsInfoV1(CgroupsInfo.Type.ROOT, /* cgroupDir= */ new File(mountPath));
    String procSelfCgroupPath =
        createFakeAbsoluteFile(
            "/proc/self/cgroup",
            "8:net:/netdir/action-6",
            "7:memory,hugetlb:/memdir/action-6",
            "6:job:/jobdir/action-16",
            "5:io:/iodir/action-1");
    scratch.dir(root + "/dev/cgroup/memory/memdir/action-6").createDirectoryAndParents();
    String blazeSpawnsPath =
        root
            + "/dev/cgroup/memory/memdir/action-6/blaze_"
            + ProcessHandle.current().pid()
            + "_spawns";
    scratch.dir(blazeSpawnsPath).createDirectoryAndParents();

    CgroupsInfo cgroup = rootCgroup.createBlazeSpawnsCgroup(procSelfCgroupPath);

    assertThat(cgroup.getCgroupDir().exists()).isTrue();
    assertThat(cgroup.getCgroupDir().getPath()).isEqualTo(blazeSpawnsPath);
    assertThat(cgroup.getType()).isEqualTo(CgroupsInfo.Type.BLAZE_SPAWNS);
    assertThat(cgroup.getVersion()).isEqualTo(CgroupsInfo.Version.V1);
  }

  @Test
  public void testCreateBlazeSpawnsCgroup_v2() throws IOException {
    String mountPath = root + "/sys/fs/cgroup";
    CgroupsInfo rootCgroup =
        new CgroupsInfoV2(CgroupsInfo.Type.ROOT, /* cgroupDir= */ new File(mountPath));
    String procSelfCgroupPath =
        createFakeAbsoluteFile("/proc/self/cgroup", "0::/user.slice/session.scope");
    // In v2, the blaze spawns cgroup is created one step up from where the blaze process is defined
    // in the /proc/self/cgroup file (defined above). Specifically, here it is in the
    // ".../user.slice".
    String blazeSpawnsPath =
        mountPath + "/user.slice/blaze_" + ProcessHandle.current().pid() + "_spawns.slice";
    // Even though the blaze spawn's cgroup directory is meant to be created in the method call,
    // we create it here so that we can prepare the controller files that are expected beforehand.
    scratch.dir(blazeSpawnsPath).createDirectoryAndParents();
    // Since this controllers file is missing `pids`, we expect that to be written to it.
    scratch.file(blazeSpawnsPath + "/cgroup.controllers", "memory pids");
    scratch.file(blazeSpawnsPath + "/cgroup.subtree_control", "memory");

    CgroupsInfo cgroup = rootCgroup.createBlazeSpawnsCgroup(procSelfCgroupPath);

    assertThat(cgroup.getVersion()).isEqualTo(CgroupsInfo.Version.V2);
    assertThat(cgroup.getCgroupDir().exists()).isTrue();
    assertThat(cgroup.getCgroupDir().getPath()).isEqualTo(blazeSpawnsPath);
    assertThat(cgroup.getType()).isEqualTo(CgroupsInfo.Type.BLAZE_SPAWNS);
    assertThat(cgroup.getVersion()).isEqualTo(CgroupsInfo.Version.V2);
    // This is not what an actual cgroups v2 file would contain, but it's what we expect to write to
    // it to enable subtree control.
    assertThat(Files.readLines(new File(blazeSpawnsPath + "/cgroup.subtree_control"), UTF_8))
        .containsExactly("+memory +pids");
  }

  @Test
  public void testCreateIndividualSpawnCgroup_withLimit_v1() throws IOException {
    String blazeSpawnsPath = root + "/dev/cgroup/memory/memdir/action-6/blaze_1234_spawns";
    scratch.dir(blazeSpawnsPath).createDirectoryAndParents();
    CgroupsInfo blazeSpawnsCgroup =
        new CgroupsInfoV1(CgroupsInfo.Type.BLAZE_SPAWNS, new File(blazeSpawnsPath));

    CgroupsInfo cgroup = blazeSpawnsCgroup.createIndividualSpawnCgroup("spawn_1", 100);

    assertThat(cgroup.exists()).isTrue();
    assertThat(cgroup.getCgroupDir().getPath()).isEqualTo(blazeSpawnsPath + "/spawn_1");
    assertThat(cgroup.getType()).isEqualTo(CgroupsInfo.Type.SPAWN);
    assertThat(cgroup.getVersion()).isEqualTo(CgroupsInfo.Version.V1);
    assertThat(Files.readLines(new File(blazeSpawnsPath + "/spawn_1/memory.limit_in_bytes"), UTF_8))
        .containsExactly("104857600");
  }

  @Test
  public void testCreateIndividualSpawnCgroup_noLimit_v1() throws IOException {
    String blazeSpawnsPath = root + "/dev/cgroup/memory/memdir/action-6/blaze_1234_spawns";
    scratch.dir(blazeSpawnsPath).createDirectoryAndParents();
    CgroupsInfo blazeSpawnsCgroup =
        new CgroupsInfoV1(CgroupsInfo.Type.BLAZE_SPAWNS, new File(blazeSpawnsPath));

    CgroupsInfo cgroup = blazeSpawnsCgroup.createIndividualSpawnCgroup("spawn_1", 0);

    assertThat(cgroup.exists()).isTrue();
    assertThat(cgroup.getCgroupDir().getPath()).isEqualTo(blazeSpawnsPath + "/spawn_1");
    assertThat(cgroup.getType()).isEqualTo(CgroupsInfo.Type.SPAWN);
    assertThat(cgroup.getVersion()).isEqualTo(CgroupsInfo.Version.V1);
    // In reality, cgroups should still create this file automatically, but since we don't have
    // that in our tests, the memory limits file should not have been created since it isn't written
    // to.
    assertThat(new File(blazeSpawnsPath + "/spawn_1/memory.limit_in_bytes").exists()).isFalse();
  }

  @Test
  public void testCreateIndividualSpawnCgroup_withLimit_v2() throws IOException {
    String blazeSpawnsPath = root + "/sys/fs/cgroup/user.slice/blaze_1234_spawns.slice";
    scratch.dir(blazeSpawnsPath).createDirectoryAndParents();
    CgroupsInfo blazeSpawnsCgroup =
        new CgroupsInfoV2(CgroupsInfo.Type.BLAZE_SPAWNS, new File(blazeSpawnsPath));

    CgroupsInfo cgroup = blazeSpawnsCgroup.createIndividualSpawnCgroup("spawn_1", 100);

    assertThat(cgroup.exists()).isTrue();
    assertThat(cgroup.getCgroupDir().getPath()).isEqualTo(blazeSpawnsPath + "/spawn_1.scope");
    assertThat(cgroup.getType()).isEqualTo(CgroupsInfo.Type.SPAWN);
    assertThat(cgroup.getVersion()).isEqualTo(CgroupsInfo.Version.V2);
    assertThat(
            Files.readLines(new File(blazeSpawnsPath + "/spawn_1.scope/memory.oom.group"), UTF_8))
        .containsExactly("1");
    assertThat(Files.readLines(new File(blazeSpawnsPath + "/spawn_1.scope/memory.max"), UTF_8))
        .containsExactly("104857600");
  }

  @Test
  public void testCreateIndividualSpawnCgroup_noLimit_v2() throws IOException {
    String blazeSpawnsPath = root + "/sys/fs/cgroup/user.slice/blaze_1234_spawns.slice";
    scratch.dir(blazeSpawnsPath).createDirectoryAndParents();
    CgroupsInfo blazeSpawnsCgroup =
        new CgroupsInfoV2(CgroupsInfo.Type.BLAZE_SPAWNS, new File(blazeSpawnsPath));

    CgroupsInfo cgroup = blazeSpawnsCgroup.createIndividualSpawnCgroup("spawn_1", 0);

    assertThat(cgroup.exists()).isTrue();
    assertThat(cgroup.getCgroupDir().getPath()).isEqualTo(blazeSpawnsPath + "/spawn_1.scope");
    assertThat(cgroup.getType()).isEqualTo(CgroupsInfo.Type.SPAWN);
    assertThat(cgroup.getVersion()).isEqualTo(CgroupsInfo.Version.V2);
    // In reality, cgroups should still create this file automatically, but since we don't have
    // that in our tests, the memory limits files should not have been created since they aren't
    // written to.
    assertThat(new File(blazeSpawnsPath + "/spawn_1.scope/memory.oom.group").exists()).isFalse();
    assertThat(new File(blazeSpawnsPath + "/spawn_1.scope/memory.max").exists()).isFalse();
  }

  @Test
  public void testGetMemoryUsageInKb_v1() throws IOException {
    String cgroupPath = root + "/dev/cgroup/memory/memdir/action-1";
    scratch.dir(cgroupPath).createDirectoryAndParents();
    CgroupsInfo cgroupsInfo =
        new CgroupsInfoV1(CgroupsInfo.Type.SPAWN, /* cgroupDir= */ new File(cgroupPath));

    // It should return 0 if the /path/to/cgroup/memory.usage_in_bytes does not exist.
    assertThat(cgroupsInfo.getMemoryUsageInKb()).isEqualTo(0);

    scratch.file(cgroupPath + "/memory.usage_in_bytes", "1024000");

    assertThat(cgroupsInfo.getMemoryUsageInKb()).isEqualTo(1000);
  }

  @Test
  public void testGetMemoryUsageInKb_v2() throws IOException {
    String cgroupPath = root + "/sys/fs/cgroup/memdir/action-1";
    scratch.dir(cgroupPath).createDirectoryAndParents();
    CgroupsInfo cgroupsInfo =
        new CgroupsInfoV2(CgroupsInfo.Type.SPAWN, /* cgroupDir= */ new File(cgroupPath));

    // It should return 0 if the /path/to/cgroup/memory.current does not exist.
    assertThat(cgroupsInfo.getMemoryUsageInKb()).isEqualTo(0);

    scratch.file(cgroupPath + "/memory.current", "1024000");
    // Divided by 1024.

    assertThat(cgroupsInfo.getMemoryUsageInKb()).isEqualTo(1000);
  }

  @Test
  public void testAddProcess_v1() throws IOException {
    String cgroupPath = root + "/dev/cgroup/memory/memdir/action-1";
    scratch.dir(cgroupPath).createDirectoryAndParents();
    CgroupsInfo cgroupsInfo =
        new CgroupsInfoV1(CgroupsInfo.Type.SPAWN, /* cgroupDir= */ new File(cgroupPath));

    cgroupsInfo.addProcess(1234);

    assertThat(Files.readLines(new File(cgroupsInfo.getCgroupDir(), "cgroup.procs"), UTF_8))
        .containsExactly("1234");
  }

  @Test
  public void testAddProcess_v2() throws IOException {
    String cgroupPath = root + "/sys/fs/cgroup/memdir/action-1";
    scratch.dir(cgroupPath).createDirectoryAndParents();
    CgroupsInfo cgroupsInfo =
        new CgroupsInfoV2(CgroupsInfo.Type.SPAWN, /* cgroupDir= */ new File(cgroupPath));

    cgroupsInfo.addProcess(1234);

    assertThat(Files.readLines(new File(cgroupsInfo.getCgroupDir(), "cgroup.procs"), UTF_8))
        .containsExactly("1234");
  }

  @Test
  public void testGetRootCgroup_returnsInvalidCgroup_whenMountNotFound() throws IOException {
    String pathString = createFakeAbsoluteFile("/proc/self/mounts", "");

    CgroupsInfo cgroup = CgroupsInfo.getRootCgroup(new File(pathString));

    assertThat(cgroup.getClass()).isEqualTo(InvalidCgroupsInfo.class);
    assertThat(cgroup.getType()).isEqualTo(CgroupsInfo.Type.ROOT);
  }

  @Test
  public void testCreateCgroupFromInvalidCgroup_returnsInvalidCgroup() {
    String errorMessage = "Some error message";
    CgroupsInfo invalidRootCgroup =
        new InvalidCgroupsInfo(CgroupsInfo.Type.ROOT, CgroupsInfo.Version.V1, errorMessage);
    CgroupsInfo invalidBlazeSpawnsCgroup =
        new InvalidCgroupsInfo(CgroupsInfo.Type.BLAZE_SPAWNS, CgroupsInfo.Version.V2, errorMessage);

    CgroupsInfo createdBlazeSpawnsCgroup = invalidRootCgroup.createBlazeSpawnsCgroup("");
    CgroupsInfo createdSpawnCgroup =
        invalidBlazeSpawnsCgroup.createIndividualSpawnCgroup("spawn_1", 1);

    assertThat(createdBlazeSpawnsCgroup.getClass()).isEqualTo(InvalidCgroupsInfo.class);
    // Should still have the same version as the parent cgroup that attempted to create it.
    assertThat(createdBlazeSpawnsCgroup.getVersion()).isEqualTo(CgroupsInfo.Version.V1);
    assertThat(createdBlazeSpawnsCgroup.getType()).isEqualTo(CgroupsInfo.Type.BLAZE_SPAWNS);

    assertThat(createdSpawnCgroup.getClass()).isEqualTo(InvalidCgroupsInfo.class);
    // Should still have the same version as the parent cgroup that attempted to create it.
    assertThat(createdSpawnCgroup.getVersion()).isEqualTo(CgroupsInfo.Version.V2);
    assertThat(createdSpawnCgroup.getType()).isEqualTo(CgroupsInfo.Type.SPAWN);
  }

  private String createFakeAbsoluteFile(String fileName, String... contents) throws IOException {
    return scratch.file(root + fileName, contents).getPathString();
  }
}
