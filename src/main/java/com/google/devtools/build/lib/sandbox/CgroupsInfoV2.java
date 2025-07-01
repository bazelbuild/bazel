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

package com.google.devtools.build.lib.sandbox;

import static com.google.common.base.Preconditions.checkArgument;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.base.Joiner;
import com.google.common.base.Splitter;
import com.google.common.io.Files;
import java.io.File;
import java.io.IOException;
import java.util.List;
import javax.annotation.Nullable;

/** Represents a v2 cgroup. */
public class CgroupsInfoV2 extends CgroupsInfo {

  public CgroupsInfoV2(Type type, @Nullable File cgroupDir) {
    super(type, Version.V2, cgroupDir);
  }

  @Override
  public CgroupsInfo createBlazeSpawnsCgroup(String procSelfCgroupPath) {
    checkArgument(
        type == Type.ROOT, "Should only be creating the Blaze spawns cgroup from the root cgroup.");
    File blazeProcessCgroupDir;
    File blazeSpawnsDir;
    try {
      blazeProcessCgroupDir = getBlazeProcessCgroupDir(cgroupDir, procSelfCgroupPath);
      // In cgroups v2, we need to step back from the leaf node to make a further hierarchy.
      blazeSpawnsDir =
          new File(
              blazeProcessCgroupDir.getParentFile(),
              "blaze_" + ProcessHandle.current().pid() + "_spawns.slice");
      blazeSpawnsDir.mkdirs();
      blazeSpawnsDir.deleteOnExit();
      setSubtreeControllers(blazeSpawnsDir);
    } catch (Exception e) {
      return new InvalidCgroupsInfo(Type.BLAZE_SPAWNS, getVersion(), e);
    }

    return new CgroupsInfoV2(Type.BLAZE_SPAWNS, blazeSpawnsDir);
  }

  @Override
  public CgroupsInfo createIndividualSpawnCgroup(String dirName, int memoryLimitMb) {
    checkArgument(
        type == Type.BLAZE_SPAWNS,
        "Should only be creating the individual spawn's cgroup from the Blaze spawns cgroup.");
    if (!canWrite()) {
      return new InvalidCgroupsInfo(
          Type.SPAWN,
          getVersion(),
          String.format("Cgroup %s is invalid, unable to create spawn's cgroup here.", cgroupDir));
    }
    File spawnCgroupDir = new File(cgroupDir, dirName + ".scope");
    spawnCgroupDir.mkdirs();
    spawnCgroupDir.deleteOnExit();
    try {
      if (memoryLimitMb > 0) {
        // In cgroups v2, we need to propagate the controllers into new subdirs.
        Files.asCharSink(new File(spawnCgroupDir, "memory.oom.group"), UTF_8).write("1\n");
        Files.asCharSink(new File(spawnCgroupDir, "memory.max"), UTF_8)
            .write(Long.toString(memoryLimitMb * 1024L * 1024L));
        // Set swap to 0 so that it doesn't unexpecedly consume more than
        // the memory limit when swap is enabled.
        Files.asCharSink(new File(spawnCgroupDir, "memory.swap.max"), UTF_8)
            .write(Long.toString(0L));
      }
    } catch (Exception e) {
      return new InvalidCgroupsInfo(Type.SPAWN, getVersion(), e);
    }
    return new CgroupsInfoV2(Type.SPAWN, spawnCgroupDir);
  }

  @Override
  public int getMemoryUsageInKb() {
    return getMemoryUsageInKbFromFile("memory.current");
  }

  /**
   * Returns the path to the cgroup containing the Blaze process.
   *
   * <p>In v2, there is only one entry, and it looks something like this:
   *
   * <pre>
   * 0::/user.slice/user-123.slice/session-1.scope
   * </pre>
   *
   * @param mountPoint the directory where the cgroup hierarchy is mounted.
   * @param procSelfCgroupPath path for the /proc/self/cgroup file.
   * @throws IOException if there are errors reading the given procs cgroup file.
   */
  private static File getBlazeProcessCgroupDir(File mountPoint, String procSelfCgroupPath)
      throws IOException {
    List<String> contents = Files.readLines(new File(procSelfCgroupPath), UTF_8);
    if (contents.isEmpty()) {
      throw new IllegalStateException(
          "Found no memory cgroup entries in '" + procSelfCgroupPath + "'");
    }
    List<String> parts = Splitter.on(":").limit(3).splitToList(contents.get(0));
    return new File(mountPoint, parts.get(2));
  }

  /**
   * Sets the subtree controllers we need. This also checks that the controllers are available.
   *
   * @param blazeDir A directory in the cgroups hierarchy.
   * @throws IOException If reading or writing the {@code cgroup.controllers} or {@code
   *     cgroup.subtree_control} file fails.
   * @throws IllegalStateException if the {@code memory} and {code pids} controllers are either not
   *     available or cannot be set for subtrees.
   */
  private static void setSubtreeControllers(File blazeDir) throws IOException {
    var controllers =
        Joiner.on(' ').join(Files.readLines(new File(blazeDir, "cgroup.controllers"), UTF_8));
    if (!(controllers.contains("memory") && controllers.contains("pids"))) {
      throw new IllegalStateException(
          String.format(
              "Required controllers 'memory' and 'pids' not found in %s/cgroup.controllers",
              blazeDir));
    }
    var subtreeControllers =
        Joiner.on(' ').join(Files.readLines(new File(blazeDir, "cgroup.subtree_control"), UTF_8));
    if (!subtreeControllers.contains("memory") || !subtreeControllers.contains("pids")) {
      Files.asCharSink(new File(blazeDir, "cgroup.subtree_control"), UTF_8)
          .write("+memory +pids\n");
    }
  }
}
