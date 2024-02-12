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

import com.google.common.base.Splitter;
import com.google.common.io.Files;
import java.io.File;
import java.io.IOException;
import java.util.List;
import javax.annotation.Nullable;

/** Represents a v1 cgroup. */
public class CgroupsInfoV1 extends CgroupsInfo {
  public CgroupsInfoV1(Type type, @Nullable File cgroupDir) {
    super(type, Version.V1, cgroupDir);
  }

  @Override
  public CgroupsInfo createBlazeSpawnsCgroup(String procSelfCgroupPath) {
    checkArgument(
        type == Type.ROOT, "Should only be creating the Blaze spawns cgroup from the root cgroup.");
    File blazeProcessCgroupDir;
    try {
      blazeProcessCgroupDir = getBlazeProcessCgroupDir(cgroupDir, procSelfCgroupPath);
    } catch (Exception e) {
      return new InvalidCgroupsInfo(Type.BLAZE_SPAWNS, getVersion(), e);
    }
    File blazeSpawnsDir =
        new File(blazeProcessCgroupDir, "blaze_" + ProcessHandle.current().pid() + "_spawns");
    blazeSpawnsDir.mkdirs();
    blazeSpawnsDir.deleteOnExit();
    return new CgroupsInfoV1(Type.BLAZE_SPAWNS, blazeSpawnsDir);
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
    File spawnCgroupDir = new File(cgroupDir, dirName);
    spawnCgroupDir.mkdirs();
    spawnCgroupDir.deleteOnExit();
    try {
      if (memoryLimitMb > 0) {
        Files.asCharSink(new File(spawnCgroupDir, "memory.limit_in_bytes"), UTF_8)
            .write(Long.toString(memoryLimitMb * 1024L * 1024L));
      }
    } catch (Exception e) {
      return new InvalidCgroupsInfo(Type.SPAWN, getVersion(), e);
    }
    return new CgroupsInfoV1(Type.SPAWN, spawnCgroupDir);
  }

  /**
   * Returns the path to the cgroup containing the Blaze process.
   *
   * <p>The <code>/proc/self/cgroup</code> file look like this in v1:
   *
   * <pre>
   * 8:net:/some/path
   * 7:memory,hugetlb:/some/other/path
   * ...
   * </pre>
   *
   * @param mountPoint the directory where the cgroup hierarchy is mounted.
   * @param procSelfCgroupPath path for the /proc/self/cgroup file.
   * @throws IOException if there are errors reading the given procs cgroup file.
   */
  private static File getBlazeProcessCgroupDir(File mountPoint, String procSelfCgroupPath)
      throws IOException {
    List<String> controllers = Files.readLines(new File(procSelfCgroupPath), UTF_8);
    String memoryController =
        controllers.stream()
            .filter(controller -> controller.contains("memory"))
            .findFirst()
            .orElseThrow(
                () ->
                    new IllegalStateException(
                        "Found no memory cgroup entries in '" + procSelfCgroupPath + "'"));
    List<String> parts = Splitter.on(":").limit(3).splitToList(memoryController);
    return new File(mountPoint, parts.get(2));
  }

  @Override
  public int getMemoryUsageInKb() {
    return getMemoryUsageInKbFromFile("memory.usage_in_bytes");
  }
}
