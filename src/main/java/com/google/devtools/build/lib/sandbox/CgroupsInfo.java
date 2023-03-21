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

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Joiner;
import com.google.common.base.Splitter;
import com.google.common.flogger.GoogleLogger;
import com.google.common.io.Files;
import com.google.devtools.build.lib.util.Pair;
import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import javax.annotation.Nullable;

/** This class manages cgroups directories for memory-limiting sandboxed processes. */
public class CgroupsInfo {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  /**
   * A regexp that matches cgroups entries in {@code /proc/mounts}.
   *
   * <p>Group 1 is empty (cgroups v1) or '2' (cgroups v2) Group 2 is the mount point. Group 3 is the
   * options, which for v1 includes which hierarchies are mounted here.
   */
  private static final Pattern CGROUPS_MOUNT_PATTERN =
      Pattern.compile("^cgroup(|2)\\s+(\\S*)\\s+cgroup2?\\s+(\\S*).*");

  public static final String PROC_SELF_MOUNTS_PATH = "/proc/self/mounts";
  public static final String PROC_SELF_CGROUP_PATH = "/proc/self/cgroup";
  /** If non-null, this is a cgroups directory that sandboxes can put their directories into. */
  @Nullable private static volatile CgroupsInfo instance;

  private final boolean isCgroupsV2;
  private final File blazeDir;
  private final File mountPoint;

  private CgroupsInfo(boolean isCgroupsV2, File blazeDir, File mountPoint) {
    this.isCgroupsV2 = isCgroupsV2;
    this.blazeDir = blazeDir;
    this.mountPoint = mountPoint;
  }

  /**
   * Creates a cgroups directory for Blaze to place sandboxes in. Figures out whether cgroups v1 or
   * v2 is available, and for cgroups v2 sets subtree control for the <code>memory</code> and <code>
   * pids</code> controllers.
   *
   * <p>The cgroups directory is created at most once per Blaze instance.
   *
   * @return A CgroupsInfo object that defines the cgroups directory that Blaze can use for
   *     sub-processes. The Blaze process itself is not moved into this directory.
   * @throws IOException If there are errors reading any of the required files.
   */
  public static CgroupsInfo create() throws IOException {
    return create(PROC_SELF_MOUNTS_PATH, PROC_SELF_CGROUP_PATH);
  }

  @VisibleForTesting
  static CgroupsInfo create(String procSelfMountsPath, String procSelfCgroupPath)
      throws IOException {
    Pair<File, Boolean> cgroupsMount = getMemoryCgroupInfo(new File(procSelfMountsPath));
    File blazeDir;
    File cgroupsMountPoint = cgroupsMount.first;
    if (cgroupsMount.second) {
      File cgroupsNode = getBlazeMemoryCgroup(cgroupsMountPoint, 0, procSelfCgroupPath);
      // In cgroups v2, we need to step back from the leaf node to make a further hierarchy.
      blazeDir =
          new File(
              cgroupsNode.getParentFile(),
              "blaze_" + ProcessHandle.current().pid() + "_spawns.slice");
      blazeDir.mkdirs();
      blazeDir.deleteOnExit();
      setSubtreeControllers(blazeDir);
      logger.atInfo().log("Creating cgroups v2 node at %s", blazeDir);
      return new CgroupsInfo(true, blazeDir, cgroupsMountPoint);
    } else {
      int memoryHierarchy = getMemoryHierarchy(new File(procSelfCgroupPath));
      File cgroupsNode =
          getBlazeMemoryCgroup(cgroupsMountPoint, memoryHierarchy, procSelfCgroupPath);
      blazeDir = new File(cgroupsNode, "blaze_" + ProcessHandle.current().pid() + "_spawns");
      blazeDir.mkdirs();
      blazeDir.deleteOnExit();
      logger.atInfo().log("Creating cgroups v1 node at %s", blazeDir);
      return new CgroupsInfo(false, blazeDir, cgroupsMountPoint);
    }
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

  public boolean isCgroupsV2() {
    return isCgroupsV2;
  }

  /** A cgroups directory for this Blaze instance to put sandboxes in. */
  public File getBlazeDir() {
    return blazeDir;
  }

  /** The place where the cgroups (memory) file system is mounted. */
  public File getMountPoint() {
    return mountPoint;
  }

  /**
   * Reads from the given file (e.g. /proc/mounts) where cgroups are mounted. If both cgroups v1 and
   * cgroups v2 are mounted, the one that has the memory controller is used.
   *
   * <p>In cgroups v1, a typical mount line looks like this (note {@code memory} in the options):
   *
   * <pre>
   * cgroup /dev/cgroup/memory cgroup rw,memory,hugetlb 0 0
   * </pre>
   *
   * In cgroups v2, there is only one relevant line, and it looks like this:
   *
   * <pre>
   * cgroup2 /sys/fs/cgroup cgroup2 rw,[...] 0 0
   * </pre>
   *
   * @param procMountsPath Paths of the mounts file, e.g. /proc/mounts
   * @return Pair of
   *     <ol>
   *       <li>the path of the cgroups mount (for cgroups v1, this is the memory hierarchy) and
   *       <li>whether this is cgroups v2.
   *     </ol>
   *
   * @throws IOException If there are errors reading the given file.
   */
  @VisibleForTesting
  static Pair<File, Boolean> getMemoryCgroupInfo(File procMountsPath) throws IOException {
    var procMountContents = Files.readLines(procMountsPath, UTF_8);
    Pair<File, Boolean> v1 = null;
    Pair<File, Boolean> v2 = null;
    for (String s : procMountContents) {
      Matcher m = CGROUPS_MOUNT_PATTERN.matcher(s);
      if (m.matches()) {
        if (m.group(1).isEmpty()) {
          // v1
          if (m.group(3).contains("memory")) {
            // For now, we only care about the memory cgroup
            v1 = Pair.of(new File(m.group(2)), false);
          }
        } else {
          // v2
          v2 = Pair.of(new File(m.group(2)), true);
        }
      }
    }
    // If we found the memory controller in v1, we use that, just in case we have a hybrid system
    // where some controllers are v1 and some are v2. It would be harder to detect if v2 has the
    // memory controller
    if (v1 != null) {
      return v1;
    }
    if (v2 != null) {
      return v2;
    }
    throw new IllegalStateException(
        "Cgroups requested, but no applicable cgroups are mounted on this machine");
  }

  /**
   * Returns the number of the memory cgroups v1 hierarchy.
   *
   * <p>The <code>/proc/self/cgroup</code> file look like this in v1:
   *
   * <pre>
   * 8:net:/some/path
   * 7:memory,hugetlb:/some/other/path
   * ...
   * </pre>
   *
   * In v2, there is only one entry, and it looks something like
   *
   * <pre>
   * 0::/user.slice/user-123.slice/session-1.scope
   * </pre>
   *
   * @param procSelfCgroupPath Path for the <code>/proc/self/cgroup</code> file.
   * @return The hierarchy number for the cgroups v1 hierarchy that contains the memory controller.
   * @throws IOException If there are errors reading the file.
   */
  @VisibleForTesting
  static int getMemoryHierarchy(File procSelfCgroupPath) throws IOException {
    List<String> devCgroupContents = Files.readLines(procSelfCgroupPath, UTF_8);
    for (String s : devCgroupContents) {
      if (s.contains("memory")) {
        return Integer.parseInt(Splitter.on(":").split(s).iterator().next());
      }
    }
    throw new IllegalStateException(
        String.format(
            "Cgroups v1 requested, but no memory cgroup found in %s", procSelfCgroupPath));
  }

  /**
   * Returns the path of the memory cgroups node that Blaze itself runs inside.
   *
   * @param mountPoint Where the cgroups hierarchy (with the memory controller for v1) is mounted.
   * @param memoryHierarchyId The v1 hierarchy that contains the memory controller, or 0 for v2.
   * @param procSelfPath The path of the <code>/proc/self/cgroup</code> file.
   * @return A <code>File</code> object of the cgroup directory of the current process.
   * @throws IOException If the given file cannot be read.
   */
  @VisibleForTesting
  static File getBlazeMemoryCgroup(File mountPoint, int memoryHierarchyId, String procSelfPath)
      throws IOException {
    var procSelfCgroupContents = Files.readLines(new File(procSelfPath), UTF_8);
    if (procSelfCgroupContents.isEmpty()) {
      throw new IOException("Cgroups requested, but /proc/self/cgroup is empty");
    }
    File cgroupsNode = null;
    for (String s : procSelfCgroupContents) {
      List<String> parts = Splitter.on(":").limit(3).splitToList(s);
      if (parts.size() == 3 && Integer.parseInt(parts.get(0)) == memoryHierarchyId) {
        String path = parts.get(2);
        if (path.startsWith(File.pathSeparator)) {
          path = path.substring(1);
        }
        cgroupsNode = new File(mountPoint, path);
        break;
      }
    }
    if (cgroupsNode == null) {
      throw new IllegalStateException("Found no memory cgroups entries in '" + procSelfPath + "'");
    }
    if (!cgroupsNode.exists()) {
      throw new IllegalStateException("Cgroups node '" + cgroupsNode + "' does not exist");
    }
    if (!cgroupsNode.isDirectory()) {
      throw new IllegalStateException("Cgroups node " + cgroupsNode + " is not a directory");
    }
    if (!cgroupsNode.canWrite()) {
      throw new IllegalStateException("Cgroups node " + cgroupsNode + " is not writable");
    }
    return cgroupsNode;
  }

  public static CgroupsInfo getInstance() throws IOException {
    if (instance == null) {
      synchronized (CgroupsInfo.class) {
        if (instance == null) {
          instance = create();
        }
      }
    }
    return instance;
  }

  /**
   * Creates a cgroups directory with the given memory limit.
   *
   * @param memoryLimit Memory limit in megabytes (MiB).
   * @param dirName Base name of the directory created. In cgroups v2, <code>.scope</code> gets
   *     appended.
   */
  public String createMemoryLimitCgroupDir(String dirName, int memoryLimit) throws IOException {
    File cgroupsDir;
    if (isCgroupsV2) {
      cgroupsDir = new File(blazeDir, dirName + ".scope");
      cgroupsDir.mkdirs();
      cgroupsDir.deleteOnExit();
      // In cgroups v2, we need to propagate the controllers into new subdirs.
      Files.asCharSink(new File(cgroupsDir, "memory.oom.group"), UTF_8).write("1\n");
      Files.asCharSink(new File(cgroupsDir, "memory.max"), UTF_8)
          .write(Long.toString(memoryLimit * 1024L * 1024L));
    } else {
      cgroupsDir = new File(getBlazeDir(), dirName);
      cgroupsDir.mkdirs();
      cgroupsDir.deleteOnExit();
      Files.asCharSink(new File(cgroupsDir, "memory.limit_in_bytes"), UTF_8)
          .write(Long.toString(memoryLimit * 1024L * 1024L));
    }
    return cgroupsDir.toString();
  }
}
