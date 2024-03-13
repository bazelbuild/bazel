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
import com.google.common.base.Suppliers;
import com.google.common.flogger.GoogleLogger;
import com.google.common.io.Files;
import com.google.devtools.build.lib.util.OS;
import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.function.Supplier;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import javax.annotation.Nullable;

/** This class manages cgroups directories for memory-limiting sandboxed processes. */
public abstract class CgroupsInfo implements Cgroup {

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  /**
   * A regexp that matches cgroups entries in {@code /proc/mounts}.
   *
   * <p>Group 1 is empty (cgroups v1) or '2' (cgroups v2) Group 2 is the mount point. Group 3 is the
   * options, which for v1 includes which hierarchies are mounted here.
   */
  private static final Pattern CGROUPS_MOUNT_PATTERN =
      Pattern.compile("^cgroup(|2)\\s+(\\S*)\\s+cgroup2?\\s+(\\S*).*");

  private static final String PROC_SELF_MOUNTS_PATH = "/proc/self/mounts";
  private static final String PROC_SELF_CGROUP_PATH = "/proc/self/cgroup";

  private static final CgroupsInfo rootCgroup = getRootCgroup(new File(PROC_SELF_MOUNTS_PATH));

  private static final Supplier<CgroupsInfo> blazeSpawnsCgroupSupplier =
      Suppliers.memoize(CgroupsInfo::createBlazeSpawnsCgroup);

  /** Returns whether the local machine supports cgroups. */
  public static boolean isSupported() {
    return OS.getCurrent() == OS.LINUX && getBlazeSpawnsCgroup().canWrite();
  }

  /**
   * Returns an instance of the root cgroup of the hierarchy, {@link InvalidCgroupsInfo} if invalid.
   *
   * <p>For v1, we only care about the memory hierarchy.
   *
   * @param procMountsFile the /proc/self/mounts file.
   */
  @VisibleForTesting
  static CgroupsInfo getRootCgroup(File procMountsFile) {
    if (OS.getCurrent() != OS.LINUX) {
      return new InvalidCgroupsInfo(
          Type.ROOT, /* version= */ null, "Croups is not supported on non-linux environments.");
    }

    List<String> procMountsContents;
    try {
      procMountsContents = Files.readLines(procMountsFile, UTF_8);
    } catch (IOException e) {
      return new InvalidCgroupsInfo(Type.ROOT, /* version= */ null, e);
    }
    File v1RootDir = null;
    File v2RootDir = null;
    for (String s : procMountsContents) {
      Matcher m = CGROUPS_MOUNT_PATTERN.matcher(s);
      if (m.matches()) {
        if (m.group(1).isEmpty()) {
          // v1
          if (m.group(3).contains("memory")) {
            // For now, we only care about the memory cgroup
            v1RootDir = new File(m.group(2));
          }
        } else {
          v2RootDir = new File(m.group(2));
        }
      }
    }
    // If we found the memory controller in v1, we use that, just in case we have a hybrid system
    // where some controllers are v1 and some are v2. It would be harder to detect if v2 has the
    // memory controller
    if (v1RootDir != null) {
      return new CgroupsInfoV1(Type.ROOT, v1RootDir);
    }
    if (v2RootDir != null) {
      return new CgroupsInfoV2(Type.ROOT, v2RootDir);
    }
    return new InvalidCgroupsInfo(
        Type.ROOT,
        /* version= */ null,
        String.format(
            "No cgroups mounted in %s: %s", procMountsFile.getPath(), procMountsContents));
  }

  /**
   * Returns the singleton {@link Type.BLAZE_SPAWNS} cgroup created under the root cgroup, {@link
   * InvalidCgroupsInfo} if invalid.
   */
  public static CgroupsInfo getBlazeSpawnsCgroup() {
    return blazeSpawnsCgroupSupplier.get();
  }

  private static CgroupsInfo createBlazeSpawnsCgroup() {
    if (!rootCgroup.exists()) {
      return new InvalidCgroupsInfo(
          Type.BLAZE_SPAWNS, rootCgroup.getVersion(), "Root cgroup does not exist.");
    }
    return rootCgroup.createBlazeSpawnsCgroup(PROC_SELF_CGROUP_PATH);
  }

  /**
   * Creates a cgroups directory for Blaze to place spawns in.
   *
   * <p>This cgroups directory is created at most once per Blaze instance.
   *
   * @param procSelfCgroupPath path to the <code>/proc/self/cgroup</code> file
   * @return A CgroupsInfo object representing the created cgroup that Blaze can use for
   *     sub-processes (the Blaze process itself is not moved into this directory). If unable to
   *     create, returns an {@link InvalidCgroupsInfo} containing the exception.
   */
  public abstract CgroupsInfo createBlazeSpawnsCgroup(String procSelfCgroupPath);

  /** The version of Cgroups that is currently being used. */
  public enum Version {
    V1,
    V2,
  }

  @Nullable protected Version version;

  /**
   * The types of cgroups relevant to Blaze:
   *
   * <ul>
   *   <li>ROOT: corresponds to the root cgroup where * the hierarchy is mounted at; one of
   *       "/dev/cgroup/{controller}" or "/sys/fs/cgroup".
   *   <li>BLAZE_SPAWNS: corresponds the overarching cgroup that contains children {@link
   *       Type.SPAWN} cgroups.
   *   <li>SPAWN: corresponds to the cgroup for a single spawn - this could be a locally executed
   *       action or a worker process.
   * </ul>
   */
  public enum Type {
    ROOT,
    BLAZE_SPAWNS,
    SPAWN,
  }

  protected Type type;

  /**
   * This is the directory where the cgroup is in, any related files pertaining to limits / resource
   * usage or child cgroups (nested directories) are found here.
   */
  @Nullable protected final File cgroupDir;

  public CgroupsInfo(Type type, @Nullable Version version, @Nullable File cgroupDir) {
    this.version = version;
    this.type = type;
    this.cgroupDir = cgroupDir;
    // Valid.
    if (exists()) {
      logger.atInfo().log(
          "Successfully found / created %s (%s) cgroup at %s", version, type, cgroupDir.getPath());
    }
  }

  /** Returns whether the cgroup at {@code cgroupDir} exists. */
  public boolean exists() {
    return cgroupDir != null && cgroupDir.exists() && cgroupDir.isDirectory();
  }

  /** Returns whether Blaze can write to the current cgroup at {@code cgroupDir}. */
  public boolean canWrite() {
    return exists() && cgroupDir.canWrite();
  }

  /** A cgroups directory for this Blaze instance to put sandboxes in. */
  public File getCgroupDir() {
    return cgroupDir;
  }

  @Nullable
  public Version getVersion() {
    return version;
  }

  public Type getType() {
    return type;
  }

  public int getMemoryUsageInKb() {
    return 0;
  }

  public int getMemoryUsageInKbFromFile(String filename) {
    try {
      String val = Files.readLines(new File(cgroupDir, filename), UTF_8).get(0);
      return (int) (Long.parseLong(val) / 1024);
    } catch (IOException e) {
      return 0;
    }
  }

  public void addProcess(long pid) throws IOException {
    Files.asCharSink(new File(cgroupDir, "cgroup.procs"), UTF_8).write(Long.toString(pid));
  }

  /**
   * Creates a cgroups directory for individual spawns (local / workers).
   *
   * <p>Has to be called from a {@link Type.BLAZE_SPAWNS} cgroup.
   *
   * @param dirName the directory name of the spawn's cgroup.
   * @param memoryLimitMb memory limit in Mb to set on the cgroup. If 0, no limit is set.
   * @return an instance of the spawn's cgroup; if unable to create, returns an {@link
   *     InvalidCgroupsInfo} containing the exception.
   */
  public abstract CgroupsInfo createIndividualSpawnCgroup(String dirName, int memoryLimitMb);

  /**
   * Represents an invalid cgroup so that we can distinguish between whether a cgroup was not meant
   * to be created (null) or if it was attempted but failed.
   */
  public static class InvalidCgroupsInfo extends CgroupsInfo {

    private final Exception exception;

    public InvalidCgroupsInfo(Type type, @Nullable Version version, String errorMessage) {
      super(type, version, null);
      this.exception = new IllegalStateException(errorMessage);
      logger.atInfo().withCause(exception).log("Unable to create cgroup.");
    }

    public InvalidCgroupsInfo(Type type, @Nullable Version version, Exception exception) {
      super(type, version, null);
      logger.atInfo().withCause(exception).log("Unable to create cgroup.");
      this.exception = exception;
    }

    @Override
    public boolean exists() {
      return false;
    }

    @Override
    public boolean canWrite() {
      return false;
    }

    public Exception getException() {
      return exception;
    }

    @Override
    public CgroupsInfo createBlazeSpawnsCgroup(String procSelfCgroupPath) {
      return new InvalidCgroupsInfo(
          Type.BLAZE_SPAWNS,
          getVersion(),
          "Unable to create BLAZE_SPAWNS cgroup from an invalid cgroup.");
    }

    @Override
    public CgroupsInfo createIndividualSpawnCgroup(String dirName, int memoryLimitMb) {
      return new InvalidCgroupsInfo(
          Type.SPAWN, getVersion(), "Unable to create SPAWN cgroup from an invalid cgroup.");
    }
  }
}
