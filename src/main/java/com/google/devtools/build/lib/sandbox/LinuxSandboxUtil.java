// Copyright 2017 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.sandbox;

import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.runtime.BlazeWorkspace;
import com.google.devtools.build.lib.server.FailureDetails.Sandbox.Code;
import com.google.devtools.build.lib.util.OsUtils;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import java.util.Map;

/** Utility functions for the {@code linux-sandbox} embedded tool. */
public final class LinuxSandboxUtil {

  private static final String LINUX_SANDBOX = "linux-sandbox" + OsUtils.executableExtension();

  /** Returns whether using the {@code linux-sandbox} is supported in the command environment. */
  public static boolean isSupported(BlazeWorkspace blazeWorkspace) {
    // We can only use the linux-sandbox if the linux-sandbox exists in the embedded tools.
    // This might not always be the case, e.g. while bootstrapping.
    return getLinuxSandbox(blazeWorkspace) != null;
  }

  /** Returns the path of the {@code linux-sandbox} binary, or null if it doesn't exist. */
  public static Path getLinuxSandbox(BlazeWorkspace blazeWorkspace) {
    return blazeWorkspace.getBinTools().getEmbeddedPath(LINUX_SANDBOX);
  }

  /**
   * This method does the following things:
   *
   * <ul>
   *   <li>If mount source does not exist on the host system, throw an error message
   *   <li>If mount target exists, check whether the source and target are of the same type
   *   <li>If mount target does not exist on the host system, throw an error message
   * </ul>
   *
   * @param bindMounts the bind mounts map with target as key and source as value
   * @throws UserExecException if any of the mount points are not valid
   */
  public static void validateBindMounts(Map<Path, Path> bindMounts) throws UserExecException {
    for (Map.Entry<Path, Path> bindMount : bindMounts.entrySet()) {
      final Path source = bindMount.getValue();
      final Path target = bindMount.getKey();
      // Mount source should exist in the file system
      if (!source.exists()) {
        throw new UserExecException(
            SandboxHelpers.createFailureDetail(
                String.format("Mount source '%s' does not exist.", source),
                Code.MOUNT_SOURCE_DOES_NOT_EXIST));
      }
      // If target exists, but is not of the same type as the source, then we cannot mount it.
      if (target.exists()) {
        boolean areBothDirectories = source.isDirectory() && target.isDirectory();
        boolean isSourceFile = source.isFile() || source.isSymbolicLink();
        boolean isTargetFile = target.isFile() || target.isSymbolicLink();
        boolean areBothFiles = isSourceFile && isTargetFile;
        if (!(areBothDirectories || areBothFiles)) {
          // Source and target are not of the same type; we cannot mount it.
          throw new UserExecException(
              SandboxHelpers.createFailureDetail(
                  String.format(
                      "Mount target '%s' is a %s but mount source '%s' is a %s, they must be the"
                          + " same type.",
                      target,
                      (isTargetFile ? "file" : "directory"),
                      source,
                      (isSourceFile ? "file" : "directory")),
                  Code.MOUNT_SOURCE_TARGET_TYPE_MISMATCH));
        }
      } else {
        // Mount target should exist in the file system
        throw new UserExecException(
            SandboxHelpers.createFailureDetail(
                String.format(
                    "Mount target '%s' does not exist. Bazel only supports bind mounting on top of "
                        + "existing files/directories. Please create an empty file or directory at "
                        + "the mount target path according to the type of mount source.",
                    target),
                Code.MOUNT_TARGET_DOES_NOT_EXIST));
      }
    }
  }

  /** Returns a newly created inaccessible file underneath the given directory. */
  public static Path getInaccessibleHelperFile(Path sandboxBase) throws IOException {
    // The order of the permissions settings calls matters, see
    // https://github.com/bazelbuild/bazel/issues/16364
    Path inaccessibleHelperFile = sandboxBase.getRelative(SandboxHelpers.INACCESSIBLE_HELPER_FILE);
    FileSystemUtils.touchFile(inaccessibleHelperFile);
    inaccessibleHelperFile.setExecutable(false);
    inaccessibleHelperFile.setWritable(false);
    inaccessibleHelperFile.setReadable(false);
    return inaccessibleHelperFile;
  }

  /** Returns a newly created inaccessible directory underneath the given directory. */
  public static Path getInaccessibleHelperDir(Path sandboxBase) throws IOException {
    // The order of the permissions settings calls matters, see
    // https://github.com/bazelbuild/bazel/issues/16364
    Path inaccessibleHelperDir = sandboxBase.getRelative(SandboxHelpers.INACCESSIBLE_HELPER_DIR);
    inaccessibleHelperDir.createDirectory();
    inaccessibleHelperDir.setExecutable(false);
    inaccessibleHelperDir.setWritable(false);
    inaccessibleHelperDir.setReadable(false);
    return inaccessibleHelperDir;
  }
}
