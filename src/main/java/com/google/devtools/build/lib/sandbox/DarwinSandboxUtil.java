// Copyright 2023 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.Lists;
import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.server.FailureDetails.Sandbox.Code;
import com.google.devtools.build.lib.vfs.Path;
import java.util.List;
import java.util.Map;
import java.util.SortedMap;

/** Utility functions for the {@code darwin-sandbox} embedded tool. */
public final class DarwinSandboxUtil {
  /**
   * This method does the following things:
   *
   * <ul>
   *   <li>If mount source does not exist on the host system, throw an error message
   *   <li>If mount target exists under sandbox execution root, throw an error message
   *   <li>If mount target is under sandbox execution root and is a child of another mount target, throw an error message
   * </ul>
   *
   * @param sandboxExecRoot    the sandbox execution root
   * @param bindMounts         the bind mounts map with target as key and source as value
   * @throws UserExecException if any of the mount points are not valid
   */
  public static void validateBindMounts(Path sandboxExecRoot, SortedMap<Path, Path> bindMounts) throws UserExecException {
    final List<Path> validatedTargets = Lists.newArrayListWithCapacity(bindMounts.size());

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
      if (target.startsWith(sandboxExecRoot)) {
        // Mount target should not exist
        if (target.exists()) {
          throw new UserExecException(
              SandboxHelpers.createFailureDetail(
                  String.format(
                      "Mount target '%s' already exists",
                      target),
                  Code.MOUNT_TARGET_ALREADY_EXIST));
        }
        // Mount target should not be child of another target
        for (int i = validatedTargets.size() - 1; i >= 0; --i) {
          final Path validatedTarget = validatedTargets.get(i);
          if (target.startsWith(validatedTarget)) {
            throw new UserExecException(
                SandboxHelpers.createFailureDetail(
                    String.format(
                        "Mount target '%s' is a child of target '%s'", validatedTarget, target),
                    Code.BIND_MOUNT_ANALYSIS_FAILURE));
          }
        }
        validatedTargets.add(target);
      }
    }
  }
}
