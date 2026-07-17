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
package com.google.devtools.build.lib.runtime.commands;

import static java.nio.charset.StandardCharsets.ISO_8859_1;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.server.CommandProtos.PathToReplace;
import com.google.devtools.build.lib.vfs.Path;
import com.google.protobuf.ByteString;
import javax.annotation.Nullable;

/** Helpers for constructing {@link ExecRequest}s. */
public class PathToReplaceUtils {

  /** Returns the common required {@link PathToReplace} list. */
  public static ImmutableList<PathToReplace> getPathsToReplace(CommandEnvironment env) {
    ImmutableList.Builder<PathToReplace> pathsToReplace = ImmutableList.builder();
    pathsToReplace
        .add(
            PathToReplace.newBuilder()
                .setType(PathToReplace.Type.OUTPUT_BASE)
                .setValue(bytes(env.getOutputBase().getPathString()))
                .build())
        .add(
            PathToReplace.newBuilder()
                .setType(PathToReplace.Type.BUILD_WORKING_DIRECTORY)
                .setValue(bytes(env.getWorkingDirectory().getPathString()))
                .build());
    @Nullable Path workspacePath = env.getWorkspace();
    if (workspacePath != null) {
      pathsToReplace.add(
          PathToReplace.newBuilder()
              .setType(PathToReplace.Type.BUILD_WORKSPACE_DIRECTORY)
              .setValue(bytes(workspacePath.getPathString()))
              .build());
    }
    return pathsToReplace.build();
  }

  /** Converts a string to bytes for use in {@link ExecRequest} bytes fields. */
  public static ByteString bytes(String string) {
    return ByteString.copyFrom(string, ISO_8859_1);
  }

  private PathToReplaceUtils() {}
}
