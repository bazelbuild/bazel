// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.actions;

/**
 * A flag to enable / disable reporting source directories as directories via {@link
 * Artifact.SourceArtifact#isDirectory()}, and therefore the Starlark {@code File.is_directory}
 * field. Uses a system property which can be set via a startup flag ({@code
 * --host_jvm_args=-DBAZEL_SOURCE_DIRECTORY_IS_DIRECTORY=1}). This ensures that toggling the flag
 * causes a server restart and discards Skyframe state, which matters because {@code isDirectory()}
 * is consulted throughout analysis, execution and remote code that has no access to {@code
 * StarlarkSemantics}.
 *
 * <p>When enabled, input file targets take a Skyframe dependency on their {@code FileValue}, which
 * supplies the type without additional filesystem access and re-analyzes the target when the type
 * changes.
 *
 * <p>Off by default, since reporting source directories as directories is a breaking change.
 * Enable together with {@code TrackSourceDirectoriesFlag}: without it, {@code ArtifactFunction}
 * produces regular-file metadata for source directories.
 */
public final class SourceDirectoryIsDirectoryFlag {
  private static final boolean SOURCE_DIRECTORY_IS_DIRECTORY =
      System.getProperty("BAZEL_SOURCE_DIRECTORY_IS_DIRECTORY", "").equals("1");

  public static boolean sourceDirectoryIsDirectory() {
    return SOURCE_DIRECTORY_IS_DIRECTORY;
  }

  private SourceDirectoryIsDirectoryFlag() {}
}
