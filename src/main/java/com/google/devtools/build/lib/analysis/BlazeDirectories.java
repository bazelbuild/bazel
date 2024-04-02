// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis;

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.base.Ascii;
import com.google.common.hash.HashCode;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.ArtifactRoot.RootType;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.util.StringCanonicalizer;
import com.google.devtools.build.lib.vfs.Path;
import javax.annotation.Nullable;

/**
 * Encapsulates the directories related to a workspace.
 *
 * <p>A {@code workspace>} is a directory tree containing the source files you want to build.
 *
 * <p>The {@code workspace Path} object this class stores is the workspace's root directory, which
 * contains a {@code WORKSPACE} file that marks and configures the workspace. When you build {@code
 * //my:project}, this signifies a target named {@code project} in a {@code BUILD} file in the
 * {@code my} subdirectory under the workspace root. You can find the workspace root directory by
 * running {@code $ bazel info | grep workspace}.
 *
 * <p>The {@code outputBase} is where all workspace output is written. This includes both build
 * outputs and internal files Bazel uses to support builds (like the action cache, log files, and
 * external repository mappings). This path is only meaningful for core Bazel devs: it's not part of
 * the public user API. This path is not under the workspace root (since its purpose isn't to host
 * workspace source files). This appears as {@code _bazel_$USER/$SOME_HASH/} under some local file
 * system root. Exact paths vary depending on what machine you're running Bazel on. You can find
 * this path by running {@code $ bazel info | grep output_base}.
 *
 * <p>The {@code execRoot} is the working directory for all spawned tools. It includes both the
 * subdirectory where Bazel writes build outputs (the {@code outputPath}) and the symlink forest
 * Bazel constructs to map workspace source files the spawned tool can access when it runs. It
 * generally looks like {@code $OUTPUT_BASE/execroot/$WORKSPACE_IDENTIFIER}. You can find this path
 * by running {@code $ bazel info | grep execution_root}.
 *
 * <p>The {@code outputPath} (confusingly similar name to {@code outputBase}, alas) is the root path
 * where Bazel writes build outputs. In other words, any action transforming a source file into a
 * generated output writes that output under this path. It generally looks like {@code
 * $OUTPUT_BASE/execroot/$WORKSPACE_IDENTIFIER/bazel-out}. You can find this path by running {@code
 * $ bazel info | grep output_path}.
 *
 * <p>Care must be taken to avoid multiple Bazel instances trying to write to the same output tree.
 * This is enforced by requiring a 1:1 correspondence between a running Bazel instance and an output
 * base.
 *
 * <p>If the user does not qualify an output base directory, the startup code will derive it
 * deterministically from the workspace. Note also that while the Bazel server process runs with the
 * workspace directory as its working directory, the client process may have a different working
 * directory, typically a subdirectory.
 *
 * <p>Do not put shortcuts to specific files here!
 */
@Immutable
public final class BlazeDirectories {
  private static final String DEFAULT_EXEC_ROOT = "default-exec-root";

  private final ServerDirectories serverDirectories;
  /** Workspace root and server CWD. */
  private final Path workspace;
  /**
   * The root of the user's local JDK install, to be used as the default target javabase and as a
   * fall-back host_javabase. This is not the embedded JDK.
   */
  private final Path defaultSystemJavabase;
  private final Path blazeExecRoot;

  // These two are kept to avoid creating new objects every time they are accessed. This showed up
  // in a profiler.
  private final Path blazeOutputPath;
  private final Path localOutputPath;
  private final String productName;

  public BlazeDirectories(
      ServerDirectories serverDirectories,
      Path workspace,
      Path defaultSystemJavabase,
      String productName) {
    this.serverDirectories = serverDirectories;
    this.workspace = workspace;
    this.defaultSystemJavabase = defaultSystemJavabase;
    this.productName = productName;
    Path outputBase = serverDirectories.getOutputBase();
    if (Ascii.equalsIgnoreCase(productName, "blaze")) {
      boolean useDefaultExecRootName =
          this.workspace == null || this.workspace.getParentDirectory() == null;
      if (useDefaultExecRootName) {
        // TODO(bazel-team): if workspace is null execRoot should be null, but at the moment there
        // is a lot of code that depends on it being non-null.
        this.blazeExecRoot =
            outputBase.getChild(ServerDirectories.EXECROOT).getChild(DEFAULT_EXEC_ROOT);
      } else {
        this.blazeExecRoot =
            outputBase.getChild(ServerDirectories.EXECROOT).getChild(workspace.getBaseName());
      }
      this.blazeOutputPath = blazeExecRoot.getRelative(getRelativeOutputPath());
    } else {
      this.blazeExecRoot = null;
      this.blazeOutputPath = null;
    }
    this.localOutputPath = outputBase.getRelative(getRelativeOutputPath());
  }

  public ServerDirectories getServerDirectories() {
    return serverDirectories;
  }

  /** Returns the installation base directory. */
  public Path getInstallBase() {
    return serverDirectories.getInstallBase();
  }

  /**
   * Returns the workspace directory to use for build artifacts.
   *
   * <p>It may effectively differ from the working directory. Please use {@link
   * #getWorkingDirectory()} for writes within the working directory.
   */
  @Nullable
  public Path getWorkspace() {
    // Make sure to use the same file system as exec root.
    return workspace != null
        ? getExecRootBase().getFileSystem().getPath(workspace.asFragment())
        : null;
  }

  /** Returns working directory of the server. */
  public Path getWorkingDirectory() {
    return workspace;
  }

  /** Returns the root of the user's local JDK install (not the embedded JDK). */
  public Path getLocalJavabase() {
    return defaultSystemJavabase;
  }

  /** Returns if the workspace directory is a valid workspace. */
  public boolean inWorkspace() {
    return this.workspace != null;
  }

  /**
   * Returns the base of the output tree, which hosts all build and scratch output for a user and
   * workspace.
   */
  public Path getOutputBase() {
    return serverDirectories.getOutputBase();
  }

  /** Returns the effective execution root, which may be virtualized. */
  public Path getExecRootBase() {
    return serverDirectories.getExecRootBase();
  }

  /**
   * Returns the local execution root of Google-internal Blaze. Virtualization is not respected.
   *
   * <p>This method throws {@link NullPointerException} in Bazel. Use {@link #getExecRoot} instead.
   */
  public Path getBlazeExecRoot() {
    return checkNotNull(blazeExecRoot, "No Blaze exec root in Bazel");
  }

  /**
   * Returns the execution root for a particular repository. This is the directory underneath which
   * Blaze builds the source symlink forest, to represent the merged view of different workspaces
   * specified with --package_path.
   */
  public Path getExecRoot(String workspaceName) {
    return getExecRootBase().getRelative(workspaceName);
  }

  /**
   * Returns the local output path of Google-internal Blaze. Virtualization is not respected.
   *
   * <p>This method throws {@link NullPointerException} in Bazel. Use {@link #getOutputPath}
   * instead.
   */
  public Path getBlazeOutputPath() {
    return checkNotNull(blazeOutputPath, "No Blaze output path in Bazel");
  }

  /** Returns the output path used by this Blaze instance. */
  public Path getOutputPath(String workspaceName) {
    return getExecRoot(workspaceName).getRelative(getRelativeOutputPath());
  }

  /** Returns the local output path used by this Blaze instance. */
  public Path getLocalOutputPath() {
    return localOutputPath;
  }

  /**
   * Returns the directory where actions can store temporary files (such as their stdout and stderr)
   * during a build. If the directory already exists, the directory is cleaned.
   */
  public Path getActionTempsDirectory(Path execRoot) {
    return execRoot.getRelative(getRelativeOutputPath()).getRelative("_tmp/actions");
  }

  /** Returns the installed embedded binaries directory, under the shared installBase location. */
  public Path getEmbeddedBinariesRoot() {
    return serverDirectories.getEmbeddedBinariesRoot();
  }

  /**
   * Returns the configuration-independent root where the build-data should be placed, given the
   * {@link BlazeDirectories} of this server instance. Nothing else should be placed here.
   */
  public ArtifactRoot getBuildDataDirectory(String workspaceName) {
    return ArtifactRoot.asDerivedRoot(
        getExecRoot(workspaceName), RootType.Output, getRelativeOutputPath(productName));
  }

  /**
   * Returns the MD5 content hash of the blaze binary (includes deploy JAR, embedded binaries, and
   * anything else that ends up in the install_base).
   */
  public HashCode getInstallMD5() {
    return serverDirectories.getInstallMD5();
  }

  /**
   * Returns the directory where Bazel writes build outputs, relative to the execRoot.
   *
   * <p>For example: {@code "bazel-out"}.
   */
  public String getRelativeOutputPath() {
    return BlazeDirectories.getRelativeOutputPath(productName);
  }

  /**
   * Returns the directory where Bazel writes build outputs, relative to the execRoot.
   *
   * <p>For example: {@code "bazel-out"}.
   */
  public static String getRelativeOutputPath(String productName) {
    return StringCanonicalizer.intern(productName + "-out");
  }

  public String getProductName() {
    return productName;
  }
}
