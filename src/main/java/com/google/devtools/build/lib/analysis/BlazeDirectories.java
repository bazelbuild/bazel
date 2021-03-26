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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Ascii;
import com.google.common.hash.HashCode;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.ArtifactRoot.RootType;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.util.StringCanonicalizer;
import com.google.devtools.build.lib.vfs.Path;
import java.util.Objects;
import javax.annotation.Nullable;

/**
 * Encapsulates the directories related to a workspace.
 *
 * <p>The <code>workspace</code> is the top-level directory in the user's client (possibly
 * read-only). The <code>execRoot</code> is the working directory for all spawned tools, which is
 * generally below the <code>outputBase</code>.
 *
 * <p>Care must be taken to avoid multiple Bazel instances trying to write to the same output
 * directory. At this time, this is enforced by requiring a 1:1 correspondence between a running
 * Bazel instance and an output base directory, though this requirement may be softened in the
 * future.
 *
 * <p>If the user does not qualify an output base directory, the startup code will derive it
 * deterministically from the workspace. Note also that while the Bazel server process runs with the
 * workspace directory as its working directory, the client process may have a different working
 * directory, typically a subdirectory.
 *
 * <p>Do not put shortcuts to specific files here!
 */
@AutoCodec
@Immutable
public final class BlazeDirectories {
  // Include directory name, relative to execRoot/blaze-out/configuration. Only one segment allowed.
  public static final String RELATIVE_INCLUDE_DIR = StringCanonicalizer.intern("include");
  @VisibleForTesting static final String DEFAULT_EXEC_ROOT = "default-exec-root";

  private final ServerDirectories serverDirectories;
  /** Workspace root and server CWD. */
  private final Path workspace;
  /**
   * The root of the user's local JDK install, to be used as the default target javabase and as a
   * fall-back host_javabase. This is not the embedded JDK.
   */
  private final Path defaultSystemJavabase;
  /** The root of all build actions. */
  private final Path blazeExecRoot;

  // These two are kept to avoid creating new objects every time they are accessed. This showed up
  // in a profiler.
  private final Path blazeOutputPath;
  private final Path localOutputPath;
  private final String productName;

  @AutoCodec.Instantiator
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
        this.blazeExecRoot = serverDirectories.getExecRootBase().getChild(DEFAULT_EXEC_ROOT);
      } else {
        this.blazeExecRoot = serverDirectories.getExecRootBase().getChild(workspace.getBaseName());
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

  /** Returns the workspace directory, which is also the working dir of the server. */
  public Path getWorkspace() {
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

  public Path getExecRootBase() {
    return serverDirectories.getExecRootBase();
  }

  /**
   * Returns the execution root of Blaze.
   *
   * @deprecated Avoid using this method as it will only work if your workspace is named like
   *     Google's internal workspace. This method will not work in Bazel. Use {@link
   *     #getExecRoot(String)} instead.
   *     <p><em>AVOID USING THIS METHOD</em>
   */
  @Nullable
  @Deprecated
  public Path getBlazeExecRoot() {
    return blazeExecRoot;
  }

  /**
   * Returns the execution root for a particular repository. This is the directory underneath which
   * Blaze builds the source symlink forest, to represent the merged view of different workspaces
   * specified with --package_path.
   */
  public Path getExecRoot(String workspaceName) {
    return serverDirectories.getExecRootBase().getRelative(workspaceName);
  }

  /**
   * Returns the output path of Blaze.
   *
   * @deprecated Avoid using this method as it will only work if your workspace is named like
   *     Google's internal workspace. This method will not work in Bazel. Use {@link
   *     #getOutputPath(String)} instead.
   *     <p><em>AVOID USING THIS METHOD</em>
   */
  @Nullable
  @Deprecated
  public Path getBlazeOutputPath() {
    return blazeOutputPath;
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

  public Path getPersistentActionOutsDirectory(Path execRoot) {
    return execRoot.getRelative(getRelativeOutputPath()).getRelative("_actions");
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

  public String getRelativeOutputPath() {
    return BlazeDirectories.getRelativeOutputPath(productName);
  }

  public String getProductName() {
    return productName;
  }

  /**
   * Returns the output directory name, relative to the execRoot. TODO(bazel-team): (2011) make this
   * private?
   */
  public static String getRelativeOutputPath(String productName) {
    return StringCanonicalizer.intern(productName + "-out");
  }

  @Override
  public int hashCode() {
    // blazeExecRoot is derivable from other fields, but better safe than sorry.
    return Objects.hash(serverDirectories, workspace, productName);
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (!(obj instanceof BlazeDirectories)) {
      return false;
    }
    BlazeDirectories that = (BlazeDirectories) obj;
    return this.serverDirectories.equals(that.serverDirectories)
        && this.workspace.equals(that.workspace)
        && this.productName.equals(that.productName);
  }
}
