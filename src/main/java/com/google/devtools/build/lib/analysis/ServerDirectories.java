// Copyright 2016 The Bazel Authors. All rights reserved.
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
import com.google.common.base.Preconditions;
import com.google.common.base.Strings;
import com.google.common.hash.HashCode;
import com.google.common.hash.Hashing;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.vfs.Path;
import java.util.Objects;
import javax.annotation.Nullable;

/**
 * Represents the relevant directories for the server: the location of the embedded binaries and the
 * output directories.
 */
@Immutable
public final class ServerDirectories {
  public static final String EXECROOT = "execroot";

  /** Where Bazel gets unpacked. */
  private final Path installBase;
  /** The content hash of everything in installBase. */
  @Nullable private final HashCode installMD5;

  /** The root of the temp and output trees. */
  private final Path outputBase;

  /** Top-level user output directory; used, e.g., as default location for caches. */
  private final Path outputUserRoot;

  private final Path execRootBase;

  // TODO(bazel-team): Use a builder to simplify/unify these constructors. This makes it easier to
  // have sensible defaults, e.g. execRootBase = outputBase + "/execroot". Then reorder the fields
  // to be consistent throughout this class.

  public ServerDirectories(
      Path installBase,
      Path outputBase,
      Path outputUserRoot,
      Path execRootBase,
      @Nullable String installMD5) {
    this.installBase = installBase;
    this.installMD5 =
        Strings.isNullOrEmpty(installMD5) ? null : checkMD5(HashCode.fromString(installMD5));
    this.outputBase = outputBase;
    this.outputUserRoot = outputUserRoot;
    this.execRootBase = execRootBase;
  }

  public ServerDirectories(Path installBase, Path outputBase, Path outputUserRoot) {
    this(
        // Some tests set installBase to null.
        // TODO(bazel-team): Be more consistent about whether nulls are permitted. (e.g. equals()
        // presently doesn't tolerate them for some fields). We should probably just disallow them.
        installBase, outputBase, outputUserRoot, outputBase.getRelative(EXECROOT), null);
  }

  private static HashCode checkMD5(HashCode hash) {
    Preconditions.checkArgument(
        hash.bits() == Hashing.md5().bits(), "Hash '%s' has %s bits", hash, hash.bits());
    return hash;
  }

  /** Returns the installation base directory. */
  public Path getInstallBase() {
    return installBase;
  }

  /**
   * Returns the MD5 content hash of the blaze binary (includes deploy JAR, embedded binaries, and
   * anything else that ends up in the install_base).
   */
  public HashCode getInstallMD5() {
    return installMD5;
  }

  /**
   * Returns the base of the output tree, which hosts all build and scratch output for a user and
   * workspace.
   */
  public Path getOutputBase() {
    return outputBase;
  }

  /**
   * Returns the root directory for user output. In particular default caches will be located here.
   */
  public Path getOutputUserRoot() {
    return outputUserRoot;
  }

  /**
   * Parent of all execution roots.
   *
   * <p>This is physically, always /outputbase/execroot, but might be virtualized.
   */
  public Path getExecRootBase() {
    return execRootBase;
  }

  /** Returns the installed embedded binaries directory, under the shared installBase location. */
  public Path getEmbeddedBinariesRoot() {
    return getEmbeddedBinariesRoot(installBase);
  }

  @VisibleForTesting
  public static Path getEmbeddedBinariesRoot(Path installBase) {
    return installBase;
  }

  @Override
  public int hashCode() {
    return Objects.hash(installBase, installMD5, outputBase, outputUserRoot, execRootBase);
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (!(obj instanceof ServerDirectories)) {
      return false;
    }
    ServerDirectories that = (ServerDirectories) obj;
    return this.installBase.equals(that.installBase)
        && Objects.equals(this.installMD5, that.installMD5)
        && this.outputBase.equals(that.outputBase)
        && this.outputUserRoot.equals(that.outputUserRoot)
        && this.execRootBase.equals(that.execRootBase);
  }
}
