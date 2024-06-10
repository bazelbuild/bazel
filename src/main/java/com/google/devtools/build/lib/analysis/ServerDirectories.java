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

import static com.google.common.base.Preconditions.checkArgument;

import com.google.common.base.Strings;
import com.google.common.hash.HashCode;
import com.google.common.hash.Hashing;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.Root;
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
  @Nullable private final Root virtualSourceRoot; // Null if the source root is not virtualized.

  // TODO(bazel-team): Use a builder to simplify/unify these constructors. This makes it easier to
  // have sensible defaults, e.g. execRootBase = outputBase + "/execroot". Then reorder the fields
  // to be consistent throughout this class.

  public ServerDirectories(
      Path installBase,
      Path outputBase,
      Path outputUserRoot,
      Path execRootBase,
      @Nullable Root virtualSourceRoot,
      @Nullable String installMD5) {
    this.installBase = installBase;
    this.installMD5 = toMD5HashCode(installMD5);
    this.outputBase = outputBase;
    this.outputUserRoot = outputUserRoot;
    this.execRootBase = execRootBase;
    this.virtualSourceRoot = virtualSourceRoot;
  }

  public ServerDirectories(Path installBase, Path outputBase, Path outputUserRoot) {
    this(
        // Some tests set installBase to null.
        // TODO(bazel-team): Be more consistent about whether nulls are permitted. (e.g. equals()
        // presently doesn't tolerate them for some fields). We should probably just disallow them.
        installBase,
        outputBase,
        outputUserRoot,
        outputBase.getRelative(EXECROOT),
        /* virtualSourceRoot= */ null,
        /* installMD5= */ null);
  }

  @Nullable
  private static HashCode toMD5HashCode(@Nullable String installMD5) {
    if (Strings.isNullOrEmpty(installMD5)) {
      return null;
    }
    HashCode hash = HashCode.fromString(installMD5);
    checkArgument(hash.bits() == Hashing.md5().bits(), "Hash '%s' has %s bits", hash, hash.bits());
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
   * <p>By default, this is a folder called {@linkplain #EXECROOT execroot} in {@link
   * #getOutputBase}. However, some {@link com.google.devtools.build.lib.vfs.FileSystem}
   * implementations may choose to virtualize the execroot (in other words, it is not a real on-disk
   * path, but one that the {@link com.google.devtools.build.lib.vfs.FileSystem} recognizes).
   *
   * <p>This is virtual if and only if {@link #getVirtualSourceRoot} is present.
   */
  public Path getExecRootBase() {
    return execRootBase;
  }

  /**
   * Returns a stable, virtual root that (if present) should be used as the effective package path
   * for all commands during the server's lifetime.
   *
   * <p>If present, the server's {@link com.google.devtools.build.lib.vfs.FileSystem} is responsible
   * for translating paths under this root to the actual requested {@code --package_path} for a
   * given command.
   *
   * <p>Present if and only if {@link #getExecRootBase} is virtualized.
   */
  @Nullable
  public Root getVirtualSourceRoot() {
    return virtualSourceRoot;
  }
}
