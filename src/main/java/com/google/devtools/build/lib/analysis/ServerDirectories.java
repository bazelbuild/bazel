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

import com.google.common.base.Strings;
import com.google.common.hash.HashCode;
import com.google.common.hash.Hashing;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;

import javax.annotation.Nullable;

/**
 * Represents the server install directory, which contains the Bazel installation and embedded
 * binaries.
 *
 * <p>The <code>installBase</code> is the directory where the Blaze binary has been installed. The
 * <code>outputBase</code> is the directory below which Blaze puts all its state.
 */
@Immutable
public final class ServerDirectories {
  /** Where Blaze gets unpacked. */
  private final Path installBase;
  /** The content hash of everything in installBase. */
  private final HashCode installMD5;
  /** The root of the temp and output trees. */
  private final Path outputBase;

  public ServerDirectories(Path installBase, Path outputBase, @Nullable String installMD5) {
    this.installBase = installBase;
    this.outputBase = outputBase;
    this.installMD5 =
        Strings.isNullOrEmpty(installMD5) ? null : checkMD5(HashCode.fromString(installMD5));
  }

  public ServerDirectories(Path installBase, Path outputBase) {
    this(installBase, outputBase, null);
  }

  private static HashCode checkMD5(HashCode hash) {
    Preconditions.checkArgument(hash.bits() == Hashing.md5().bits(),
                                "Hash '%s' has %s bits", hash, hash.bits());
    return hash;
  }

  /**
   * Returns the Filesystem that all of our directories belong to. Handy for
   * resolving absolute paths.
   */
  public FileSystem getFileSystem() {
    return installBase.getFileSystem();
  }

  /**
   * Returns the installation base directory. Currently used by info command only.
   */
  public Path getInstallBase() {
    return installBase;
  }

  /**
   * Returns the base of the output tree, which hosts all build and scratch
   * output for a user and workspace.
   */
  public Path getOutputBase() {
    return outputBase;
  }

  /**
   * Returns the installed embedded binaries directory, under the shared
   * installBase location.
   */
  public Path getEmbeddedBinariesRoot() {
    return installBase.getChild("_embedded_binaries");
  }

 /**
  * Returns the MD5 content hash of the blaze binary (includes deploy JAR, embedded binaries, and
  * anything else that ends up in the install_base).
  */
  public HashCode getInstallMD5() {
    return installMD5;
  }
}