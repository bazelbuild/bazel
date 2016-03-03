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
package com.google.devtools.build.lib.skyframe;

import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

/**
 * A value that represents a package lookup result.
 *
 * <p>Package lookups will always produce a value. On success, the {@code #getRoot} returns the
 * package path root under which the package resides and the package's BUILD file is guaranteed to
 * exist (unless this is looking up a WORKSPACE file, in which case the underlying file may or may
 * not exist. On failure, {@code #getErrorReason} and {@code #getErrorMsg} describe why the package
 * doesn't exist.
 *
 * <p>Implementation detail: we use inheritance here to optimize for memory usage.
 */
public abstract class PackageLookupValue implements SkyValue {

  public static final NoBuildFilePackageLookupValue NO_BUILD_FILE_VALUE =
      new NoBuildFilePackageLookupValue();
  public static final DeletedPackageLookupValue DELETED_PACKAGE_VALUE =
      new DeletedPackageLookupValue();

  enum ErrorReason {
    // There is no BUILD file.
    NO_BUILD_FILE,

    // The package name is invalid.
    INVALID_PACKAGE_NAME,

    // The package is considered deleted because of --deleted_packages.
    DELETED_PACKAGE
  }

  protected PackageLookupValue() {
  }

  public static PackageLookupValue success(Path root) {
    return new SuccessfulPackageLookupValue(root);
  }

  public static PackageLookupValue workspace(Path root) {
    return new WorkspacePackageLookupValue(root);
  }

  public static PackageLookupValue invalidPackageName(String errorMsg) {
    return new InvalidNamePackageLookupValue(errorMsg);
  }

  public boolean isExternalPackage() {
    return false;
  }

  /**
   * For a successful package lookup, returns the root (package path entry) that the package
   * resides in.
   */
  public abstract Path getRoot();

  /**
   * Returns whether the package lookup was successful.
   */
  public abstract boolean packageExists();

  /**
   * For an unsuccessful package lookup, gets the reason why {@link #packageExists} returns
   * {@code false}.
   */
  abstract ErrorReason getErrorReason();

  /**
   * For an unsuccessful package lookup, gets a detailed error message for {@link #getErrorReason}
   * that is suitable for reporting to a user.
   */
  abstract String getErrorMsg();

  static SkyKey key(PathFragment directory) {
    Preconditions.checkArgument(!directory.isAbsolute(), directory);
    return key(PackageIdentifier.createInDefaultRepo(directory));
  }

  public static SkyKey key(PackageIdentifier pkgIdentifier) {
    return SkyKey.create(SkyFunctions.PACKAGE_LOOKUP, pkgIdentifier);
  }

  private static class SuccessfulPackageLookupValue extends PackageLookupValue {

    private final Path root;

    private SuccessfulPackageLookupValue(Path root) {
      this.root = root;
    }

    @Override
    public boolean packageExists() {
      return true;
    }

    @Override
    public Path getRoot() {
      return root;
    }

    @Override
    ErrorReason getErrorReason() {
      throw new IllegalStateException();
    }

    @Override
    String getErrorMsg() {
      throw new IllegalStateException();
    }

    @Override
    public boolean equals(Object obj) {
      if (!(obj instanceof SuccessfulPackageLookupValue)) {
        return false;
      }
      SuccessfulPackageLookupValue other = (SuccessfulPackageLookupValue) obj;
      return root.equals(other.root);
    }

    @Override
    public int hashCode() {
      return root.hashCode();
    }
  }

  // TODO(kchodorow): fix these semantics.  This class should not exist, WORKSPACE lookup should
  // just return success/failure like a "normal" package.
  private static class WorkspacePackageLookupValue extends SuccessfulPackageLookupValue {

    private WorkspacePackageLookupValue(Path root) {
      super(root);
    }

    // TODO(kchodorow): get rid of this, the semantics are wrong (successful package lookup should
    // mean the package exists).
    @Override
    public boolean packageExists() {
      return getRoot().exists();
    }

    @Override
    public boolean isExternalPackage() {
      return true;
    }
  }

  private abstract static class UnsuccessfulPackageLookupValue extends PackageLookupValue {

    @Override
    public boolean packageExists() {
      return false;
    }

    @Override
    public Path getRoot() {
      throw new IllegalStateException();
    }
  }

  /** Marker value for no build file found. */
  public static class NoBuildFilePackageLookupValue extends UnsuccessfulPackageLookupValue {

    private NoBuildFilePackageLookupValue() {
    }

    @Override
    ErrorReason getErrorReason() {
      return ErrorReason.NO_BUILD_FILE;
    }

    @Override
    String getErrorMsg() {
      return "BUILD file not found on package path";
    }
  }

  private static class InvalidNamePackageLookupValue extends UnsuccessfulPackageLookupValue {

    private final String errorMsg;

    private InvalidNamePackageLookupValue(String errorMsg) {
      this.errorMsg = errorMsg;
    }

    @Override
    ErrorReason getErrorReason() {
      return ErrorReason.INVALID_PACKAGE_NAME;
    }

    @Override
    String getErrorMsg() {
      return errorMsg;
    }

    @Override
    public boolean equals(Object obj) {
      if (!(obj instanceof InvalidNamePackageLookupValue)) {
        return false;
      }
      InvalidNamePackageLookupValue other = (InvalidNamePackageLookupValue) obj;
      return errorMsg.equals(other.errorMsg);
    }

    @Override
    public int hashCode() {
      return errorMsg.hashCode();
    }
  }

  /** Marker value for a deleted package. */
  public static class DeletedPackageLookupValue extends UnsuccessfulPackageLookupValue {

    private DeletedPackageLookupValue() {
    }

    @Override
    ErrorReason getErrorReason() {
      return ErrorReason.DELETED_PACKAGE;
    }

    @Override
    String getErrorMsg() {
      return "Package is considered deleted due to --deleted_packages";
    }
  }
}
