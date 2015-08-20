// Copyright 2014 Google Inc. All rights reserved.
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

import com.google.common.base.Optional;
import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.packages.PackageIdentifier;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import java.util.Objects;

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

  enum ErrorReason {
    // There is no BUILD file.
    NO_BUILD_FILE,

    // The package name is invalid.
    INVALID_PACKAGE_NAME,

    // The package is considered deleted because of --deleted_packages.
    DELETED_PACKAGE,

    // The //external package could not be loaded, either because the WORKSPACE file could not be
    // parsed or the packages it references cannot be loaded.
    NO_EXTERNAL_PACKAGE
  }

  protected PackageLookupValue() {
  }

  public static PackageLookupValue success(Path root) {
    return new SuccessfulPackageLookupValue(root);
  }

  public static PackageLookupValue overlaidBuildFile(
      Path root, Optional<FileValue> overlaidBuildFile) {
    return new OverlaidPackageLookupValue(root, overlaidBuildFile);
  }

  public static PackageLookupValue workspace(Path root) {
    return new WorkspacePackageLookupValue(root);
  }

  public static PackageLookupValue noBuildFile() {
    return NoBuildFilePackageLookupValue.INSTANCE;
  }

  public static PackageLookupValue noExternalPackage() {
    return NoExternalPackageLookupValue.INSTANCE;
  }

  public static PackageLookupValue invalidPackageName(String errorMsg) {
    return new InvalidNamePackageLookupValue(errorMsg);
  }

  public static PackageLookupValue deletedPackage() {
    return DeletedPackageLookupValue.INSTANCE;
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
    return new SkyKey(SkyFunctions.PACKAGE_LOOKUP, pkgIdentifier);
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

  /**
   * A package under external/ that has a BUILD file that is not under external/.
   *
   * <p>This is kind of a hack to get around our assumption that external/ is immutable.</p>
   */
  private static class OverlaidPackageLookupValue extends SuccessfulPackageLookupValue {

    private final Optional<FileValue> overlaidBuildFile;

    public OverlaidPackageLookupValue(Path root, Optional<FileValue> overlaidBuildFile) {
      super(root);
      this.overlaidBuildFile = overlaidBuildFile;
    }

    @Override
    public boolean equals(Object obj) {
      if (!(obj instanceof OverlaidPackageLookupValue)) {
        return false;
      }
      OverlaidPackageLookupValue other = (OverlaidPackageLookupValue) obj;
      return super.equals(other) && overlaidBuildFile.equals(other.overlaidBuildFile);
    }

    @Override
    public int hashCode() {
      return Objects.hash(super.hashCode(), overlaidBuildFile);
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

  private static class NoBuildFilePackageLookupValue extends UnsuccessfulPackageLookupValue {

    public static final NoBuildFilePackageLookupValue INSTANCE =
        new NoBuildFilePackageLookupValue();

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

  private static class NoExternalPackageLookupValue extends UnsuccessfulPackageLookupValue {

    public static final NoExternalPackageLookupValue INSTANCE =
        new NoExternalPackageLookupValue();

    private NoExternalPackageLookupValue() {
    }

    @Override
    ErrorReason getErrorReason() {
      return ErrorReason.NO_EXTERNAL_PACKAGE;
    }

    @Override
    String getErrorMsg() {
      return "Error loading the //external package";
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

  private static class DeletedPackageLookupValue extends UnsuccessfulPackageLookupValue {

    public static final DeletedPackageLookupValue INSTANCE = new DeletedPackageLookupValue();

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
