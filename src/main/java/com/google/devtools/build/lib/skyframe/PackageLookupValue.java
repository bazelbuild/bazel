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

import com.google.common.base.Objects;
import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.packages.BuildFileName;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.LegacySkyKey;
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
  public static final NoRepositoryPackageLookupValue NO_SUCH_REPOSITORY_VALUE =
      new NoRepositoryPackageLookupValue();

  enum ErrorReason {
    /** There is no BUILD file. */
    NO_BUILD_FILE,

    /** The package name is invalid. */
    INVALID_PACKAGE_NAME,

    /** The package is considered deleted because of --deleted_packages. */
    DELETED_PACKAGE,

    /** The repository was not found. */
    REPOSITORY_NOT_FOUND
  }

  protected PackageLookupValue() {
  }

  public static PackageLookupValue success(Path root, BuildFileName buildFileName) {
    return new SuccessfulPackageLookupValue(root, buildFileName);
  }

  public static PackageLookupValue invalidPackageName(String errorMsg) {
    return new InvalidNamePackageLookupValue(errorMsg);
  }

  public static PackageLookupValue incorrectRepositoryReference(
      PackageIdentifier invalidPackage, PackageIdentifier correctPackage) {
    return new IncorrectRepositoryReferencePackageLookupValue(invalidPackage, correctPackage);
  }

  /**
   * For a successful package lookup, returns the root (package path entry) that the package
   * resides in.
   */
  public abstract Path getRoot();

  /** For a successful package lookup, returns the build file name that the package uses. */
  public abstract BuildFileName getBuildFileName();

  /** Returns whether the package lookup was successful. */
  public abstract boolean packageExists();

  /**
   * For a successful package lookup, returns the {@link RootedPath} for the build file that defines
   * the package.
   */
  public RootedPath getRootedPath(PackageIdentifier packageIdentifier) {
    return RootedPath.toRootedPath(
        getRoot(), getBuildFileName().getBuildFileFragment(packageIdentifier));
  }

  /**
   * For an unsuccessful package lookup, gets the reason why {@link #packageExists} returns {@code
   * false}.
   */
  abstract ErrorReason getErrorReason();

  /**
   * For an unsuccessful package lookup, gets a detailed error message for {@link #getErrorReason}
   * that is suitable for reporting to a user.
   */
  public abstract String getErrorMsg();

  public static SkyKey key(PathFragment directory) {
    Preconditions.checkArgument(!directory.isAbsolute(), directory);
    return key(PackageIdentifier.createInMainRepo(directory));
  }

  public static SkyKey key(PackageIdentifier pkgIdentifier) {
    Preconditions.checkArgument(!pkgIdentifier.getRepository().isDefault());
    return LegacySkyKey.create(SkyFunctions.PACKAGE_LOOKUP, pkgIdentifier);
  }

  /** Successful lookup value. */
  public static class SuccessfulPackageLookupValue extends PackageLookupValue {

    private final Path root;
    private final BuildFileName buildFileName;

    private SuccessfulPackageLookupValue(Path root, BuildFileName buildFileName) {
      this.root = root;
      this.buildFileName = buildFileName;
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
    public BuildFileName getBuildFileName() {
      return buildFileName;
    }

    @Override
    ErrorReason getErrorReason() {
      throw new IllegalStateException();
    }

    @Override
    public String getErrorMsg() {
      throw new IllegalStateException();
    }

    @Override
    public boolean equals(Object obj) {
      if (!(obj instanceof SuccessfulPackageLookupValue)) {
        return false;
      }
      SuccessfulPackageLookupValue other = (SuccessfulPackageLookupValue) obj;
      return root.equals(other.root) && buildFileName == other.buildFileName;
    }

    @Override
    public int hashCode() {
      return Objects.hashCode(root.hashCode(), buildFileName.hashCode());
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

    @Override
    public BuildFileName getBuildFileName() {
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
    public String getErrorMsg() {
      return "BUILD file not found on package path";
    }
  }

  /** Value indicating the package name was in error. */
  public static class InvalidNamePackageLookupValue extends UnsuccessfulPackageLookupValue {

    private final String errorMsg;

    private InvalidNamePackageLookupValue(String errorMsg) {
      this.errorMsg = errorMsg;
    }

    @Override
    ErrorReason getErrorReason() {
      return ErrorReason.INVALID_PACKAGE_NAME;
    }

    @Override
    public String getErrorMsg() {
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

    @Override
    public String toString() {
      return String.format("%s: %s", this.getClass().getSimpleName(), this.errorMsg);
    }
  }

  /** Value indicating the package name was in error. */
  public static class IncorrectRepositoryReferencePackageLookupValue
      extends UnsuccessfulPackageLookupValue {

    private final PackageIdentifier invalidPackageIdentifier;
    private final PackageIdentifier correctedPackageIdentifier;

    private IncorrectRepositoryReferencePackageLookupValue(
        PackageIdentifier invalidPackageIdentifier, PackageIdentifier correctedPackageIdentifier) {
      this.invalidPackageIdentifier = invalidPackageIdentifier;
      this.correctedPackageIdentifier = correctedPackageIdentifier;
    }

    public PackageIdentifier getInvalidPackageIdentifier() {
      return invalidPackageIdentifier;
    }

    public PackageIdentifier getCorrectedPackageIdentifier() {
      return correctedPackageIdentifier;
    }

    @Override
    ErrorReason getErrorReason() {
      return ErrorReason.INVALID_PACKAGE_NAME;
    }

    @Override
    public String getErrorMsg() {
      return String.format(
          "Invalid package reference %s crosses into repository %s:"
              + " did you mean to use %s instead?",
          invalidPackageIdentifier,
          correctedPackageIdentifier.getRepository(),
          correctedPackageIdentifier);
    }

    @Override
    public boolean equals(Object obj) {
      if (!(obj instanceof IncorrectRepositoryReferencePackageLookupValue)) {
        return false;
      }
      IncorrectRepositoryReferencePackageLookupValue other =
          (IncorrectRepositoryReferencePackageLookupValue) obj;
      return Objects.equal(invalidPackageIdentifier, other.invalidPackageIdentifier)
          && Objects.equal(correctedPackageIdentifier, other.correctedPackageIdentifier);
    }

    @Override
    public int hashCode() {
      return Objects.hashCode(invalidPackageIdentifier, correctedPackageIdentifier);
    }

    @Override
    public String toString() {
      return String.format(
          "%s: invalidPackageIdenfitier: %s, corrected: %s",
          this.getClass().getSimpleName(),
          this.invalidPackageIdentifier,
          this.correctedPackageIdentifier);
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
    public String getErrorMsg() {
      return "Package is considered deleted due to --deleted_packages";
    }
  }

  /**
   * Marker value for repository we could not find. This can happen when looking for a label that
   * specifies a non-existent repository.
   */
  public static class NoRepositoryPackageLookupValue extends UnsuccessfulPackageLookupValue {

    private NoRepositoryPackageLookupValue() {}

    @Override
    ErrorReason getErrorReason() {
      return ErrorReason.REPOSITORY_NOT_FOUND;
    }

    @Override
    public String getErrorMsg() {
      return "The repository could not be resolved";
    }
  }
}
