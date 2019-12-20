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
import com.google.common.collect.Interner;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.packages.BuildFileName;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.AbstractSkyKey;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import javax.annotation.Nullable;

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

  @AutoCodec
  public static final NoBuildFilePackageLookupValue NO_BUILD_FILE_VALUE =
      new NoBuildFilePackageLookupValue();

  @AutoCodec
  public static final DeletedPackageLookupValue DELETED_PACKAGE_VALUE =
      new DeletedPackageLookupValue();

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

  protected PackageLookupValue() {}

  public static PackageLookupValue success(
      RepositoryValue repository, Root root, BuildFileName buildFileName) {
    return new SuccessfulPackageLookupValue(repository, root, buildFileName);
  }

  public static PackageLookupValue success(Root root, BuildFileName buildFileName) {
    return new SuccessfulPackageLookupValue(null, root, buildFileName);
  }

  public static PackageLookupValue invalidPackageName(String errorMsg) {
    return new InvalidNamePackageLookupValue(errorMsg);
  }

  public static PackageLookupValue incorrectRepositoryReference(
      PackageIdentifier invalidPackage, PackageIdentifier correctPackage) {
    return new IncorrectRepositoryReferencePackageLookupValue(invalidPackage, correctPackage);
  }

  /**
   * For a successful package lookup, returns the root (package path entry) that the package resides
   * in.
   */
  public abstract Root getRoot();

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

  public static Key key(PackageIdentifier pkgIdentifier) {
    Preconditions.checkArgument(!pkgIdentifier.getRepository().isDefault());
    return Key.create(pkgIdentifier);
  }

  /** {@link SkyKey} for {@link PackageLookupValue} computation. */
  @AutoCodec.VisibleForSerialization
  @AutoCodec
  public static class Key extends AbstractSkyKey<PackageIdentifier> {
    private static final Interner<Key> interner = BlazeInterners.newWeakInterner();

    private Key(PackageIdentifier arg) {
      super(arg);
    }

    @AutoCodec.VisibleForSerialization
    @AutoCodec.Instantiator
    static Key create(PackageIdentifier arg) {
      return interner.intern(new Key(arg));
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.PACKAGE_LOOKUP;
    }
  }

  /** Successful lookup value. */
  public static class SuccessfulPackageLookupValue extends PackageLookupValue {
    /**
     * The repository value the meaning of the path depends on (e.g., an external repository
     * controlling a symbolic link the path goes trough). Can be {@code null}, if does not depend on
     * such a repository; will always be {@code null} for packages in the main repository.
     */
    @Nullable private final RepositoryValue repository;

    private final Root root;
    private final BuildFileName buildFileName;

    SuccessfulPackageLookupValue(
        @Nullable RepositoryValue repository, Root root, BuildFileName buildFileName) {
      this.repository = repository;
      this.root = root;
      this.buildFileName = buildFileName;
    }

    @Nullable
    public RepositoryValue repository() {
      return repository;
    }

    @Override
    public boolean packageExists() {
      return true;
    }

    @Override
    public Root getRoot() {
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
      return root.equals(other.root)
          && buildFileName == other.buildFileName
          && Objects.equal(repository, other.repository);
    }

    @Override
    public int hashCode() {
      return Objects.hashCode(root.hashCode(), buildFileName.hashCode(), repository);
    }
  }

  private abstract static class UnsuccessfulPackageLookupValue extends PackageLookupValue {

    @Override
    public boolean packageExists() {
      return false;
    }

    @Override
    public Root getRoot() {
      throw new IllegalStateException();
    }

    @Override
    public BuildFileName getBuildFileName() {
      throw new IllegalStateException();
    }
  }

  /** Marker value for no build file found. */
  public static class NoBuildFilePackageLookupValue extends UnsuccessfulPackageLookupValue {

    private NoBuildFilePackageLookupValue() {}

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

    InvalidNamePackageLookupValue(String errorMsg) {
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

    IncorrectRepositoryReferencePackageLookupValue(
        PackageIdentifier invalidPackageIdentifier, PackageIdentifier correctedPackageIdentifier) {
      this.invalidPackageIdentifier = invalidPackageIdentifier;
      this.correctedPackageIdentifier = correctedPackageIdentifier;
    }

    PackageIdentifier getInvalidPackageIdentifier() {
      return invalidPackageIdentifier;
    }

    PackageIdentifier getCorrectedPackageIdentifier() {
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
    private DeletedPackageLookupValue() {}

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
   * Value for repository we could not find. This can happen when looking for a label that specifies
   * a non-existent repository.
   */
  public static class NoRepositoryPackageLookupValue extends UnsuccessfulPackageLookupValue {
    private final String repositoryName;

    NoRepositoryPackageLookupValue(String repositoryName) {
      this.repositoryName = repositoryName;
    }

    @Override
    ErrorReason getErrorReason() {
      return ErrorReason.REPOSITORY_NOT_FOUND;
    }

    @Override
    public String getErrorMsg() {
      return String.format("The repository '%s' could not be resolved", repositoryName);
    }
  }
}
