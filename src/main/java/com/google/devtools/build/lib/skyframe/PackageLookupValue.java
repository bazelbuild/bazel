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
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.packages.BuildFileName;
import com.google.devtools.build.lib.rules.repository.RepositoryDirectoryValue;
import com.google.devtools.build.lib.skyframe.serialization.VisibleForSerialization;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.SerializationConstant;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.AbstractSkyKey;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.function.Predicate;
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

  @SerializationConstant
  public static final NoBuildFilePackageLookupValue NO_BUILD_FILE_VALUE =
      new NoBuildFilePackageLookupValue();

  @SerializationConstant
  public static final DeletedPackageLookupValue DELETED_PACKAGE_VALUE =
      new DeletedPackageLookupValue();

  @SerializationConstant
  public static final InvalidProjectLookupValue INVALID_PROJECT_VALUE =
      new InvalidProjectLookupValue();

  enum ErrorReason {
    /** There is no BUILD file. */
    NO_BUILD_FILE,

    /** The package name is invalid. */
    INVALID_PACKAGE_NAME,

    /** The package is considered deleted because of --deleted_packages. */
    DELETED_PACKAGE,

    /** The repository was not found. */
    REPOSITORY_NOT_FOUND,

    /** The project file isn't actually a file. */
    INVALID_PROJECT_FILE
  }

  protected PackageLookupValue() {}

  public static PackageLookupValue success(
      RepositoryDirectoryValue repository, Root root, BuildFileName buildFileName) {
    return new SuccessfulPackageLookupValue(
        repository, root, buildFileName, /* hasProjectFile= */ false);
  }

  public static PackageLookupValue success(Root root, BuildFileName buildFileName) {
    return success(root, buildFileName, /* hasProjectFile= */ false);
  }

  public static PackageLookupValue success(
      Root root, BuildFileName buildFileName, boolean hasProjectFile) {
    return new SuccessfulPackageLookupValue(null, root, buildFileName, hasProjectFile);
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

  public abstract boolean hasProjectFile();

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
    return Key.create(pkgIdentifier);
  }

  static boolean appliesToKey(SkyKey key, Predicate<PackageIdentifier> identifierPredicate) {
    return SkyFunctions.PACKAGE_LOOKUP.equals(key.functionName())
        && identifierPredicate.test((PackageIdentifier) key.argument());
  }

  /** {@link SkyKey} for {@link PackageLookupValue} computation. */
  @VisibleForSerialization
  @AutoCodec
  static class Key extends AbstractSkyKey<PackageIdentifier> {
    private static final SkyKeyInterner<Key> interner = SkyKey.newInterner();

    private Key(PackageIdentifier arg) {
      super(arg);
    }

    private static Key create(PackageIdentifier arg) {
      return interner.intern(new Key(arg));
    }

    @VisibleForSerialization
    @AutoCodec.Interner
    static Key intern(Key key) {
      return interner.intern(key);
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.PACKAGE_LOOKUP;
    }

    @Override
    public SkyKeyInterner<Key> getSkyKeyInterner() {
      return interner;
    }
  }

  /** Successful lookup value. */
  public static class SuccessfulPackageLookupValue extends PackageLookupValue {
    /**
     * The repository value the meaning of the path depends on (e.g., an external repository
     * controlling a symbolic link the path goes trough). Can be {@code null}, if does not depend on
     * such a repository; will always be {@code null} for packages in the main repository.
     */
    @Nullable private final RepositoryDirectoryValue repository;

    private final Root root;
    private final BuildFileName buildFileName;

    private final boolean hasProjectFile;

    SuccessfulPackageLookupValue(
        @Nullable RepositoryDirectoryValue repository,
        Root root,
        BuildFileName buildFileName,
        boolean hasProjectFile) {
      this.repository = repository;
      this.root = root;
      this.buildFileName = buildFileName;
      this.hasProjectFile = hasProjectFile;
    }

    @Override
    public boolean packageExists() {
      return true;
    }

    @Override
    public boolean hasProjectFile() {
      return hasProjectFile;
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
    public boolean hasProjectFile() {
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

  /** Marker for project file not actually being a file. */
  public static class InvalidProjectLookupValue extends UnsuccessfulPackageLookupValue {
    private InvalidProjectLookupValue() {}

    @Override
    ErrorReason getErrorReason() {
      return ErrorReason.INVALID_PROJECT_FILE;
    }

    @Override
    public String getErrorMsg() {
      return "PROJECT.scl not a file";
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
          "%s: invalidPackageIdentifier: %s, corrected: %s",
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
    private final RepositoryName repositoryName;
    private final String reason;

    NoRepositoryPackageLookupValue(RepositoryName repositoryName, String reason) {
      this.repositoryName = repositoryName;
      this.reason = reason;
    }

    @Override
    ErrorReason getErrorReason() {
      return ErrorReason.REPOSITORY_NOT_FOUND;
    }

    @Override
    public String getErrorMsg() {
      return String.format("The repository '%s' could not be resolved: %s", repositoryName, reason);
    }
  }

  /**
   * Creates the error message for the input {@linkplain Label label} has a subpackage crossing
   * boundary.
   *
   * <p>Returns {@code null} if no subpackage is discovered or the subpackage is marked as DELETED.
   */
  @Nullable
  static String getErrorMessageForLabelCrossingPackageBoundary(
      Root pkgRoot,
      Label label,
      PackageIdentifier subpackageIdentifier,
      PackageLookupValue packageLookupValue) {
    String message = null;
    if (packageLookupValue.packageExists()) {
      message =
          String.format(
              "Label '%s' is invalid because '%s' is a subpackage", label, subpackageIdentifier);
      Root subPackageRoot = packageLookupValue.getRoot();

      if (pkgRoot.equals(subPackageRoot)) {
        PathFragment labelRootPathFragment = label.getPackageIdentifier().getSourceRoot();
        PathFragment subpackagePathFragment = subpackageIdentifier.getSourceRoot();
        if (subpackagePathFragment.startsWith(labelRootPathFragment)) {
          PathFragment labelNameInSubpackage =
              PathFragment.create(label.getName())
                  .subFragment(
                      subpackagePathFragment.segmentCount() - labelRootPathFragment.segmentCount());
          message += "; perhaps you meant to put the" + " colon here: '";
          if (subpackageIdentifier.getRepository().isMain()) {
            message += "//";
          }
          message += subpackageIdentifier + ":" + labelNameInSubpackage + "'?";
        } else {
          // TODO: Is this a valid case? How do we handle this case?
        }
      } else {
        message +=
            "; have you deleted "
                + subpackageIdentifier
                + "/BUILD? "
                + "If so, use the --deleted_packages="
                + subpackageIdentifier
                + " option";
      }
    } else if (packageLookupValue instanceof IncorrectRepositoryReferencePackageLookupValue) {
      message =
          String.format(
              "Label '%s' is invalid because '%s' is a subpackage",
              label,
              ((IncorrectRepositoryReferencePackageLookupValue) packageLookupValue)
                  .correctedPackageIdentifier);
    }
    return message;
  }
}
