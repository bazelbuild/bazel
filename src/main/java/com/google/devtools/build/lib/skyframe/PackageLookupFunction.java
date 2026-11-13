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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.FileStateValue;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.cmdline.IgnoredSubdirectories.IgnoredReason;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.cmdline.LabelValidator;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.io.FileSymlinkException;
import com.google.devtools.build.lib.io.InconsistentFilesystemException;
import com.google.devtools.build.lib.packages.BuildFileName;
import com.google.devtools.build.lib.packages.BuildFileNotFoundException;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.RepositoryFetchException;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.rules.repository.RepositoryDirectoryValue;
import com.google.devtools.build.lib.rules.repository.RepositoryDirectoryValue.Failure;
import com.google.devtools.build.lib.rules.repository.RepositoryDirectoryValue.Success;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.PackageLoading;
import com.google.devtools.build.lib.server.FailureDetails.PackageLoading.Code;
import com.google.devtools.build.lib.skyframe.DirectorySpellingValue.NonCanonical;
import com.google.devtools.build.lib.skyframe.PackageLookupValue.NoRepositoryPackageLookupValue;
import com.google.devtools.build.lib.skyframe.PackageLookupValue.NonCanonicalSpellingPackageLookupValue;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.vfs.DetailedIOException;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunction.Environment.SkyKeyComputeState;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.SkyframeLookupResult;
import java.io.IOException;
import java.util.concurrent.atomic.AtomicReference;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;

/** SkyFunction for {@link PackageLookupValue}s. */
public class PackageLookupFunction implements SkyFunction {
  private final AtomicReference<ImmutableSet<PackageIdentifier>> deletedPackages;
  private final ImmutableList<BuildFileName> buildFilesByPriority;

  public PackageLookupFunction(
      AtomicReference<ImmutableSet<PackageIdentifier>> deletedPackages,
      ImmutableList<BuildFileName> buildFilesByPriority) {
    this.deletedPackages = deletedPackages;
    this.buildFilesByPriority = buildFilesByPriority;
  }

  private static class State implements SkyKeyComputeState {
    private int packagePathEntryPos = 0;
    private int buildFileNamePos = 0;
    @Nullable private PackageLookupValue nonCanonicalSpelling;
  }

  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws PackageLookupFunctionException, InterruptedException {
    PathPackageLocator pkgLocator = PrecomputedValue.PATH_PACKAGE_LOCATOR.get(env);

    PackageIdentifier packageKey = (PackageIdentifier) skyKey.argument();

    String packageNameErrorMsg =
        LabelValidator.validatePackageName(packageKey.getPackageFragment().getPathString());
    if (packageNameErrorMsg != null) {
      return PackageLookupValue.invalidPackageName(
          "Invalid package name '" + packageKey + "': " + packageNameErrorMsg);
    }

    if (deletedPackages.get().contains(packageKey)) {
      return PackageLookupValue.DELETED_PACKAGE_VALUE;
    }

    if (!packageKey.getRepository().isMain()) {
      return computeExternalPackageLookupValue(skyKey, env, packageKey);
    }

    if (packageKey.equals(LabelConstants.EXTERNAL_PACKAGE_IDENTIFIER)) {
      return PackageLookupValue.NO_BUILD_FILE_VALUE;
    }

    // Check .bazelignore file under main repository.
    IgnoredSubdirectoriesValue ignoredPatternsValue =
        (IgnoredSubdirectoriesValue) env.getValue(IgnoredSubdirectoriesValue.key());
    if (ignoredPatternsValue == null) {
      return null;
    }

    PathFragment packageFragment = packageKey.getPackageFragment();
    IgnoredReason reason =
        ignoredPatternsValue.asIgnoredSubdirectories().matchingEntryReason(packageFragment);
    if (reason != null) {
      return reason == IgnoredReason.BAZELIGNORE
          ? PackageLookupValue.DELETED_BY_BAZELIGNORE_VALUE
          : PackageLookupValue.DELETED_BY_REPO_BAZEL_VALUE;
    }

    return findPackageByBuildFile(env, pkgLocator, packageKey);
  }

  /**
   * For a package identifier {@code packageKey} such that the compute for {@code
   * PackageLookupValue.key(packageKey)} returned {@code NO_BUILD_FILE_VALUE}, provide a
   * human-readable error message with more details on where we searched for the package.
   */
  public static String explainNoBuildFileValue(PackageIdentifier packageKey, Environment env)
      throws InterruptedException {
    String educationalMessage = "Add a BUILD file to a directory to mark it as a package.";
    if (packageKey.getRepository().isMain()) {
      PathPackageLocator pkgLocator = PrecomputedValue.PATH_PACKAGE_LOCATOR.get(env);
      StringBuilder message = new StringBuilder();
      message.append("BUILD file not found in any of the following directories. ");
      message.append(educationalMessage);
      for (Root root : pkgLocator.getPathEntries()) {
        message
            .append("\n - ")
            .append(
                pkgLocator.getPathEntries().size() == 1
                    ? packageKey.getPackageFragment().getPathString()
                    : root.asPath().getRelative(packageKey.getPackageFragment()).getPathString());
      }
      return message.toString();
    } else {
      return "BUILD file not found in directory '"
          + packageKey.getPackageFragment()
          + "' of external repository "
          + packageKey.getRepository()
          + ". "
          + educationalMessage;
    }
  }

  @Nullable
  private PackageLookupValue findPackageByBuildFile(
      Environment env, PathPackageLocator pkgLocator, PackageIdentifier packageKey)
      throws PackageLookupFunctionException, InterruptedException {
    State state = env.getState(State::new);
    while (state.packagePathEntryPos < pkgLocator.getPathEntries().size()) {
      while (state.buildFileNamePos < buildFilesByPriority.size()) {
        Root packagePathEntry = pkgLocator.getPathEntries().get(state.packagePathEntryPos);
        BuildFileName buildFileName = buildFilesByPriority.get(state.buildFileNamePos);
        PackageLookupValue result =
            getPackageLookupValue(env, packagePathEntry, packageKey, buildFileName);
        if (result == null) {
          return null;
        }
        if (result instanceof NonCanonicalSpellingPackageLookupValue nonCanonical) {
          if (state.nonCanonicalSpelling == null) {
            state.nonCanonicalSpelling = result;
          }
          if (!nonCanonical.isBuildFileName()) {
            // Other build file names can't help since the package directory itself is misspelled.
            break;
          }
          state.buildFileNamePos++;
          continue;
        }
        if (result != PackageLookupValue.NO_BUILD_FILE_VALUE) {
          return result;
        }
        state.buildFileNamePos++;
      }
      state.buildFileNamePos = 0;
      state.packagePathEntryPos++;
    }
    return state.nonCanonicalSpelling != null
        ? state.nonCanonicalSpelling
        : PackageLookupValue.NO_BUILD_FILE_VALUE;
  }

  @Nullable
  private static FileValue getFileValue(
      RootedPath fileRootedPath, Environment env, PackageIdentifier packageIdentifier)
      throws PackageLookupFunctionException, InterruptedException {
    String basename = fileRootedPath.asPath().getBaseName();
    SkyKey fileSkyKey = FileValue.key(fileRootedPath);
    FileValue fileValue;
    try {
      fileValue = (FileValue) env.getValueOrThrow(fileSkyKey, IOException.class);
    } catch (InconsistentFilesystemException e) {
      // This error is not transient from the perspective of the PackageLookupFunction.
      throw new PackageLookupFunctionException(e, Transience.PERSISTENT);
    } catch (FileSymlinkException e) {
      String message =
          e.getMessage()
              + " detected while trying to find "
              + basename
              + " file "
              + fileRootedPath.asPath();
      throw new PackageLookupFunctionException(
          new BuildFileNotFoundException(
              packageIdentifier,
              message,
              DetailedExitCode.of(
                  FailureDetail.newBuilder()
                      .setMessage(message)
                      .setPackageLoading(
                          PackageLoading.newBuilder()
                              .setCode(Code.SYMLINK_CYCLE_OR_INFINITE_EXPANSION))
                      .build())),
          Transience.PERSISTENT);
    } catch (DetailedIOException e) {
      String message =
          "IO errors while looking for "
              + basename
              + " file reading "
              + fileRootedPath.asPath()
              + ": "
              + e.getMessage();
      throw new PackageLookupFunctionException(
          new BuildFileNotFoundException(
              packageIdentifier,
              message,
              DetailedExitCode.of(
                  e.getDetailedExitCode().getFailureDetail().toBuilder()
                      .setMessage(message)
                      .build())),
          e.getTransience());
    } catch (IOException e) {
      String message =
          "IO errors while looking for "
              + basename
              + " file reading "
              + fileRootedPath.asPath()
              + ": "
              + e.getMessage();
      throw new PackageLookupFunctionException(
          new BuildFileNotFoundException(
              packageIdentifier,
              message,
              DetailedExitCode.of(
                  FailureDetail.newBuilder()
                      .setMessage(message)
                      .setPackageLoading(
                          PackageLoading.newBuilder().setCode(Code.OTHER_IO_EXCEPTION))
                      .build())),
          Transience.PERSISTENT);
    }
    return fileValue;
  }

  @Nullable
  private PackageLookupValue getPackageLookupValue(
      Environment env,
      Root packagePathEntry,
      PackageIdentifier packageIdentifier,
      BuildFileName buildFileName)
      throws InterruptedException, PackageLookupFunctionException {
    PathFragment buildFileFragment = buildFileName.getBuildFileFragment(packageIdentifier);
    RootedPath packageDirectory =
        RootedPath.toRootedPath(packagePathEntry, buildFileFragment.getParentDirectory());

    // Check for the existence of the build file.
    RootedPath buildFileRootedPath =
        packageDirectory.getChild(buildFileName.getFilenameFragment().getPathString());
    FileValue fileValue = getFileValue(buildFileRootedPath, env, packageIdentifier);
    if (fileValue == null) {
      return null;
    }

    if (fileValue.isFile()) {
      return checkSpelling(env, packageDirectory, buildFileName, packageIdentifier);
    }

    return PackageLookupValue.NO_BUILD_FILE_VALUE;
  }

  /**
   * Returns the lookup value for a package whose build file {@code buildFileName} exists in {@code
   * packageDirectory}, taking {@code --incompatible_strict_label_casing} into account.
   */
  @Nullable
  private static PackageLookupValue checkSpelling(
      Environment env,
      RootedPath packageDirectory,
      BuildFileName buildFileName,
      PackageIdentifier packageIdentifier)
      throws PackageLookupFunctionException, InterruptedException {
    Root root = packageDirectory.getRoot();
    // File systems that compare names exactly only find the build file via its canonical spelling.
    if (!root.getFileSystem().mayBeCaseOrNormalizationInsensitive()) {
      return PackageLookupValue.success(root, buildFileName);
    }
    Boolean strictLabelCasing = PrecomputedValue.STRICT_LABEL_CASING.get(env);
    if (strictLabelCasing == null) {
      return null;
    }
    if (!strictLabelCasing) {
      return PackageLookupValue.success(root, buildFileName);
    }

    SkyKey spellingKey = DirectorySpellingValue.key(packageDirectory);
    SkyKey listingKey = DirectoryListingValue.key(packageDirectory);
    SkyframeLookupResult deps =
        env.getValuesAndExceptions(ImmutableList.of(spellingKey, listingKey));
    DirectorySpellingValue spelling;
    DirectoryListingValue listing;
    try {
      spelling = (DirectorySpellingValue) deps.getOrThrow(spellingKey, IOException.class);
      listing = (DirectoryListingValue) deps.getOrThrow(listingKey, IOException.class);
    } catch (InconsistentFilesystemException e) {
      throw new PackageLookupFunctionException(e, Transience.PERSISTENT);
    } catch (IOException e) {
      String message =
          "IO errors while checking the spelling of "
              + packageDirectory.asPath()
              + ": "
              + e.getMessage();
      throw new PackageLookupFunctionException(
          new BuildFileNotFoundException(
              packageIdentifier,
              message,
              DetailedExitCode.of(
                  FailureDetail.newBuilder()
                      .setMessage(message)
                      .setPackageLoading(
                          PackageLoading.newBuilder().setCode(Code.OTHER_IO_EXCEPTION))
                      .build())),
          Transience.PERSISTENT);
    }
    if (spelling == null || listing == null) {
      return null;
    }

    if (spelling instanceof NonCanonical(PathFragment canonicalPath)) {
      return PackageLookupValue.nonCanonicalDirectory(
          String.format(
              "the package name doesn't match the spelling of its directory%s on disk (see"
                  + " --incompatible_strict_label_casing)",
              canonicalPath != null ? " '" + canonicalPath.getPathString() + "'" : ""));
    }
    String buildFileBaseName = buildFileName.getFilenameFragment().getPathString();
    if (listing.getDirents().maybeGetDirent(buildFileBaseName) == null) {
      String spellingOnDisk =
          DirectorySpellingFunction.findUniqueLooseMatch(listing.getDirents(), buildFileBaseName);
      if (spellingOnDisk == null) {
        return PackageLookupValue.NO_BUILD_FILE_VALUE;
      }
      return PackageLookupValue.nonCanonicalBuildFileName(
          String.format(
              "'%s' is not a valid BUILD file name, use '%s' instead (see"
                  + " --incompatible_strict_label_casing)",
              spellingOnDisk, buildFileBaseName));
    }
    return PackageLookupValue.success(root, buildFileName);
  }

  /**
   * Gets a PackageLookupValue from a different Bazel repository.
   *
   * <p>To do this, it looks up the "external" package and finds a path mapping for the repository
   * name.
   */
  @Nullable
  private PackageLookupValue computeExternalPackageLookupValue(
      SkyKey skyKey, Environment env, PackageIdentifier packageIdentifier)
      throws PackageLookupFunctionException, InterruptedException {
    PackageIdentifier id = (PackageIdentifier) skyKey.argument();
    SkyKey repositoryKey = RepositoryDirectoryValue.key(id.getRepository());
    RepositoryDirectoryValue repositoryValue;
    try {
      repositoryValue =
          (RepositoryDirectoryValue)
              env.getValueOrThrow(
                  repositoryKey,
                  NoSuchPackageException.class,
                  IOException.class,
                  EvalException.class,
                  AlreadyReportedException.class);
      if (repositoryValue == null) {
        return null;
      }
    } catch (NoSuchPackageException e) {
      throw new PackageLookupFunctionException(
          new BuildFileNotFoundException(id, e.getMessage()), Transience.PERSISTENT);
    } catch (IOException | EvalException | AlreadyReportedException e) {
      throw new PackageLookupFunctionException(
          new RepositoryFetchException(id, e.getMessage()), Transience.PERSISTENT);
    }
    if (repositoryValue instanceof Failure(String errorMsg)) {
      return new NoRepositoryPackageLookupValue(id.getRepository(), errorMsg);
    }

    // Check .bazelignore file after fetching the external repository.
    IgnoredSubdirectoriesValue ignoredPatternsValue =
        (IgnoredSubdirectoriesValue)
            env.getValue(IgnoredSubdirectoriesValue.key(id.getRepository()));
    if (ignoredPatternsValue == null) {
      return null;
    }

    PathFragment packageFragment = id.getPackageFragment();
    IgnoredReason reason =
        ignoredPatternsValue.asIgnoredSubdirectories().matchingEntryReason(packageFragment);
    if (reason != null) {
      return reason == IgnoredReason.BAZELIGNORE
          ? PackageLookupValue.DELETED_BY_BAZELIGNORE_VALUE
          : PackageLookupValue.DELETED_BY_REPO_BAZEL_VALUE;
    }

    Root root = ((Success) repositoryValue).root();

    // This checks for the build file names in the correct precedence order.
    RootedPath packageDirectory = RootedPath.toRootedPath(root, id.getPackageFragment());
    PackageLookupValue misspelledBuildFileName = null;
    for (BuildFileName buildFileName : buildFilesByPriority) {
      RootedPath buildFileRootedPath =
          packageDirectory.getChild(buildFileName.getFilenameFragment().getPathString());
      FileValue fileValue = getFileValue(buildFileRootedPath, env, packageIdentifier);
      if (fileValue == null) {
        return null;
      }

      if (fileValue.isFile()) {
        PackageLookupValue result =
            checkSpelling(env, packageDirectory, buildFileName, packageIdentifier);
        if (result instanceof NonCanonicalSpellingPackageLookupValue nonCanonical
            && nonCanonical.isBuildFileName()) {
          if (misspelledBuildFileName == null) {
            misspelledBuildFileName = result;
          }
        } else if (result != PackageLookupValue.NO_BUILD_FILE_VALUE) {
          return result;
        }
      }
    }

    return misspelledBuildFileName != null
        ? misspelledBuildFileName
        : PackageLookupValue.NO_BUILD_FILE_VALUE;
  }

  /**
   * Used to declare all the exception types that can be wrapped in the exception thrown by {@link
   * PackageLookupFunction#compute}. Note that {@link InconsistentFilesystemException} can only be
   * thrown during target pattern parsing because of Bazel's end-to-end behavior: {@link
   * FileStateValue} throws {@link InconsistentFilesystemException} only if a
   * cached-on-this-evaluation directory listing said that an entry was a file but the stat had no
   * result. However, the only time Bazel lists a directory without first accessing its
   * BUILD/BUILD.bazel file is during evaluation of a recursive target pattern (like foo/...).
   */
  private static final class PackageLookupFunctionException extends SkyFunctionException {
    PackageLookupFunctionException(BuildFileNotFoundException e, Transience transience) {
      super(e, transience);
    }

    PackageLookupFunctionException(RepositoryFetchException e, Transience transience) {
      super(e, transience);
    }

    PackageLookupFunctionException(InconsistentFilesystemException e, Transience transience) {
      super(e, transience);
    }
  }
}
