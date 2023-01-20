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

package com.google.devtools.build.lib.analysis.config;

import static com.google.common.base.Predicates.not;
import static java.util.stream.Collectors.joining;

import com.google.common.base.Strings;
import com.google.common.collect.ImmutableSortedMap;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.ArtifactRoot.RootType;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.PlatformOptions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.server.FailureDetails.BuildConfiguration.Code;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.PathFragment.InvalidBaseNameException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * Logic for figuring out what base directories to place outputs generated from a given
 * configuration.
 *
 * <p>In other words, when your output ends up in <code>blaze-out/x86-fastbuild/...</code>, this
 * class is why.
 */
public class OutputDirectories {
  /**
   * Directories in the output tree.
   *
   * <p>The computation of the output directory should be a non-injective mapping from
   * BuildConfigurationValue instances to strings. The result should identify the aspects of the
   * configuration that should be reflected in the output file names. Furthermore the returned
   * string must not contain shell metacharacters.
   *
   * <p>For configuration settings which are NOT part of the output directory name, rebuilding with
   * a different value of such a setting will build in the same output directory. This means that
   * any actions whose keys (see Action.getKey()) have changed will be rerun. That may result in a
   * lot of recompilation.
   *
   * <p>For configuration settings which ARE part of the output directory name, rebuilding with a
   * different value of such a setting will rebuild in a different output directory; this will
   * result in higher disk usage and more work the <i>first</i> time you rebuild with a different
   * setting, but will result in less work if you regularly switch back and forth between different
   * settings.
   *
   * <p>With one important exception, it's sound to choose any subset of the config's components for
   * this string, it just alters the dimensionality of the cache. In other words, it's a trade-off
   * on the "injectiveness" scale: at one extreme (output directory name contains all data in the
   * config, and is thus injective) you get extremely precise caching (no competition for the same
   * output-file locations) but you have to rebuild for even the slightest change in configuration.
   * At the other extreme (the output (directory name is a constant) you have very high competition
   * for output-file locations, but if a slight change in configuration doesn't affect a particular
   * build step, you're guaranteed not to have to rebuild it. The important exception has to do with
   * multiple configurations: every configuration in the build must have a different output
   * directory name so that their artifacts do not conflict.
   */
  public enum OutputDirectory {
    BIN("bin"),
    GENFILES("genfiles"),
    MIDDLEMAN("internal"),
    TESTLOGS("testlogs"),
    COVERAGE("coverage-metadata"),
    BUILDINFO(BlazeDirectories.RELATIVE_BUILD_INFO_DIR),
    OUTPUT("");

    private final String name;

    OutputDirectory(String name) {
      // Must be a legal basename for root - multiple segments not allowed.
      if (!name.isEmpty()) {
        FileSystemUtils.checkBaseName(name);
      }
      this.name = name;
    }

    public ArtifactRoot getRoot(
        String outputDirName, BlazeDirectories directories, RepositoryName mainRepositoryName) {
      // e.g., execroot/repo1
      Path execRoot = directories.getExecRoot(mainRepositoryName.getName());
      // e.g., [[execroot/repo1]/bazel-out/config/bin]
      return ArtifactRoot.asDerivedRoot(
          execRoot,
          this == MIDDLEMAN ? RootType.Middleman : RootType.Output,
          directories.getRelativeOutputPath(),
          outputDirName,
          name);
    }
  }

  private final BlazeDirectories directories;
  private final String mnemonic;

  private final ArtifactRoot outputDirectory;
  private final ArtifactRoot binDirectory;
  private final ArtifactRoot buildInfoDirectory;
  private final ArtifactRoot genfilesDirectory;
  private final ArtifactRoot coverageDirectory;
  private final ArtifactRoot testlogsDirectory;
  private final ArtifactRoot middlemanDirectory;

  private final boolean mergeGenfilesDirectory;

  private final boolean siblingRepositoryLayout;

  private final Path execRoot;

  OutputDirectories(
      BlazeDirectories directories,
      CoreOptions options,
      @Nullable PlatformOptions platformOptions,
      ImmutableSortedMap<Class<? extends Fragment>, Fragment> fragments,
      RepositoryName mainRepositoryName,
      boolean siblingRepositoryLayout,
      String transitionDirectoryNameFragment)
      throws InvalidMnemonicException {
    this.directories = directories;
    this.mnemonic =
        buildMnemonic(options, platformOptions, fragments, transitionDirectoryNameFragment);

    this.outputDirectory =
        OutputDirectory.OUTPUT.getRoot(mnemonic, directories, mainRepositoryName);
    this.binDirectory = OutputDirectory.BIN.getRoot(mnemonic, directories, mainRepositoryName);
    this.buildInfoDirectory =
        OutputDirectory.BUILDINFO.getRoot(mnemonic, directories, mainRepositoryName);
    this.genfilesDirectory =
        OutputDirectory.GENFILES.getRoot(mnemonic, directories, mainRepositoryName);
    this.coverageDirectory =
        OutputDirectory.COVERAGE.getRoot(mnemonic, directories, mainRepositoryName);
    this.testlogsDirectory =
        OutputDirectory.TESTLOGS.getRoot(mnemonic, directories, mainRepositoryName);
    this.middlemanDirectory =
        OutputDirectory.MIDDLEMAN.getRoot(mnemonic, directories, mainRepositoryName);

    this.mergeGenfilesDirectory = options.mergeGenfilesDirectory;
    this.siblingRepositoryLayout = siblingRepositoryLayout;
    this.execRoot = directories.getExecRoot(mainRepositoryName.getName());
  }

  private static void addMnemonicPart(
      List<String> nameParts, String part, String errorTemplate, Object... spec)
      throws InvalidMnemonicException {
    if (Strings.isNullOrEmpty(part)) {
      return;
    }

    validateMnemonicPart(part, errorTemplate, spec);

    nameParts.add(part);
  }

  /**
   * Validate that part is valid for use in the mnemonic, emitting an error message based on the
   * template if not.
   *
   * <p>The error template is expanded with the part itself as the first argument, and any remaining
   * elements of errorArgs following.
   */
  private static void validateMnemonicPart(String part, String errorTemplate, Object... errorArgs)
      throws InvalidMnemonicException {
    try {
      PathFragment.checkSeparators(part);
    } catch (InvalidBaseNameException e) {
      Object[] args = new Object[errorArgs.length + 1];
      args[0] = part;
      System.arraycopy(errorArgs, 0, args, 1, errorArgs.length);
      String message = String.format(errorTemplate, args);
      throw new InvalidMnemonicException(message, e);
    }
  }

  private static String buildMnemonic(
      CoreOptions options,
      @Nullable PlatformOptions platformOptions,
      ImmutableSortedMap<Class<? extends Fragment>, Fragment> fragments,
      String transitionDirectoryNameFragment)
      throws InvalidMnemonicException {
    // See explanation at declaration for outputRoots.
    List<String> nameParts = new ArrayList<>();

    // Add the fragment-specific sections.
    for (Map.Entry<Class<? extends Fragment>, Fragment> entry : fragments.entrySet()) {
      String outputDirectoryName = entry.getValue().getOutputDirectoryName();
      addMnemonicPart(
          nameParts,
          outputDirectoryName,
          "Output directory name '%s' specified by %s",
          entry.getKey().getSimpleName());
    }

    // Add the compilation mode.
    addMnemonicPart(nameParts, options.compilationMode.toString(), "Compilation mode '%s'");

    // Add the platform suffix, if any.
    addMnemonicPart(nameParts, options.platformSuffix, "Platform suffix '%s'");

    // Add the transition suffix.
    addMnemonicPart(
        nameParts, transitionDirectoryNameFragment, "Transition directory name fragment '%s'");

    // Join all the parts.
    String mnemonic = nameParts.stream().filter(not(Strings::isNullOrEmpty)).collect(joining("-"));

    // Replace the CPU idenfitier.
    String cpuIdentifier = buildCpuIdentifier(options, platformOptions);
    validateMnemonicPart(cpuIdentifier, "CPU name '%s'");
    mnemonic = mnemonic.replace("{CPU}", cpuIdentifier);

    return mnemonic;
  }

  private static String buildCpuIdentifier(
      CoreOptions options, @Nullable PlatformOptions platformOptions) {
    if (options.platformInOutputDir && platformOptions != null) {
      Label targetPlatform = platformOptions.computeTargetPlatform();
      // Only use non-default platforms.
      if (!PlatformOptions.platformIsDefault(targetPlatform)) {
        return targetPlatform.getName();
      }
    }

    // Fall back to using the CPU.
    return options.cpu;
  }

  private ArtifactRoot buildDerivedRoot(
      String nameFragment, RepositoryName repository, boolean isMiddleman) {
    // e.g., execroot/mainRepoName/bazel-out/[repoName/]config/bin
    // TODO(jungjw): Ideally, we would like to do execroot_base/repoName/bazel-out/config/bin
    // instead. However, it requires individually symlinking the top-level elements of external
    // repositories, which is blocked by a Windows symlink issue #8704.
    RootType rootType;
    if (repository.isMain()) {
      rootType = isMiddleman ? RootType.SiblingMainMiddleman : RootType.SiblingMainOutput;
    } else {
      rootType = isMiddleman ? RootType.SiblingExternalMiddleman : RootType.SiblingExternalOutput;
    }
    return ArtifactRoot.asDerivedRoot(
        execRoot,
        rootType,
        directories.getRelativeOutputPath(),
        repository.getName(),
        mnemonic,
        nameFragment);
  }

  /** Returns the output directory for this build configuration. */
  ArtifactRoot getOutputDirectory(RepositoryName repositoryName) {
    return siblingRepositoryLayout ? buildDerivedRoot("", repositoryName, false) : outputDirectory;
  }

  /** Returns the bin directory for this build configuration. */
  ArtifactRoot getBinDirectory(RepositoryName repositoryName) {
    return siblingRepositoryLayout ? buildDerivedRoot("bin", repositoryName, false) : binDirectory;
  }

  /** Returns the build-info directory for this build configuration. */
  ArtifactRoot getBuildInfoDirectory(RepositoryName repositoryName) {
    return siblingRepositoryLayout
        ? buildDerivedRoot(BlazeDirectories.RELATIVE_BUILD_INFO_DIR, repositoryName, false)
        : buildInfoDirectory;
  }

  /** Returns the genfiles directory for this build configuration. */
  ArtifactRoot getGenfilesDirectory(RepositoryName repositoryName) {
    return mergeGenfilesDirectory
        ? getBinDirectory(repositoryName)
        : siblingRepositoryLayout
            ? buildDerivedRoot("genfiles", repositoryName, false)
            : genfilesDirectory;
  }

  /**
   * Returns the directory where coverage-related artifacts and metadata files should be stored.
   * This includes for example uninstrumented class files needed for Jacoco's coverage reporting
   * tools.
   */
  ArtifactRoot getCoverageMetadataDirectory(RepositoryName repositoryName) {
    return siblingRepositoryLayout
        ? buildDerivedRoot("coverage-metadata", repositoryName, false)
        : coverageDirectory;
  }

  /** Returns the testlogs directory for this build configuration. */
  ArtifactRoot getTestLogsDirectory(RepositoryName repositoryName) {
    return siblingRepositoryLayout
        ? buildDerivedRoot("testlogs", repositoryName, false)
        : testlogsDirectory;
  }

  /** Returns a relative path to the genfiles directory at execution time. */
  PathFragment getGenfilesFragment(RepositoryName repositoryName) {
    return getGenfilesDirectory(repositoryName).getExecPath();
  }

  /**
   * Returns the path separator for the host platform. This is basically the same as {@link
   * java.io.File#pathSeparator}, except that that returns the value for this JVM, which may or may
   * not match the host platform. You should only use this when invoking tools that are known to use
   * the native path separator, i.e., the path separator for the machine that they run on.
   */
  String getHostPathSeparator() {
    // TODO(bazel-team): Maybe do this in the constructor instead? This isn't serialization-safe.
    return OS.getCurrent() == OS.WINDOWS ? ";" : ":";
  }

  /** Returns the internal directory (used for middlemen) for this build configuration. */
  ArtifactRoot getMiddlemanDirectory(RepositoryName repositoryName) {
    return siblingRepositoryLayout
        ? buildDerivedRoot("internal", repositoryName, true)
        : middlemanDirectory;
  }

  String getMnemonic() {
    return mnemonic;
  }

  String getOutputDirName() {
    return getMnemonic();
  }

  boolean mergeGenfilesDirectory() {
    return mergeGenfilesDirectory;
  }

  BlazeDirectories getDirectories() {
    return directories;
  }

  /** Indicates a failure to construct the mnemonic for an output directory. */
  public static class InvalidMnemonicException extends InvalidConfigurationException {
    InvalidMnemonicException(String message, InvalidBaseNameException e) {
      super(
          message + " is invalid as part of a path: " + e.getMessage(),
          Code.INVALID_OUTPUT_DIRECTORY_MNEMONIC);
    }
  }
}
