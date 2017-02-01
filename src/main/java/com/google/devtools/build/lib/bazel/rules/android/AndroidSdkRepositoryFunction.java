// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.bazel.rules.android;

import com.android.repository.Revision;
import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.rules.repository.RepositoryDirectoryValue;
import com.google.devtools.build.lib.rules.repository.RepositoryFunction;
import com.google.devtools.build.lib.rules.repository.WorkspaceAttributeMapper;
import com.google.devtools.build.lib.skyframe.DirectoryListingValue;
import com.google.devtools.build.lib.skyframe.Dirents;
import com.google.devtools.build.lib.skyframe.FileSymlinkException;
import com.google.devtools.build.lib.skyframe.FileValue;
import com.google.devtools.build.lib.skyframe.InconsistentFilesystemException;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.util.ResourceFileLoader;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import java.io.IOException;
import java.util.Map;
import java.util.Properties;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import javax.annotation.Nullable;

/**
 * Implementation of the {@code android_sdk_repository} rule.
 */
public class AndroidSdkRepositoryFunction extends RepositoryFunction {
  private static final String BUILD_TOOLS_DIR_NAME = "build-tools";
  private static final String PLATFORMS_DIR_NAME = "platforms";
  private static final Pattern PLATFORMS_API_LEVEL_PATTERN = Pattern.compile("android-(\\d+)");
  private static final Revision MIN_BUILD_TOOLS_REVISION = new Revision(24, 0, 3);
  private static final String PATH_ENV_VAR = "ANDROID_HOME";

  @Override
  public boolean isLocal(Rule rule) {
    return true;
  }

  @Override
  public RepositoryDirectoryValue.Builder fetch(Rule rule, Path outputDirectory,
      BlazeDirectories directories, Environment env, Map<String, String> markerData)
      throws SkyFunctionException, InterruptedException {

    prepareLocalRepositorySymlinkTree(rule, outputDirectory);
    WorkspaceAttributeMapper attributes = WorkspaceAttributeMapper.of(rule);
    FileSystem fs = directories.getOutputBase().getFileSystem();
    Path androidSdkPath;
    if (attributes.isAttributeValueExplicitlySpecified("path")) {
      androidSdkPath = fs.getPath(getTargetPath(rule, directories.getWorkspace()));
    } else if (clientEnvironment.containsKey(PATH_ENV_VAR)){
      androidSdkPath =
          fs.getPath(getAndroidHomeEnvironmentVar(directories.getWorkspace(), clientEnvironment));
    } else {
      throw new RepositoryFunctionException(
          new EvalException(
              rule.getLocation(),
              "Either the path attribute of android_sdk_repository or the ANDROID_HOME environment "
                  + " variable must be set."),
          Transience.PERSISTENT);
    }

    if (!symlinkLocalRepositoryContents(outputDirectory, androidSdkPath)) {
      return null;
    }

    DirectoryListingValue platformsDirectoryValue =
        getDirectoryListing(fs, androidSdkPath.getChild(PLATFORMS_DIR_NAME), env);
    if (platformsDirectoryValue == null) {
      return null;
    }

    ImmutableSortedSet<Integer> apiLevels = getApiLevels(platformsDirectoryValue.getDirents());
    if (apiLevels.isEmpty()) {
      throw new RepositoryFunctionException(
          new EvalException(
              rule.getLocation(),
              "android_sdk_repository requires that at least one Android SDK Platform is installed "
                  + "in the Android SDK. Please install an Android SDK Platform through the "
                  + "Android SDK manager."),
          Transience.PERSISTENT);
    }

    String defaultApiLevel;
    if (attributes.isAttributeValueExplicitlySpecified("api_level")) {
      try {
        defaultApiLevel = attributes.get("api_level", Type.INTEGER).toString();
      } catch (EvalException e) {
        throw new RepositoryFunctionException(e, Transience.PERSISTENT);
      }
    } else {
      // If the api_level attribute is not explicitly set, we select the highest api level that is
      // available in the SDK.
      defaultApiLevel = String.valueOf(apiLevels.first());
    }

    String buildToolsDirectory;
    if (attributes.isAttributeValueExplicitlySpecified("build_tools_version")) {
      try {
        buildToolsDirectory = attributes.get("build_tools_version", Type.STRING);
      } catch (EvalException e) {
        throw new RepositoryFunctionException(e, Transience.PERSISTENT);
      }
    } else {
      // If the build_tools_version attribute is not explicitly set, we select the highest version
      // installed in the SDK.
      DirectoryListingValue directoryValue =
          getDirectoryListing(fs, androidSdkPath.getChild(BUILD_TOOLS_DIR_NAME), env);
      if (directoryValue == null) {
        return null;
      }
      buildToolsDirectory = getNewestBuildToolsDirectory(rule, directoryValue.getDirents());
    }

    // android_sdk_repository.build_tools_version is technically actually the name of the
    // directory in $sdk/build-tools. Most of the time this is just the actual build tools
    // version, but for preview build tools, the directory is something like 24.0.0-preview, and
    // the actual version is something like "24 rc3". The android_sdk rule in the template needs
    // the real version.
    String buildToolsVersion;
    if (buildToolsDirectory.contains("-preview")) {

      Properties sourceProperties =
          getBuildToolsSourceProperties(outputDirectory, buildToolsDirectory, env);
      if (env.valuesMissing()) {
        return null;
      }

      buildToolsVersion = sourceProperties.getProperty("Pkg.Revision");

    } else {
      buildToolsVersion = buildToolsDirectory;
    }

    try {
      assertValidBuildToolsVersion(rule, buildToolsVersion);
    } catch (EvalException e) {
      throw new RepositoryFunctionException(e, Transience.PERSISTENT);
    }

    String template = getStringResource("android_sdk_repository_template.txt");

    String buildFile = template
        .replace("%repository_name%", rule.getName())
        .replace("%build_tools_version%", buildToolsVersion)
        .replace("%build_tools_directory%", buildToolsDirectory)
        .replace("%api_levels%", Iterables.toString(apiLevels))
        .replace("%default_api_level%", defaultApiLevel);

    // All local maven repositories that are shipped in the Android SDK.
    // TODO(ajmichael): Create SkyKeys so that if the SDK changes, this function will get rerun.
    Iterable<Path> localMavenRepositories = ImmutableList.of(
        outputDirectory.getRelative("extras/android/m2repository"),
        outputDirectory.getRelative("extras/google/m2repository"));
    try {
      SdkMavenRepository sdkExtrasRepository =
          SdkMavenRepository.create(Iterables.filter(localMavenRepositories, new Predicate<Path>() {
            @Override
            public boolean apply(@Nullable Path path) {
              return path.isDirectory();
            }
          }));
      sdkExtrasRepository.writeBuildFiles(outputDirectory);
      buildFile = buildFile.replace(
          "%exported_files%", sdkExtrasRepository.getExportsFiles(outputDirectory));
    } catch (IOException e) {
      throw new RepositoryFunctionException(e, Transience.TRANSIENT);
    }

    writeBuildFile(outputDirectory, buildFile);
    return RepositoryDirectoryValue.builder().setPath(outputDirectory);
  }

  @Override
  public Class<? extends RuleDefinition> getRuleDefinition() {
    return AndroidSdkRepositoryRule.class;
  }

  private static PathFragment getAndroidHomeEnvironmentVar(
      Path workspace, Map<String, String> env) {
    return workspace.getRelative(new PathFragment(env.get(PATH_ENV_VAR))).asFragment();
  }

  private static String getStringResource(String name) {
    try {
      return ResourceFileLoader.loadResource(
          AndroidSdkRepositoryFunction.class, name);
    } catch (IOException e) {
      throw new IllegalStateException(e);
    }
  }

  /** Gets a DirectoryListingValue for {@code dirPath} or returns null. */
  private static DirectoryListingValue getDirectoryListing(
      FileSystem fs, Path dirPath, Environment env)
      throws RepositoryFunctionException, InterruptedException {
    try {
      return (DirectoryListingValue)
          env.getValueOrThrow(
              DirectoryListingValue.key(RootedPath.toRootedPath(fs.getRootDirectory(), dirPath)),
              InconsistentFilesystemException.class);
    } catch (InconsistentFilesystemException e) {
      throw new RepositoryFunctionException(new IOException(e), Transience.PERSISTENT);
    }
  }

  /**
   * Gets the numeric api levels from the contents of the platforms directory in descending order.
   */
  private static ImmutableSortedSet<Integer> getApiLevels(Dirents platformsDirectories) {
    ImmutableSortedSet.Builder<Integer> apiLevels = ImmutableSortedSet.reverseOrder();
    for (Dirent platformDirectory : platformsDirectories) {
      if (platformDirectory.getType() != Dirent.Type.DIRECTORY) {
        continue;
      }
      Matcher matcher = PLATFORMS_API_LEVEL_PATTERN.matcher(platformDirectory.getName());
      if (matcher.matches()) {
        apiLevels.add(Integer.parseInt(matcher.group(1)));
      }
    }
    return apiLevels.build();
  }

  /**
   * Gets the newest build tools directory according to {@link Revision}.
   *
   * @throws RepositoryFunctionException if none of the buildToolsDirectories are directories and
   *     have names that are parsable as build tools version.
   */
  private static String getNewestBuildToolsDirectory(Rule rule, Dirents buildToolsDirectories)
      throws RepositoryFunctionException {
    String newestBuildToolsDirectory = null;
    Revision newestBuildToolsRevision = null;
    for (Dirent buildToolsDirectory : buildToolsDirectories) {
      if (buildToolsDirectory.getType() != Dirent.Type.DIRECTORY) {
        continue;
      }
      try {
        Revision buildToolsRevision = Revision.parseRevision(buildToolsDirectory.getName());
        if (newestBuildToolsRevision == null
            || buildToolsRevision.compareTo(newestBuildToolsRevision) > 0) {
          newestBuildToolsDirectory = buildToolsDirectory.getName();
          newestBuildToolsRevision = buildToolsRevision;
        }
      } catch (NumberFormatException e) {
        // Ignore unparsable build tools directories.
      }
    }
    if (newestBuildToolsDirectory == null) {
      throw new RepositoryFunctionException(
          new EvalException(
              rule.getLocation(),
              String.format(
                  "Bazel requires Android build tools version %s or newer but none are installed. "
                      + "Please install a recent version through the Android SDK manager.",
                  MIN_BUILD_TOOLS_REVISION)),
          Transience.PERSISTENT);
    }
    return newestBuildToolsDirectory;
  }

  private static Properties getBuildToolsSourceProperties(
      Path directory, String buildToolsDirectory, Environment env)
      throws RepositoryFunctionException, InterruptedException {

    Path sourcePropertiesFilePath = directory.getRelative(
        "build-tools/" + buildToolsDirectory + "/source.properties");

    SkyKey releaseFileKey = FileValue.key(
        RootedPath.toRootedPath(directory, sourcePropertiesFilePath));

    try {
      env.getValueOrThrow(releaseFileKey,
          IOException.class,
          FileSymlinkException.class,
          InconsistentFilesystemException.class);

      Properties properties = new Properties();
      properties.load(sourcePropertiesFilePath.getInputStream());
      return properties;

    } catch (IOException | FileSymlinkException | InconsistentFilesystemException e) {
      String error = String.format(
          "Could not read %s in Android SDK: %s", sourcePropertiesFilePath, e.getMessage());
      throw new RepositoryFunctionException(new IOException(error), Transience.PERSISTENT);
    }
  }

  private static void assertValidBuildToolsVersion(Rule rule, String buildToolsVersion)
      throws EvalException {
    try {
      Revision buildToolsRevision = Revision.parseRevision(buildToolsVersion);
      if (buildToolsRevision.compareTo(MIN_BUILD_TOOLS_REVISION) < 0) {
        throw new EvalException(
            rule.getAttributeLocation("build_tools_version"),
            String.format(
                "Bazel requires Android build tools version %s or newer, %s was provided",
                MIN_BUILD_TOOLS_REVISION,
                buildToolsRevision));
      }
    } catch (NumberFormatException e) {
      throw new EvalException(
          rule.getAttributeLocation("build_tools_version"),
          String.format(
              "Bazel does not recognize Android build tools version %s",
              buildToolsVersion),
          e);
    }
  }
}
