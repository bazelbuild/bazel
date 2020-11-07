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

import static com.google.common.collect.ImmutableMap.toImmutableMap;

import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Streams;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.actions.InconsistentFilesystemException;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.rules.repository.RepositoryDirectoryValue;
import com.google.devtools.build.lib.rules.repository.WorkspaceAttributeMapper;
import com.google.devtools.build.lib.skyframe.DirectoryListingValue;
import com.google.devtools.build.lib.skyframe.Dirents;
import com.google.devtools.build.lib.util.ResourceFileLoader;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.ValueOrException;
import java.io.IOException;
import java.io.InputStream;
import java.util.Iterator;
import java.util.Map;
import java.util.Properties;
import net.starlark.java.eval.EvalException;

/** Implementation of the {@code android_sdk_repository} rule. */
public class AndroidSdkRepositoryFunction extends AndroidRepositoryFunction {

  static final class AndroidRevision implements Comparable<AndroidRevision> {

    private final String original;
    private final int major;
    private final int minor;
    private final int micro;
    private final int previewType;
    private final int preview;

    private AndroidRevision(
        String original, int major, int minor, int micro, int previewType, int preview) {
      this.original = original;
      this.major = major;
      this.minor = minor;
      this.micro = micro;
      this.previewType = previewType;
      this.preview = preview;
    }

    static AndroidRevision parse(String revisionString) {
      revisionString = revisionString.trim();
      String[] revisionAndPreview = revisionString.split("-|([ ]+)", 2);
      if (revisionAndPreview.length < 1) {
        throw new NumberFormatException("Invalid revision: " + revisionString);
      }

      Iterator<String> revision = Splitter.on('.').split(revisionAndPreview[0]).iterator();

      if (!revision.hasNext()) {
        throw new NumberFormatException("Invalid revision: " + revisionString);
      }

      int major = Integer.parseInt(revision.next());
      int minor = 0;
      int micro = 0;
      // Revisions without preview are larger than those with, so set these to MAX_VALUE and
      // if there's a preview value, these will get set below.
      int previewType = Integer.MAX_VALUE;
      int preview = Integer.MAX_VALUE;

      if (revision.hasNext()) {
        minor = Integer.parseInt(revision.next());
      }

      if (revision.hasNext()) {
        micro = Integer.parseInt(revision.next());
      }

      if (revisionAndPreview.length == 2) {
        String p = revisionAndPreview[1];
        if (p.contains("rc")) {
          previewType = 3;
        } else if (p.contains("beta")) {
          previewType = 2;
        } else if (p.contains("alpha")) {
          previewType = 1;
        } else {
          throw new NumberFormatException("Invalid revision: " + revisionString);
        }
        p = p.replace("rc", "").replace("alpha", "").replace("beta", "");
        preview = Integer.parseInt(p);
      }
      return new AndroidRevision(revisionString, major, minor, micro, previewType, preview);
    }

    @Override
    public int compareTo(AndroidRevision other) {
      int major = this.major - other.major;
      if (major != 0) {
        return major;
      }

      int minor = this.minor - other.minor;
      if (minor != 0) {
        return minor;
      }

      int micro = this.micro - other.micro;
      if (micro != 0) {
        return micro;
      }

      int previewType = this.previewType - other.previewType;
      if (previewType != 0) {
        return previewType;
      }

      int preview = this.preview - other.preview;
      if (preview != 0) {
        return preview;
      }

      return 0;
    }

    @Override
    public String toString() {
      return original;
    }
  }

  private static final PathFragment BUILD_TOOLS_DIR = PathFragment.create("build-tools");
  private static final PathFragment PLATFORMS_DIR = PathFragment.create("platforms");
  private static final PathFragment SYSTEM_IMAGES_DIR = PathFragment.create("system-images");
  private static final AndroidRevision MIN_BUILD_TOOLS_REVISION = AndroidRevision.parse("26.0.1");
  private static final String PATH_ENV_VAR = "ANDROID_HOME";
  private static final ImmutableList<String> PATH_ENV_VAR_AS_LIST = ImmutableList.of(PATH_ENV_VAR);
  private static final ImmutableList<String> LOCAL_MAVEN_REPOSITORIES =
      ImmutableList.of(
          "extras/android/m2repository",
          "extras/google/m2repository",
          "extras/m2repository");

  @Override
  public boolean isLocal(Rule rule) {
    return true;
  }

  @Override
  public boolean verifyMarkerData(Rule rule, Map<String, String> markerData, Environment env)
      throws InterruptedException {
    WorkspaceAttributeMapper attributes = WorkspaceAttributeMapper.of(rule);
    if (attributes.isAttributeValueExplicitlySpecified("path")) {
      return true;
    }
    return super.verifyEnvironMarkerData(markerData, env, PATH_ENV_VAR_AS_LIST);
  }

  @Override
  public RepositoryDirectoryValue.Builder fetch(
      Rule rule,
      final Path outputDirectory,
      BlazeDirectories directories,
      Environment env,
      Map<String, String> markerData,
      SkyKey key)
      throws RepositoryFunctionException, InterruptedException {
    Map<String, String> environ =
        declareEnvironmentDependencies(markerData, env, PATH_ENV_VAR_AS_LIST);
    if (environ == null) {
      return null;
    }
    prepareLocalRepositorySymlinkTree(rule, outputDirectory);
    WorkspaceAttributeMapper attributes = WorkspaceAttributeMapper.of(rule);
    FileSystem fs = directories.getOutputBase().getFileSystem();
    Path androidSdkPath;
    String userDefinedPath = null;
    if (attributes.isAttributeValueExplicitlySpecified("path")) {
      userDefinedPath = getPathAttr(rule);
      androidSdkPath = fs.getPath(getTargetPath(userDefinedPath, directories.getWorkspace()));
    } else if (environ.get(PATH_ENV_VAR) != null) {
      userDefinedPath = environ.get(PATH_ENV_VAR);
      androidSdkPath =
          fs.getPath(getAndroidHomeEnvironmentVar(directories.getWorkspace(), environ));
    } else {
      // Write an empty BUILD file that declares errors when referred to.
      String buildFile = getStringResource("android_sdk_repository_empty_template.txt");
      writeBuildFile(outputDirectory, buildFile);
      return RepositoryDirectoryValue.builder().setPath(outputDirectory);
    }

    if (!symlinkLocalRepositoryContents(outputDirectory, androidSdkPath, userDefinedPath)) {
      return null;
    }

    DirectoryListingValue platformsDirectoryValue =
        getDirectoryListing(androidSdkPath, PLATFORMS_DIR, env);
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

    Integer defaultApiLevel;
    if (attributes.isAttributeValueExplicitlySpecified("api_level")) {
      try {
        defaultApiLevel = attributes.get("api_level", Type.INTEGER).toIntUnchecked();
      } catch (EvalException e) {
        throw new RepositoryFunctionException(e, Transience.PERSISTENT);
      }
      if (!apiLevels.contains(defaultApiLevel)) {
        throw new RepositoryFunctionException(
            new EvalException(
                rule.getLocation(),
                String.format(
                    "Android SDK api level %s was requested but it is not installed in the Android "
                        + "SDK at %s. The api levels found were %s. Please choose an available api "
                        + "level or install api level %s from the Android SDK Manager.",
                    defaultApiLevel,
                    androidSdkPath,
                    apiLevels.toString(),
                    defaultApiLevel)),
            Transience.PERSISTENT);
      }
    } else {
      // If the api_level attribute is not explicitly set, we select the highest api level that is
      // available in the SDK.
      defaultApiLevel = apiLevels.first();
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
          getDirectoryListing(androidSdkPath, BUILD_TOOLS_DIR, env);
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

    ImmutableSortedSet<PathFragment> androidDeviceSystemImageDirs =
        getAndroidDeviceSystemImageDirs(androidSdkPath, env);
    if (androidDeviceSystemImageDirs == null) {
      return null;
    }

    StringBuilder systemImageDirsList = new StringBuilder();
    for (PathFragment systemImageDir : androidDeviceSystemImageDirs) {
      systemImageDirsList.append(String.format("        \"%s\",\n", systemImageDir));
    }

    String template = getStringResource("android_sdk_repository_template.txt");

    String buildFile = template
        .replace("%repository_name%", rule.getName())
        .replace("%build_tools_version%", buildToolsVersion)
        .replace("%build_tools_directory%", buildToolsDirectory)
        .replace("%api_levels%", Iterables.toString(apiLevels))
        .replace("%default_api_level%", String.valueOf(defaultApiLevel))
        .replace("%system_image_dirs%", systemImageDirsList);

    // All local maven repositories that are shipped in the Android SDK.
    // TODO(ajmichael): Create SkyKeys so that if the SDK changes, this function will get rerun.
    Iterable<Path> localMavenRepositories =
        Lists.transform(LOCAL_MAVEN_REPOSITORIES, outputDirectory::getRelative);
    try {
      SdkMavenRepository sdkExtrasRepository =
          SdkMavenRepository.create(Iterables.filter(localMavenRepositories, Path::isDirectory));
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
    return workspace.getRelative(PathFragment.create(env.get(PATH_ENV_VAR))).asFragment();
  }

  private static String getStringResource(String name) {
    try {
      return ResourceFileLoader.loadResource(
          AndroidSdkRepositoryFunction.class, name);
    } catch (IOException e) {
      throw new IllegalStateException(e);
    }
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
    AndroidRevision newestBuildToolsRevision = null;
    for (Dirent buildToolsDirectory : buildToolsDirectories) {
      if (buildToolsDirectory.getType() != Dirent.Type.DIRECTORY) {
        continue;
      }
      try {
        AndroidRevision buildToolsRevision = AndroidRevision.parse(buildToolsDirectory.getName());
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

    SkyKey releaseFileKey =
        FileValue.key(RootedPath.toRootedPath(Root.fromPath(directory), sourcePropertiesFilePath));

    try {
      env.getValueOrThrow(releaseFileKey, IOException.class);

      Properties properties = new Properties();
      try (InputStream in = sourcePropertiesFilePath.getInputStream()) {
        properties.load(in);
      }
      return properties;
    } catch (IOException e) {
      String error = String.format(
          "Could not read %s in Android SDK: %s", sourcePropertiesFilePath, e.getMessage());
      throw new RepositoryFunctionException(new IOException(error), Transience.PERSISTENT);
    }
  }

  private static void assertValidBuildToolsVersion(Rule rule, String buildToolsVersion)
      throws EvalException {
    try {
      AndroidRevision buildToolsRevision = AndroidRevision.parse(buildToolsVersion);
      if (buildToolsRevision.compareTo(MIN_BUILD_TOOLS_REVISION) < 0) {
        throw new EvalException(
            rule.getLocation(),
            String.format(
                "Bazel requires Android build tools version %s or newer, %s was provided",
                MIN_BUILD_TOOLS_REVISION, buildToolsRevision));
      }
    } catch (NumberFormatException e) {
      throw new EvalException(
          rule.getLocation(),
          String.format(
              "Bazel does not recognize Android build tools version %s", buildToolsVersion),
          e);
    }
  }

  /**
   * Gets PathFragments for /sdk/system-images/*&#47;*&#47;*, which are the directories in the SDK
   * that contain system images needed for android_device.
   *
   * <p>If the sdk/system-images directory does not exist, an empty set is returned.
   */
  private ImmutableSortedSet<PathFragment> getAndroidDeviceSystemImageDirs(
      Path androidSdkPath, Environment env)
      throws RepositoryFunctionException, InterruptedException {
    if (!androidSdkPath.getRelative(SYSTEM_IMAGES_DIR).exists()) {
      return ImmutableSortedSet.of();
    }
    DirectoryListingValue systemImagesDirectoryValue =
        getDirectoryListing(androidSdkPath, SYSTEM_IMAGES_DIR, env);
    if (systemImagesDirectoryValue == null) {
      return null;
    }
    ImmutableMap<PathFragment, DirectoryListingValue> apiLevelSystemImageDirs =
        getSubdirectoryListingValues(
            androidSdkPath, SYSTEM_IMAGES_DIR, systemImagesDirectoryValue, env);
    if (apiLevelSystemImageDirs == null) {
      return null;
    }

    ImmutableSortedSet.Builder<PathFragment> pathFragments = ImmutableSortedSet.naturalOrder();
    for (PathFragment apiLevelDir : apiLevelSystemImageDirs.keySet()) {
      ImmutableMap<PathFragment, DirectoryListingValue> apiTypeSystemImageDirs =
          getSubdirectoryListingValues(
              androidSdkPath, apiLevelDir, apiLevelSystemImageDirs.get(apiLevelDir), env);
      if (apiTypeSystemImageDirs == null) {
        return null;
      }
      for (PathFragment apiTypeDir : apiTypeSystemImageDirs.keySet()) {
        for (Dirent architectureSystemImageDir :
            apiTypeSystemImageDirs.get(apiTypeDir).getDirents()) {
          pathFragments.add(apiTypeDir.getRelative(architectureSystemImageDir.getName()));
        }
      }
    }
    return pathFragments.build();
  }

  /**
   * Gets DirectoryListingValues for subdirectories of the directory or returns null.
   *
   * Ignores all non-directory files.
   */
  private static ImmutableMap<PathFragment, DirectoryListingValue> getSubdirectoryListingValues(
      final Path root, final PathFragment path, DirectoryListingValue directory, Environment env)
      throws RepositoryFunctionException, InterruptedException {
    Map<PathFragment, SkyKey> skyKeysForSubdirectoryLookups =
        Streams.stream(directory.getDirents())
            .filter(dirent -> dirent.getType().equals(Dirent.Type.DIRECTORY))
            .collect(
                toImmutableMap(
                    input -> path.getRelative(input.getName()),
                    input ->
                        DirectoryListingValue.key(
                            RootedPath.toRootedPath(
                                Root.fromPath(root),
                                root.getRelative(path).getRelative(input.getName())))));

    Map<SkyKey, ValueOrException<InconsistentFilesystemException>> values =
        env.getValuesOrThrow(
            skyKeysForSubdirectoryLookups.values(), InconsistentFilesystemException.class);

    ImmutableMap.Builder<PathFragment, DirectoryListingValue> directoryListingValues =
        new ImmutableMap.Builder<>();
    for (PathFragment pathFragment : skyKeysForSubdirectoryLookups.keySet()) {
      try {
        SkyValue skyValue = values.get(skyKeysForSubdirectoryLookups.get(pathFragment)).get();
        if (skyValue == null) {
          return null;
        }
        directoryListingValues.put(pathFragment, (DirectoryListingValue) skyValue);
      } catch (InconsistentFilesystemException e) {
        throw new RepositoryFunctionException(e, Transience.PERSISTENT);
      }
    }
    return directoryListingValues.build();
  }

  @Override
  protected void throwInvalidPathException(Path path, Exception e)
      throws RepositoryFunctionException {
    throw new RepositoryFunctionException(
        new IOException(
            String.format(
                "%s Unable to read the Android SDK at %s, the path may be invalid. Is "
                    + "the path in android_sdk_repository() or %s set correctly? If the path is "
                    + "correct, the contents in the Android SDK directory may have been modified.",
                e.getMessage(), path, PATH_ENV_VAR),
            e),
        Transience.PERSISTENT);
  }
}
