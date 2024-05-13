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

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedSet;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.bazel.rules.android.ndkcrosstools.AndroidNdkCrosstools;
import com.google.devtools.build.lib.bazel.rules.android.ndkcrosstools.AndroidNdkCrosstools.NdkCrosstoolsException;
import com.google.devtools.build.lib.bazel.rules.android.ndkcrosstools.ApiLevel;
import com.google.devtools.build.lib.bazel.rules.android.ndkcrosstools.NdkMajorRevision;
import com.google.devtools.build.lib.bazel.rules.android.ndkcrosstools.NdkPaths;
import com.google.devtools.build.lib.bazel.rules.android.ndkcrosstools.NdkRelease;
import com.google.devtools.build.lib.bazel.rules.android.ndkcrosstools.StlImpl;
import com.google.devtools.build.lib.bazel.rules.android.ndkcrosstools.StlImpls;
import com.google.devtools.build.lib.bazel.rules.android.ndkcrosstools.StlImpls.GnuLibStdCppStlImpl;
import com.google.devtools.build.lib.bazel.rules.android.ndkcrosstools.StlImpls.LibCppStlImpl;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.rules.repository.RepoRecordedInput;
import com.google.devtools.build.lib.rules.repository.RepositoryDirectoryValue;
import com.google.devtools.build.lib.rules.repository.WorkspaceAttributeMapper;
import com.google.devtools.build.lib.skyframe.DirectoryListingValue;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.build.lib.util.ResourceFileLoader;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CToolchain;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CompilationMode;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CompilationModeFlags;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CrosstoolRelease;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.ToolPath;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkSemantics;

/** Implementation of the {@code android_ndk_repository} rule. */
public class AndroidNdkRepositoryFunction extends AndroidRepositoryFunction {

  private static final String TOOLCHAIN_NAME_PREFIX = "toolchain-";
  private static final String PATH_ENV_VAR = "ANDROID_NDK_HOME";
  private static final PathFragment PLATFORMS_DIR = PathFragment.create("platforms");

  private static final ImmutableSet<String> PATH_ENV_VAR_AS_SET = ImmutableSet.of(PATH_ENV_VAR);

  private static String getDefaultCrosstool(int majorRevision) {
    // From NDK 17, libc++ replaces gnu-libstdc++ as the default STL.
    return majorRevision <= 16 ? GnuLibStdCppStlImpl.NAME : LibCppStlImpl.NAME;
  }

  private static PathFragment getAndroidNdkHomeEnvironmentVar(
      Path workspace, Map<String, String> env) {
    return workspace.getRelative(PathFragment.create(env.get(PATH_ENV_VAR))).asFragment();
  }

  private static String createBuildFile(
      String ruleName, String defaultCrosstool, List<CrosstoolStlPair> crosstools) {

    String buildFileTemplate = getTemplate("android_ndk_build_file_template.txt");
    String ccToolchainTemplate = getTemplate("android_ndk_cc_toolchain_template.txt");
    String stlFilegroupTemplate = getTemplate("android_ndk_stl_filegroup_template.txt");
    String vulkanValidationLayersTemplate =
        getTemplate("android_ndk_vulkan_validation_layers_template.txt");
    String miscLibrariesTemplate = getTemplate("android_ndk_misc_libraries_template.txt");

    StringBuilder ccToolchainRules = new StringBuilder();
    StringBuilder stlFilegroups = new StringBuilder();
    StringBuilder vulkanValidationLayers = new StringBuilder();
    for (CrosstoolStlPair crosstoolStlPair : crosstools) {

      // Create the cc_toolchain_suite rule
      CrosstoolRelease crosstool = crosstoolStlPair.crosstoolRelease;

      StringBuilder toolchainMap = new StringBuilder();
      for (CToolchain toolchain : crosstool.getToolchainList()) {
        toolchainMap.append(
            String.format(
                "      '%s': ':%s',\n      '%s|%s': ':%s',\n",
                toolchain.getTargetCpu(),
                toolchain.getToolchainIdentifier(),
                toolchain.getTargetCpu(),
                toolchain.getCompiler(),
                toolchain.getToolchainIdentifier()));
      }

      // Create the cc_toolchain rules
      for (CToolchain toolchain : crosstool.getToolchainList()) {
        ccToolchainRules.append(
            createCcToolchainRule(
                ccToolchainTemplate, crosstoolStlPair.stlImpl.getName(), toolchain));
      }

      // Create the STL file group rules
      for (Map.Entry<String, String> entry :
          crosstoolStlPair.stlImpl.getFilegroupNamesAndFilegroupFileGlobPatterns().entrySet()) {

        stlFilegroups.append(
            stlFilegroupTemplate
                .replace("%name%", entry.getKey())
                .replace("%fileGlobPattern%", entry.getValue()));
      }

      // Create the Vulkan validation layers libraries
      for (CToolchain toolchain : crosstool.getToolchainList()) {
        vulkanValidationLayers.append(
            vulkanValidationLayersTemplate
                .replace("%toolchainName%", toolchain.getToolchainIdentifier())
                .replace("%cpu%", toolchain.getTargetCpu()));
      }
    }

    return buildFileTemplate
        .replace("%ruleName%", ruleName)
        .replace("%defaultCrosstool%", "//:toolchain-" + defaultCrosstool)
        .replace("%ccToolchainRules%", ccToolchainRules)
        .replace("%stlFilegroups%", stlFilegroups)
        .replace("%vulkanValidationLayers%", vulkanValidationLayers)
        .replace("%miscLibraries%", miscLibrariesTemplate);
  }

  static String createToolchainName(String stlName) {
    return TOOLCHAIN_NAME_PREFIX + stlName;
  }

  private static String createCcToolchainRule(
      String ccToolchainTemplate, String version, CToolchain toolchain) {

    // TODO(bazel-team): It's unfortunate to have to extract data from a CToolchain proto like this.
    // It would be better to have a higher-level construction (like an AndroidToolchain class)
    // from which the CToolchain proto and rule information needed here can be created.
    // Alternatively it would be nicer to just be able to glob the entire NDK and add that one glob
    // to each cc_toolchain rule, and then the complexities in the method and the templates can
    // go away, but globbing the entire NDK takes ~60 seconds, mostly because of MD5ing all the
    // binary files in the NDK (eg the .so / .a / .o files).

    // This also includes the files captured with cxx_builtin_include_directory.
    // Use gcc specifically because clang toolchains will have both gcc and llvm toolchain paths,
    // but the gcc tool will actually be clang.
    ToolPath gcc = null;
    for (ToolPath toolPath : toolchain.getToolPathList()) {
      if (toolPath.getName().equals("gcc")) {
        gcc = toolPath;
      }
    }
    checkNotNull(gcc, "gcc not found in crosstool toolpaths");
    String toolchainDirectory = NdkPaths.getToolchainDirectoryFromToolPath(gcc.getPath());

    // Create file glob patterns for the various files that the toolchain references.

    String androidPlatformIncludes =
        NdkPaths.stripRepositoryPrefix(toolchain.getBuiltinSysroot()) + "/**/*";

    List<String> toolchainFileGlobPatterns = new ArrayList<>();
    toolchainFileGlobPatterns.add(androidPlatformIncludes);

    for (String cxxFlag : toolchain.getUnfilteredCxxFlagList()) {
      if (!cxxFlag.startsWith("-")) { // Skip flag names
        toolchainFileGlobPatterns.add(NdkPaths.stripRepositoryPrefix(cxxFlag) + "/**/*");
      }
    }

    // For NDK 15 and up. Unfortunately, the toolchain does not encode the NDK revision number.
    toolchainFileGlobPatterns.add("ndk/sysroot/**/*");

    // If this is a clang toolchain, also add the corresponding gcc toolchain to the globs.
    int gccToolchainIndex = toolchain.getCompilerFlagList().indexOf("-gcc-toolchain");
    if (gccToolchainIndex > -1) {
      String gccToolchain = toolchain.getCompilerFlagList().get(gccToolchainIndex + 1);
      toolchainFileGlobPatterns.add(NdkPaths.stripRepositoryPrefix(gccToolchain) + "/**/*");
    }

    StringBuilder toolchainFileGlobs = new StringBuilder();
    for (String toolchainFileGlobPattern : toolchainFileGlobPatterns) {
      toolchainFileGlobs.append(String.format("        \"%s\",\n", toolchainFileGlobPattern));
    }

    return ccToolchainTemplate
        .replace("%toolchainName%", toolchain.getToolchainIdentifier())
        .replace("%cpu%", toolchain.getTargetCpu())
        .replace("%platform_cpu%", getPlatformCpuLabel(toolchain.getTargetCpu()))
        .replace("%compiler%", toolchain.getCompiler())
        .replace("%version%", version)
        .replace("%dynamicRuntimeLibs%", toolchain.getDynamicRuntimesFilegroup())
        .replace("%staticRuntimeLibs%", toolchain.getStaticRuntimesFilegroup())
        .replace("%toolchainDirectory%", toolchainDirectory)
        .replace("%toolchainFileGlobs%", toolchainFileGlobs.toString().trim());
  }

  private static String getPlatformCpuLabel(String targetCpu) {
    // Create a mapping of CcToolchain CPU values to platform arch constraint values
    // in @platforms//cpu
    switch (targetCpu) {
      case "x86":
        return "x86_32";
      case "armeabi-v7a":
        return "armv7";
      case "arm64-v8a":
        return "aarch64";
      default:
        return "x86_64";
    }
  }

  private static String getTemplate(String templateFile) {
    try {
      return ResourceFileLoader.loadResource(AndroidNdkRepositoryFunction.class, templateFile);
    } catch (IOException e) {
      throw new IllegalStateException(e);
    }
  }

  @Override
  public boolean isLocal(Rule rule) {
    return true;
  }

  @Override
  public boolean verifyRecordedInputs(
      Rule rule,
      BlazeDirectories directories,
      Map<RepoRecordedInput, String> recordedInputValues,
      Environment env)
      throws InterruptedException {
    WorkspaceAttributeMapper attributes = WorkspaceAttributeMapper.of(rule);
    if (attributes.isAttributeValueExplicitlySpecified("path")) {
      return true;
    }
    return super.verifyRecordedInputs(rule, directories, recordedInputValues, env);
  }

  @Override
  @Nullable
  public RepositoryDirectoryValue.Builder fetch(
      Rule rule,
      Path outputDirectory,
      BlazeDirectories directories,
      Environment env,
      Map<RepoRecordedInput, String> recordedInputValues,
      SkyKey key)
      throws InterruptedException, RepositoryFunctionException {
    ensureNativeRepoRuleEnabled(rule, env, "https://github.com/bazelbuild/rules_android_ndk");
    Map<String, String> environ =
        declareEnvironmentDependencies(recordedInputValues, env, PATH_ENV_VAR_AS_SET);
    if (environ == null) {
      return null;
    }
    try {
      outputDirectory.createDirectoryAndParents();
      FileSystemUtils.createEmptyFile(outputDirectory.getRelative(LabelConstants.REPO_FILE_NAME));
    } catch (IOException e) {
      throw new RepositoryFunctionException(e, Transience.TRANSIENT);
    }
    WorkspaceAttributeMapper attributes = WorkspaceAttributeMapper.of(rule);
    PathFragment pathFragment;
    String userDefinedPath = null;
    if (attributes.isAttributeValueExplicitlySpecified("path")) {
      userDefinedPath = getPathAttr(rule);
      pathFragment = getTargetPath(userDefinedPath, directories.getWorkspace());
    } else if (environ.get(PATH_ENV_VAR) != null) {
      userDefinedPath = environ.get(PATH_ENV_VAR);
      pathFragment = getAndroidNdkHomeEnvironmentVar(directories.getWorkspace(), environ);
    } else {
      throw new RepositoryFunctionException(
          Starlark.errorf(
              "Either the path attribute of android_ndk_repository or the ANDROID_NDK_HOME"
                  + " environment variable must be set."),
          Transience.PERSISTENT);
    }

    Path ndkSymlinkTreeDirectory = outputDirectory.getRelative("ndk");
    try {
      ndkSymlinkTreeDirectory.createDirectory();
    } catch (IOException e) {
      throw new RepositoryFunctionException(e, Transience.TRANSIENT);
    }

    Path ndkHome = directories.getOutputBase().getFileSystem().getPath(pathFragment);
    if (!symlinkLocalRepositoryContents(ndkSymlinkTreeDirectory, ndkHome, userDefinedPath)) {
      return null;
    }

    String ruleName = rule.getName();

    // We need to fetch the NDK release info from the actual home to avoid cycle in the
    // dependency graph (the path relative to the repository root depends on the
    // repository being fetched).
    NdkRelease ndkRelease = getNdkRelease(ndkHome, env);
    if (env.valuesMissing()) {
      return null;
    }

    String apiLevelString;
    if (attributes.isAttributeValueExplicitlySpecified("api_level")) {
      try {
        apiLevelString = attributes.get("api_level", Type.INTEGER).toString();
      } catch (EvalException e) {
        throw new RepositoryFunctionException(e, Transience.PERSISTENT);
      }
    } else {
      DirectoryListingValue platformsDirectoryValue =
          getDirectoryListing(ndkHome, PLATFORMS_DIR, env);
      if (platformsDirectoryValue == null) {
        return null;
      }

      ImmutableSortedSet<Integer> apiLevels = getApiLevels(platformsDirectoryValue.getDirents());
      if (apiLevels.isEmpty()) {
        // Every Android NDK to date ships with multiple api levels, so the only reason that this
        // would be empty is if the user is not pointing to a standard NDK or has tinkered with it
        // themselves.
        throwInvalidPathException(
            ndkHome,
            Starlark.errorf(
                "android_ndk_repository requires that at least one Android platform is present in"
                    + " the Android NDK platforms directory."));
      }
      apiLevelString = apiLevels.first().toString();
    }

    // NDK minor revisions should be backwards compatible within a major revision, the crosstools
    // we generate don't care about the minor revision.
    NdkMajorRevision ndkMajorRevision;
    if (!ndkRelease.isValid) {
      String warningMessage =
          String.format(
              "The revision of the Android NDK referenced by android_ndk_repository rule '%s' "
                  + "could not be determined (the revision string found is '%s'). "
                  + "Bazel will attempt to treat the NDK as if it was r%s. This may cause "
                  + "compilation and linkage problems. Please download a supported NDK version.\n",
              ruleName,
              ndkRelease.rawRelease,
              AndroidNdkCrosstools.LATEST_KNOWN_REVISION.getKey());
      env.getListener().handle(Event.warn(warningMessage));
      ndkMajorRevision = AndroidNdkCrosstools.LATEST_KNOWN_REVISION.getValue();
    } else if (!AndroidNdkCrosstools.isKnownNDKRevision(ndkRelease)) {
      String warningMessage =
          String.format(
              "The major revision of the Android NDK referenced by android_ndk_repository rule "
                  + "'%s' is %s. The major revisions supported by Bazel are %s. Bazel will attempt "
                  + "to treat the NDK as if it was r%s. This may cause compilation and linkage "
                  + "problems. Please download a supported NDK version.\n",
              ruleName,
              ndkRelease.majorRevision,
              AndroidNdkCrosstools.KNOWN_NDK_MAJOR_REVISIONS.keySet(),
              AndroidNdkCrosstools.LATEST_KNOWN_REVISION.getKey());
      env.getListener().handle(Event.warn(warningMessage));
      ndkMajorRevision = AndroidNdkCrosstools.LATEST_KNOWN_REVISION.getValue();
    } else {
      ndkMajorRevision =
          AndroidNdkCrosstools.KNOWN_NDK_MAJOR_REVISIONS.get(ndkRelease.majorRevision);
    }

    StarlarkSemantics starlarkSemantics = PrecomputedValue.STARLARK_SEMANTICS.get(env);
    if (starlarkSemantics == null) {
      return null;
    }
    boolean siblingRepositoryLayout =
        starlarkSemantics.getBool(BuildLanguageOptions.EXPERIMENTAL_SIBLING_REPOSITORY_LAYOUT);

    ApiLevel apiLevel = ndkMajorRevision.apiLevel(env.getListener(), ruleName, apiLevelString);

    ImmutableList.Builder<CrosstoolStlPair> crosstoolsAndStls = ImmutableList.builder();
    try {

      String hostPlatform = AndroidNdkCrosstools.getHostPlatform(ndkRelease);
      NdkPaths ndkPaths =
          new NdkPaths(
              ruleName, hostPlatform, apiLevel, ndkRelease.majorRevision, siblingRepositoryLayout);

      for (StlImpl stlImpl : StlImpls.get(ndkPaths, ndkRelease.majorRevision)) {
        CrosstoolRelease crosstoolRelease =
            ndkMajorRevision.crosstoolRelease(ndkPaths, stlImpl, hostPlatform);
        crosstoolsAndStls.add(new CrosstoolStlPair(crosstoolRelease, stlImpl));
      }

    } catch (NdkCrosstoolsException e) {
      throw new RepositoryFunctionException(new IOException(e), Transience.PERSISTENT);
    }

    String defaultCrosstool = getDefaultCrosstool(ndkRelease.majorRevision);

    ImmutableList<CrosstoolStlPair> crosstoolStlPairs = crosstoolsAndStls.build();
    String buildFile = createBuildFile(ruleName, defaultCrosstool, crosstoolStlPairs);
    writeBuildFile(outputDirectory, buildFile);
    ImmutableList.Builder<String> bigConditional = ImmutableList.builder();
    for (CrosstoolStlPair pair : crosstoolStlPairs) {
      for (CToolchain toolchain : pair.crosstoolRelease.getToolchainList()) {
        bigConditional.addAll(generateBzlConfigFor(pair.stlImpl.getName(), toolchain));
      }
    }
    writeFile(
        outputDirectory,
        "cc_toolchain_config.bzl",
        getTemplate("android_ndk_cc_toolchain_config_template.txt")
            .replaceAll(
                "%big_conditional_populating_variables%",
                Joiner.on("\n" + "    ").join(bigConditional.build())));
    return RepositoryDirectoryValue.builder().setPath(outputDirectory);
  }

  private ImmutableList<String> generateBzlConfigFor(String version, CToolchain toolchain) {
    ImmutableList.Builder<String> bigConditional = ImmutableList.builder();
    String cpu = toolchain.getTargetCpu();
    String compiler = toolchain.getCompiler();

    Preconditions.checkArgument(
        toolchain.getLinkingModeFlagsCount() == 0, "linking_mode_flags not supported.");
    Preconditions.checkArgument(
        toolchain.getActionConfigCount() == 0, "action_configs not supported.");
    Preconditions.checkArgument(toolchain.getFeatureCount() == 0, "features not supported.");
    Preconditions.checkArgument(toolchain.getArFlagCount() == 0, "ar_flags not supported.");
    Preconditions.checkArgument(
        toolchain.getArtifactNamePatternCount() == 0, "artifact_name_patterns not supported.");
    Preconditions.checkArgument(toolchain.getCxxFlagCount() == 0, "cxx_flags not supported.");
    Preconditions.checkArgument(
        toolchain.getDynamicLibraryLinkerFlagCount() == 0,
        "dynamic_library_linker_flags not supported.");
    Preconditions.checkArgument(
        toolchain.getLdEmbedFlagCount() == 0, "ld_embed_flags not supported.");
    Preconditions.checkArgument(
        toolchain.getObjcopyEmbedFlagCount() == 0, "objcopy_embed_flags not supported.");
    Preconditions.checkArgument(
        toolchain.getMakeVariableCount() == 0, "make_variables not supported.");
    Preconditions.checkArgument(
        toolchain.getTestOnlyLinkerFlagCount() == 0, "test_only_linker_flags not supported.");

    CompilationModeFlags fastbuild = null;
    CompilationModeFlags dbg = null;
    CompilationModeFlags opt = null;
    for (CompilationModeFlags flags : toolchain.getCompilationModeFlagsList()) {
      Preconditions.checkArgument(
          flags.getCxxFlagCount() == 0, "compilation_mode_flags.cxx_flags not supported.");
      Preconditions.checkArgument(
          flags.getLinkerFlagCount() == 0, "compilation_mode_flags.linker_flags not supported.");
      if (flags.getMode().equals(CompilationMode.FASTBUILD)) {
        fastbuild = flags;
      } else if (flags.getMode().equals(CompilationMode.DBG)) {
        dbg = flags;
      } else if (flags.getMode().equals(CompilationMode.OPT)) {
        opt = flags;
      }
    }

    bigConditional.add(
        String.format(
            "if cpu == '%s' and compiler == '%s' and version == '%s':", cpu, compiler, version),
        String.format(
            "  default_compile_flags = [%s]",
            toSequenceOfStarlarkStrings(toolchain.getCompilerFlagList())),
        String.format(
            "  unfiltered_compile_flags = [%s]",
            toSequenceOfStarlarkStrings(toolchain.getUnfilteredCxxFlagList())),
        String.format(
            "  default_link_flags = [%s]",
            toSequenceOfStarlarkStrings(toolchain.getLinkerFlagList())),
        String.format(
            "  default_fastbuild_flags = [%s]",
            toSequenceOfStarlarkStrings(
                fastbuild != null ? fastbuild.getCompilerFlagList() : ImmutableList.of())),
        String.format(
            "  default_dbg_flags = [%s]",
            toSequenceOfStarlarkStrings(
                dbg != null ? dbg.getCompilerFlagList() : ImmutableList.of())),
        String.format(
            "  default_opt_flags = [%s]",
            toSequenceOfStarlarkStrings(
                opt != null ? opt.getCompilerFlagList() : ImmutableList.of())),
        String.format(
            "  cxx_builtin_include_directories = [%s]",
            toSequenceOfStarlarkStrings(toolchain.getCxxBuiltinIncludeDirectoryList())),
        String.format("  target_cpu = '%s'", toolchain.getTargetCpu()),
        String.format("  toolchain_identifier = '%s'", toolchain.getToolchainIdentifier()),
        String.format("  host_system_name = '%s'", toolchain.getHostSystemName()),
        String.format("  target_system_name = '%s'", toolchain.getTargetSystemName()),
        String.format("  target_libc = '%s'", toolchain.getTargetLibc()),
        String.format("  target_compiler = '%s'", toolchain.getCompiler()),
        String.format("  abi_version = '%s'", toolchain.getAbiVersion()),
        String.format("  abi_libc_version = '%s'", toolchain.getAbiLibcVersion()),
        String.format("  builtin_sysroot = '%s'", toolchain.getBuiltinSysroot()));
    bigConditional.addAll(
        toolchain.getToolPathList().stream()
            .map(
                tp ->
                    String.format(
                        "  %s_path = '%s'",
                        tp.getName().toLowerCase(Locale.ROOT).replaceAll("-", "_"), tp.getPath()))
            .collect(ImmutableList.toImmutableList()));
    return bigConditional.add("").build();
  }

  private String toSequenceOfStarlarkStrings(Iterable<String> flags) {
    return "'" + Joiner.on("', '").join(flags) + "'";
  }

  @Override
  public Class<? extends RuleDefinition> getRuleDefinition() {
    return AndroidNdkRepositoryRule.class;
  }

  private NdkRelease getNdkRelease(Path directory, Environment env)
      throws RepositoryFunctionException, InterruptedException {

    // For NDK r11+
    Path releaseFilePath = directory.getRelative("source.properties");
    if (!releaseFilePath.exists()) {
      // For NDK r10e
      releaseFilePath = directory.getRelative("RELEASE.TXT");
    }

    SkyKey releaseFileKey =
        FileValue.key(RootedPath.toRootedPath(Root.fromPath(directory), releaseFilePath));

    String releaseFileContent = "";
    try {
      env.getValueOrThrow(releaseFileKey, IOException.class);

      releaseFileContent =
          new String(FileSystemUtils.readContent(releaseFilePath), StandardCharsets.UTF_8);
    } catch (IOException e) {
      throwInvalidPathException(
          directory,
          new IOException(
              "Could not read "
                  + releaseFilePath.getBaseName()
                  + " in Android NDK: "
                  + e.getMessage()));
    }

    return NdkRelease.create(releaseFileContent.trim());
  }

  @Override
  protected void throwInvalidPathException(Path path, Exception e)
      throws RepositoryFunctionException {
    throw new RepositoryFunctionException(
        new IOException(
            String.format(
                "%s Unable to read the Android NDK at %s, the path may be invalid. Is "
                    + "the path in android_ndk_repository() or %s set correctly? If the path is "
                    + "correct, the contents in the Android NDK directory may have been modified.",
                e.getMessage(), path, PATH_ENV_VAR),
            e),
        Transience.PERSISTENT);
  }

  private static final class CrosstoolStlPair {

    private final CrosstoolRelease crosstoolRelease;
    private final StlImpl stlImpl;

    private CrosstoolStlPair(CrosstoolRelease crosstoolRelease, StlImpl stlImpl) {
      this.crosstoolRelease = crosstoolRelease;
      this.stlImpl = stlImpl;
    }
  }
}
