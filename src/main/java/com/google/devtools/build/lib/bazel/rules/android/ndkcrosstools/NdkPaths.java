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

package com.google.devtools.build.lib.bazel.rules.android.ndkcrosstools;

import com.google.common.base.Function;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration.Tool;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.ToolPath;
import java.util.Arrays;
import java.util.List;

/**
 * Class for creating paths that are specific to the structure of the Android NDK, but which are
 * common to all crosstool toolchains. 
 */
public class NdkPaths {

  /**
   * Removes the NDK repository prefix from the given path. Eg:
   * "external/%repositoryName%/ndk/a/b/c" -> "ndk/a/b/c"
   */
  public static String stripRepositoryPrefix(String path) {
    return path.split("/", 3)[2];
  }

  private final String repositoryName;
  private final String hostPlatform;
  private final ApiLevel apiLevel;

  public NdkPaths(String repositoryName, String hostPlatform, ApiLevel apiLevel) {
    this.repositoryName = repositoryName;
    this.hostPlatform = hostPlatform;
    this.apiLevel = apiLevel;
  }

  public ImmutableList<ToolPath> createToolpaths(String toolchainName, String targetPlatform,
      CppConfiguration.Tool... excludedTools) {

    ImmutableList.Builder<ToolPath> toolPaths = ImmutableList.builder();

    for (Tool tool : CppConfiguration.Tool.values()) {

      // Some toolchains don't have particular tools.
      if (!Arrays.asList(excludedTools).contains(tool)) {      

        String toolPath = createToolPath(toolchainName, targetPlatform + "-" + tool.getNamePart());

        toolPaths.add(ToolPath.newBuilder()
            .setName(tool.getNamePart())
            .setPath(toolPath)
            .build());
      }
    }

    return toolPaths.build();
  }

  public ImmutableList<ToolPath> createClangToolpaths(String toolchainName, String targetPlatform,
      String llvmVersion, CppConfiguration.Tool... excludedTools) {

    // Add GCC to the list of excluded tools. It will be replaced by clang below.
    excludedTools = ImmutableList.<CppConfiguration.Tool>builder()
        .add(excludedTools)
        .add(CppConfiguration.Tool.GCC)
        .build()
        .toArray(new CppConfiguration.Tool[excludedTools.length + 1]);

    // Create the regular tool paths, then add clang.
    return ImmutableList.<ToolPath>builder()
        .addAll(createToolpaths(toolchainName, targetPlatform, excludedTools))

        .add(ToolPath.newBuilder()
            .setName("gcc")
            .setPath(createToolPath(llvmVersion == null ? "llvm" : "llvm-" + llvmVersion, "clang"))
            .build())
        .build();
  }

  private String createToolPath(String toolchainName, String toolName) {

    String toolpathTemplate = "ndk/toolchains/%toolchainName%/prebuilt/%hostPlatform%"
        + "/bin/%toolName%";

    return toolpathTemplate
        .replace("%repositoryName%", repositoryName)
        .replace("%toolchainName%", toolchainName)
        .replace("%hostPlatform%", hostPlatform)
        .replace("%toolName%", toolName);
  }

  public static String getToolchainDirectoryFromToolPath(String toolPath) {
    return toolPath.split("/")[2];
  }

  public String createGccToolchainPath(String toolchainName) {

    String gccToolchainPathTemplate =
        "external/%repositoryName%/ndk/toolchains/%toolchainName%/prebuilt/%hostPlatform%";

    return gccToolchainPathTemplate
        .replace("%repositoryName%", repositoryName)
        .replace("%toolchainName%", toolchainName)
        .replace("%hostPlatform%", hostPlatform);
  }

  /**
   * Gets the clang NDK builtin includes directories that exist in the NDK. These directories are
   * always searched for header files by clang and should be added to the CROSSTOOL in the
   * cxx_builtin_include_directories list.
   *
   * <p>You can see the list of directories and the order that they are searched in by running
   * {@code clang -E -x c++ - -v < /dev/null}.
   */
  public String createClangToolchainBuiltinIncludeDirectory(String clangVersion) {
    String clangBuiltinIncludeDirectoryPathTemplate =
        "external/%repositoryName%/ndk/toolchains/llvm/prebuilt/%hostPlatform%/lib64/clang/"
            + "%clangVersion%/include";
    return clangBuiltinIncludeDirectoryPathTemplate
        .replace("%repositoryName%", repositoryName)
        .replace("%hostPlatform%", hostPlatform)
        .replace("%clangVersion%", clangVersion);
  }

  /**
   * Gets the gcc NDK builtin includes directories that exist in the NDK. These directories are
   * always searched for header files by clang and should be added to the CROSSTOOL in the
   * cxx_builtin_include_directories list.
   *
   * <p>You can see the list of directories and the order that they are searched in by running
   * {@code gcc -E -x c++ - -v < /dev/null}.
   */
  public List<String> createGccToolchainBuiltinIncludeDirectories(
      final String toolchainName, final String targetPlatform, final String gccVersion) {
    final String toolchainIncludePathTemplate =
        "external/%repositoryName%/ndk/toolchains/%toolchainName%/prebuilt/%hostPlatform%"
            + "/lib/gcc/%targetPlatform%/%gccVersion%/%includeFolderName%";
    return Lists.transform(
        ImmutableList.of("include", "include-fixed"),
        new Function<String, String>() {
          @Override
          public String apply(String includeFolderName) {
            return toolchainIncludePathTemplate
                .replace("%repositoryName%", repositoryName)
                .replace("%toolchainName%", toolchainName)
                .replace("%hostPlatform%", hostPlatform)
                .replace("%targetPlatform%", targetPlatform)
                .replace("%gccVersion%", gccVersion)
                .replace("%includeFolderName%", includeFolderName);
          }
        });
  }

  public String createBuiltinSysroot(String targetCpu) {

    String correctedApiLevel = apiLevel.getCpuCorrectedApiLevel(targetCpu);

    String androidPlatformIncludePathTemplate =
        "external/%repositoryName%/ndk/platforms/android-%apiLevel%/arch-%arch%";

    return androidPlatformIncludePathTemplate
        .replace("%repositoryName%", repositoryName)
        .replace("%apiLevel%", correctedApiLevel)
        .replace("%arch%", targetCpu);
  }

  ImmutableList<String> createGnuLibstdcIncludePaths(String gccVersion, String targetCpu) {

    String cpuNoThumb = targetCpu.replaceAll("-thumb$", "");

    String prefix = "external/%repositoryName%/ndk/sources/cxx-stl/gnu-libstdc++/%gccVersion%/";
    List<String> includePathTemplates = Arrays.asList(
        prefix + "include",
        prefix + "libs/%targetCpu%/include",
        prefix + "include/backward");

    ImmutableList.Builder<String> includePaths = ImmutableList.builder();
    for (String template : includePathTemplates) {
      includePaths.add(
          template
            .replace("%repositoryName%", repositoryName)
            .replace("%gccVersion%", gccVersion)
            .replace("%targetCpu%", cpuNoThumb));
    }
    return includePaths.build();
  }

  ImmutableList<String> createStlportIncludePaths() {

    String prefix =
        "external/%repositoryName%/ndk/sources/cxx-stl/"
            .replace("%repositoryName%", repositoryName);

    return ImmutableList.of(prefix + "stlport/stlport", prefix + "gabi++/include");
  }

  ImmutableList<String> createLibcxxIncludePaths() {

    String prefix =
        "external/%repositoryName%/ndk/sources/".replace("%repositoryName%", repositoryName);

    return ImmutableList.of(
        prefix + "cxx-stl/llvm-libc++/libcxx/include",
        prefix + "cxx-stl/llvm-libc++abi/libcxxabi/include",
        prefix + "android/support/include");
  }

  /**
   * @param stl The STL name as it appears in the NDK path
   * @param gccVersion The GCC version "4.8" or "4.9", applicable only to gnu-libstdc++, or null
   * @param targetCpu Target CPU
   * @param fileExtension "a" or "so"
   * @return A glob pattern for the STL runtime libs in the NDK.
   */
  static String createStlRuntimeLibsGlob(
      String stl, String gccVersion, String targetCpu, String fileExtension) {
    
    if (gccVersion != null) {
      stl += "/" + gccVersion;
    }

    targetCpu = targetCpu.replaceAll("-thumb$", "/thumb");

    String template =
        "ndk/sources/cxx-stl/%stl%/libs/%targetCpu%/*.%fileExtension%";
    return template
        .replace("%stl%", stl)
        .replace("%targetCpu%", targetCpu)
        .replace("%fileExtension%", fileExtension);
  }
}
