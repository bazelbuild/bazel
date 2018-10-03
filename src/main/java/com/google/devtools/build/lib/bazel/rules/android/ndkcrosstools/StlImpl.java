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

import com.google.common.base.Verify;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CToolchain;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * An NdkStlImpl adds a specific Android NDK C++ STL implementation to a given CToolchain proto
 * builder.
 */
public abstract class StlImpl {

  private enum RuntimeType {
    DYNAMIC("so"), STATIC("a");

    private final String name;
    private final String fileExtension;

    RuntimeType(String fileExtension) {
      this.name = name().toLowerCase();
      this.fileExtension = fileExtension;
    }    
  }

  private final Map<String, String> fileGroupNameToFileGlobs = new LinkedHashMap<>();

  protected final String name;
  protected final NdkPaths ndkPaths;

  /**
   * @param name the name for this STL, which should be safe to use as a label
   * @param ndkPaths the NdkPaths
   */
  protected StlImpl(String name, NdkPaths ndkPaths) {
    this.name = name;
    this.ndkPaths = ndkPaths;
  }

  public void addStlImpl(List<CToolchain.Builder> baseToolchains, @Nullable String gccVersion) {

    for (CToolchain.Builder baseToolchain : baseToolchains) {
      addStlImpl(baseToolchain, gccVersion);
    }    
  }

  /**
   * Adds an Android NDK C++ STL implementation to the given CToolchain builder.
   *
   * @param toolchain the toolchain to add the STL implementation to
   * @param gccVersion the gcc version for the STL impl. Applicable only to gnu-libstdc++
   */
  public abstract void addStlImpl(CToolchain.Builder toolchain, @Nullable String gccVersion);

  protected void addBaseStlImpl(CToolchain.Builder toolchain, @Nullable String gccVersion) {

    toolchain
      .setToolchainIdentifier(toolchain.getToolchainIdentifier() + "-" + name)
      .setSupportsEmbeddedRuntimes(true)
      .setDynamicRuntimesFilegroup(
          createRuntimeLibrariesFilegroup(
              name, gccVersion, toolchain.getTargetCpu(), RuntimeType.DYNAMIC))
      .setStaticRuntimesFilegroup(
          createRuntimeLibrariesFilegroup(
              name, gccVersion, toolchain.getTargetCpu(), RuntimeType.STATIC));
  }

  private String createRuntimeLibrariesFilegroup(
      String stl, @Nullable String gccVersion, String targetCpu, RuntimeType type) {

    // gnu-libstlc++ has separate libraries for 4.8 and 4.9
    String fullStlName = stl;
    if (gccVersion != null) {
      fullStlName += "-" + gccVersion;
    }

    String filegroupNameTemplate = "%stl%-%targetCpu%-%type%-runtime-libraries";
    String filegroupName = filegroupNameTemplate
        .replace("%stl%", fullStlName)
        .replace("%targetCpu%", targetCpu)
        .replace("%type%", type.name);

    // At the same time that the filegroup name is created, record a corresponding file glob
    // pattern that AndroidNdkRepositoryFunction can look up later for creating the build file
    // rules.

    // These translations are unfortunate, but toolchain identifiers aren't allowed to have '+' in
    // their name.
    if (stl.equals("gnu-libstdcpp")) {
      stl = "gnu-libstdc++";
    }
    if (stl.equals("libcpp")) {
      stl = "llvm-libc++";
    }

    String glob = NdkPaths.createStlRuntimeLibsGlob(stl, gccVersion, targetCpu, type.fileExtension);
    String previousValue = fileGroupNameToFileGlobs.put(filegroupName, glob);

    // Some STL filegroups will end up being duplicates, but a filegroup should never be registered
    // with a different glob, otherwise one toolchain would get the wrong glob.
    Verify.verify(previousValue == null || previousValue.equals(glob),
        "STL filegroup glob being replaced with a different glob:\nname: %s\n%s\n%s",
        filegroupName, glob, previousValue);
    
    return filegroupName;
  }

  protected static Iterable<String> createIncludeFlags(Iterable<String> includePaths) {
    ImmutableList.Builder<String> includeFlags = ImmutableList.builder();
    for (String includePath : includePaths) {
      includeFlags.add("-isystem");
      includeFlags.add(includePath);
    }
    return includeFlags.build();
  }

  /**
   * @return a map of the names of the generated STL runtime library filegroup names to their file
   * glob patterns
   */
  public Map<String, String> getFilegroupNamesAndFilegroupFileGlobPatterns() {
    return ImmutableMap.copyOf(fileGroupNameToFileGlobs);
  }

  /**
   * @return the name of this STL impl, which is safe to use as a label
   */
  public String getName() {
    return name;
  }
}

