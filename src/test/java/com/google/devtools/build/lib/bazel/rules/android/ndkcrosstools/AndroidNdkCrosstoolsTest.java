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

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.fail;

import com.google.common.base.Function;
import com.google.common.base.Joiner;
import com.google.common.collect.Collections2;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableMultimap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.bazel.rules.android.ndkcrosstools.r10e.NdkMajorRevisionR10;
import com.google.devtools.build.lib.bazel.rules.android.ndkcrosstools.r12.NdkMajorRevisionR12;
import com.google.devtools.build.lib.events.NullEventHandler;
import com.google.devtools.build.lib.util.ResourceFileLoader;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CToolchain;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CrosstoolRelease;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.ToolPath;
import java.io.IOException;
import java.util.Collection;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

/** Tests for {@link AndroidNdkCrosstools}. */
@RunWith(Parameterized.class)
public class AndroidNdkCrosstoolsTest {
  private static final String HOST_PLATFORM = "linux-x86_64";
  private static final String REPOSITORY_NAME = "testrepository";

  private static class AndroidNdkCrosstoolsTestParams {
    private final ApiLevel apiLevel;
    private final NdkRelease ndkRelease;
    // ndkfiles.txt contains a list of every file in the ndk, created using this command at the
    // root of the Android NDK for version r10e (64-bit):
    //     find . -xtype f | sed 's|^\./||' | sort
    // and similarly for ndkdirectories, except "-xtype d" is used.
    //
    // It's unfortunate to have files like these, since they're large and brittle, but since the
    // whole NDK can't be checked in to test against, it's about the most that can be done right
    // now.
    private final String ndkFilesFilename;
    private final String ndkDirectoriesFilename;

    AndroidNdkCrosstoolsTestParams(
        ApiLevel apiLevel,
        NdkRelease ndkRelease,
        String ndkFilesFilename,
        String ndkDirectoriesFilename) {
      this.apiLevel = apiLevel;
      this.ndkRelease = ndkRelease;
      this.ndkFilesFilename = ndkFilesFilename;
      this.ndkDirectoriesFilename = ndkDirectoriesFilename;
    }

    NdkMajorRevision getNdkMajorRevision() {
      return AndroidNdkCrosstools.KNOWN_NDK_MAJOR_REVISIONS.get(ndkRelease.majorRevision);
    }

    Integer getNdkMajorRevisionNumber() {
      return ndkRelease.majorRevision;
    }

    ImmutableSet<String> getNdkFiles() throws IOException {
      String ndkFilesFileContent =
          ResourceFileLoader.loadResource(AndroidNdkCrosstoolsTest.class, ndkFilesFilename);
      ImmutableSet.Builder<String> ndkFiles = ImmutableSet.builder();
      for (String line : ndkFilesFileContent.split("\n")) {
        // The contents of the NDK are placed at "external/%repositoryName%/ndk".
        // The "external/%repositoryName%" part is removed using NdkPaths.stripRepositoryPrefix,
        // but to make it easier the "ndk/" part is added here.
        ndkFiles.add("ndk/" + line.trim());
      }
      return ndkFiles.build();
    }

    ImmutableSet<String> getNdkDirectories() throws IOException {
      String ndkFilesFileContent =
          ResourceFileLoader.loadResource(AndroidNdkCrosstoolsTest.class, ndkDirectoriesFilename);
      ImmutableSet.Builder<String> ndkDirectories = ImmutableSet.builder();
      for (String line : ndkFilesFileContent.split("\n")) {
        ndkDirectories.add("ndk/" + line.trim());
      }
      return ndkDirectories.build();
    }
  }

  @Parameters
  public static Collection<AndroidNdkCrosstoolsTestParams[]> data() {
    return ImmutableList.of(
        new AndroidNdkCrosstoolsTestParams[] {
            new AndroidNdkCrosstoolsTestParams(
                new NdkMajorRevisionR10()
                    .apiLevel(NullEventHandler.INSTANCE, REPOSITORY_NAME, "21"),
                NdkRelease.create("r10e (64-bit)"),
                "ndkfiles.txt",
                "ndkdirectories.txt"
            )
        },
        new AndroidNdkCrosstoolsTestParams[] {
            new AndroidNdkCrosstoolsTestParams(
                new NdkMajorRevisionR12()
                    .apiLevel(NullEventHandler.INSTANCE, REPOSITORY_NAME, "21"),
                NdkRelease.create("Pkg.Desc = Android NDK\nPkg.Revision = 12.1.297705\n"),
                "ndk12bfiles.txt",
                "ndk12bdirectories.txt"
            )
        });
  }

  private final ImmutableSet<String> ndkFiles;
  private final ImmutableSet<String> ndkDirectories;
  private final ImmutableList<CrosstoolRelease> crosstoolReleases;
  private final ImmutableMap<String, String> stlFilegroups;

  public AndroidNdkCrosstoolsTest(AndroidNdkCrosstoolsTestParams params) throws IOException {
    // NDK test data is based on the x86 64-bit Linux Android NDK.
    NdkPaths ndkPaths =
        new NdkPaths(
            REPOSITORY_NAME, HOST_PLATFORM, params.apiLevel, params.ndkRelease.majorRevision);

    ImmutableList.Builder<CrosstoolRelease> crosstools = ImmutableList.builder();
    ImmutableMap.Builder<String, String> stlFilegroupsBuilder = ImmutableMap.builder();
    for (StlImpl ndkStlImpl : StlImpls.get(ndkPaths, params.getNdkMajorRevisionNumber())) {
      // Protos are immutable, so this can be shared between tests.
      CrosstoolRelease crosstool =
          params.getNdkMajorRevision().crosstoolRelease(ndkPaths, ndkStlImpl, HOST_PLATFORM);
      crosstools.add(crosstool);
      stlFilegroupsBuilder.putAll(ndkStlImpl.getFilegroupNamesAndFilegroupFileGlobPatterns());
    }

    crosstoolReleases = crosstools.build();
    stlFilegroups = stlFilegroupsBuilder.build();
    ndkFiles = params.getNdkFiles();
    ndkDirectories = params.getNdkDirectories();
  }

  @Test
  public void testPathsExist() throws Exception {

    for (CrosstoolRelease crosstool : crosstoolReleases) {
      for (CToolchain toolchain : crosstool.getToolchainList()) {

        // Test that all tool paths exist.
        for (ToolPath toolpath : toolchain.getToolPathList()) {
          // TODO(tmsriram): Not all crosstools contain llvm-profdata tool yet, remove
          // the check once llvm-profdata becomes always available.
          if (toolpath.getPath().contains("llvm-profdata")) {
            continue;
          }
          assertThat(ndkFiles).contains(toolpath.getPath());
        }

        // Test that all cxx_builtin_include_directory paths exist.
        for (String includeDirectory : toolchain.getCxxBuiltinIncludeDirectoryList()) {
          // Special case for builtin_sysroot.
          if (!includeDirectory.equals("%sysroot%/usr/include")) {
            String path = NdkPaths.stripRepositoryPrefix(includeDirectory);
            assertThat(ndkDirectories).contains(path);
          }
        }

        // Test that the builtin_sysroot path exists.
        {
          String builtinSysroot = NdkPaths.stripRepositoryPrefix(toolchain.getBuiltinSysroot());
          assertThat(ndkDirectories).contains(builtinSysroot);
        }

        // Test that all include directories added through unfiltered_cxx_flag exist.
        for (String flag : toolchain.getUnfilteredCxxFlagList()) {
          if (!flag.equals("-isystem")) {
            flag = NdkPaths.stripRepositoryPrefix(flag);
            assertThat(ndkDirectories).contains(flag);
          }
        }
      }
    }
  }

  // Regression test for b/36091573
  @Test
  public void testBuiltinIncludesDirectories() {
    for (CrosstoolRelease crosstool : crosstoolReleases) {
      for (CToolchain toolchain : crosstool.getToolchainList()) {
        // Each toolchain has at least one built-in include directory
        assertThat(toolchain.getCxxBuiltinIncludeDirectoryList()).isNotEmpty();

        for (String flag : toolchain.getUnfilteredCxxFlagList()) {
          // This list only contains "-isystem" and the values after "-isystem".
          if (!flag.equals("-isystem")) {
            // We should NOT be setting -isystem for the builtin includes directories. They are
            // already on the search list and adding the -isystem flag just changes their priority.
            assertThat(toolchain.getCxxBuiltinIncludeDirectoryList()).doesNotContain(flag);
          }
        }
      }
    }
  }

  @Test
  public void testStlFilegroupPathsExist() throws Exception {

    for (String fileglob : stlFilegroups.values()) {
      String fileglobNoWildcard = fileglob.substring(0, fileglob.lastIndexOf('/'));
      assertThat(ndkDirectories).contains(fileglobNoWildcard);
      assertThat(findFileByPattern(fileglob)).isTrue();
    }
  }

  private boolean findFileByPattern(String globPattern) {

    String start = globPattern.substring(0, globPattern.indexOf('*'));
    String end = globPattern.substring(globPattern.lastIndexOf('.'));
    for (String f : ndkFiles) {
      if (f.startsWith(start) && f.endsWith(end)) {
        return true;
      }
    }
    return false;
  }

  @Test
  public void testAllToolchainsHaveRuntimesFilegroup() {
    for (CrosstoolRelease crosstool : crosstoolReleases) {
      for (CToolchain toolchain : crosstool.getToolchainList()) {
        assertThat(toolchain.getDynamicRuntimesFilegroup()).isNotEmpty();
        assertThat(toolchain.getStaticRuntimesFilegroup()).isNotEmpty();
      }
    }
  }

  /**
   * Tests that each (cpu, compiler, glibc) triple in each crosstool is unique in that crosstool.
   */
  @Test
  public void testCrosstoolTriples() {

    StringBuilder errorBuilder = new StringBuilder();
    for (CrosstoolRelease crosstool : crosstoolReleases) {

      // Create a map of (cpu, compiler, glibc) triples -> toolchain.
      ImmutableMultimap.Builder<String, CToolchain> triples = ImmutableMultimap.builder();
      for (CToolchain toolchain : crosstool.getToolchainList()) {
        String triple = "(" + Joiner.on(", ").join(
            toolchain.getTargetCpu(),
            toolchain.getCompiler(),
            toolchain.getTargetLibc()) + ")";
        triples.put(triple, toolchain);
      }

      // Collect all the duplicate triples.
      for (Map.Entry<String, Collection<CToolchain>> entry : triples.build().asMap().entrySet()) {
        if (entry.getValue().size() > 1) {
          errorBuilder.append(entry.getKey() + ": " + Joiner.on(", ").join(
              Collections2.transform(entry.getValue(), new Function<CToolchain, String>() {
                @Override public String apply(CToolchain toolchain) {
                  return toolchain.getToolchainIdentifier();
                }
              })));
          errorBuilder.append("\n");
        }
      }
      errorBuilder.append("\n");
    }

    // This is a rather awkward condition to test on, but collecting all the duplicates first is
    // the only way to make a useful error message rather than finding the errors one by one.
    String error = errorBuilder.toString().trim();
    if (!error.isEmpty()) {
      fail("Toolchains contain duplicate (cpu, compiler, glibc) triples:\n" + error);
    }
  }
}
