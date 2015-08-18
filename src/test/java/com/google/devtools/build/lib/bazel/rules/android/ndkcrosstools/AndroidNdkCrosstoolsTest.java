// Copyright 2015 Google Inc. All rights reserved.
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

import com.google.devtools.build.lib.bazel.rules.android.ndkcrosstools.AndroidNdkCrosstools.NdkCrosstoolsException;
import com.google.devtools.build.lib.events.NullEventHandler;
import com.google.devtools.build.lib.util.ResourceFileLoader;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CToolchain;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CrosstoolRelease;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.DefaultCpuToolchain;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.ToolPath;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.io.IOException;
import java.util.HashSet;
import java.util.Scanner;
import java.util.Set;

/**
 * Tests for {@link AndroidNdkCrosstools}.
 */
@RunWith(JUnit4.class)
public class AndroidNdkCrosstoolsTest {

  private static final String API_LEVEL = "21";
  private static final String REPOSITORY_NAME = "testrepository";
  private static final NdkRelease NDK_RELEASE = NdkRelease.create("r10e (64-bit)");
  private static final CrosstoolRelease CROSSTOOL_RELEASE;

  static {
    try {
      // Protos are immutable, so this can be shared between tests.
      CROSSTOOL_RELEASE = AndroidNdkCrosstools.createCrosstoolRelease(
          NullEventHandler.INSTANCE, REPOSITORY_NAME, API_LEVEL, NDK_RELEASE);

    } catch (NdkCrosstoolsException e) {
      throw new RuntimeException(e);
    }
  }

  @Test
  public void testPathsExist() throws Exception {

    // ndkfiles.txt contains a list of every file in the ndk, created using this command at the
    // root of the Android NDK for version r10e (64-bit):
    //     find . -xtype f | sed 's|^\./||' | sort
    // and similarly for ndkdirectories, except "-xtype d" is used.
    //
    // It's unfortunate to have files like these, since they're large and brittle, but since the
    // whole NDK can't be checked in to test against, it's about the most that can be done right
    // now.
    Set<String> ndkFiles = getFiles("ndkfiles.txt");
    Set<String> ndkDirectories = getFiles("ndkdirectories.txt");

    for (CToolchain toolchain : CROSSTOOL_RELEASE.getToolchainList()) {

      // Test that all tool paths exist.
      for (ToolPath toolpath : toolchain.getToolPathList()) {
        String path = NdkPaths.stripRepositoryPrefix(toolpath.getPath());
        assertThat(ndkFiles).contains(path);
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

  private static Set<String> getFiles(String fileName) {
    String ndkFilesContent;
    try {
      ndkFilesContent = ResourceFileLoader.loadResource(
          AndroidNdkCrosstoolsTest.class, fileName);
    } catch (IOException e) {
      throw new IllegalStateException(e);
    }

    Set<String> ndkFiles = new HashSet<>();
    Scanner ndkFilesContentScanner = new Scanner(ndkFilesContent);
    while (ndkFilesContentScanner.hasNext()) {
      String path = ndkFilesContentScanner.nextLine();
      // The contents of the NDK are placed at "external/%repositoryName%/ndk".
      // The "external/%repositoryName%" part is removed using NdkPaths.stripRepositoryPrefix,
      // but to make it easier the "ndk/" part is added here.
      path = "ndk/" + path;
      ndkFiles.add(path);
    }
    ndkFilesContentScanner.close();
    return ndkFiles;
  }

  @Test
  public void testDefaultToolchainsExist() {

    Set<String> toolchainNames = new HashSet<>();
    for (CToolchain toolchain : CROSSTOOL_RELEASE.getToolchainList()) {
      toolchainNames.add(toolchain.getToolchainIdentifier());
    }

    for (DefaultCpuToolchain defaultCpuToolchain : CROSSTOOL_RELEASE.getDefaultToolchainList()) {
      assertThat(toolchainNames).contains(defaultCpuToolchain.getToolchainIdentifier());
    }
  }
  
}
