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

package com.google.devtools.build.lib.rules.objc;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;
import com.google.devtools.build.lib.rules.apple.ApplePlatform;
import com.google.devtools.build.lib.util.FileType;
import com.google.devtools.build.lib.util.FileTypeSet;

/** Shared rule classes and associated utility code for Objective-C rules. */
public class ObjcRuleClasses {
  static final String STRIP = "strip";

  private ObjcRuleClasses() {
    throw new UnsupportedOperationException("static-only");
  }

  /** A string constant for feature to link a bundle. */
  static final String LINK_BUNDLE_FEATURE = "link_bundle";

  /** A string constant for feature to link a dylib. */
  static final String LINK_DYLIB_FEATURE = "link_dylib";

  /** Attribute name for a dummy target in a child configuration. */
  static final String CHILD_CONFIG_ATTR = "$child_configuration_dummy";

  /**
   * Creates a new spawn action builder with apple environment variables set that are typically
   * needed by the apple toolchain. This should be used to start to build spawn actions that, in
   * order to run, require both a darwin architecture and a collection of environment variables
   * which contain information about the target and exec architectures.
   */
  static SpawnAction.Builder spawnAppleEnvActionBuilder(
      XcodeConfigInfo xcodeConfigInfo, ApplePlatform targetPlatform) {
    return spawnOnDarwinActionBuilder(xcodeConfigInfo)
        .setEnvironment(appleToolchainEnvironment(xcodeConfigInfo, targetPlatform));
  }

  /** Returns apple environment variables that are typically needed by the apple toolchain. */
  private static ImmutableMap<String, String> appleToolchainEnvironment(
      XcodeConfigInfo xcodeConfigInfo, ApplePlatform targetPlatform) {
    return ImmutableMap.<String, String>builder()
        .putAll(
            AppleConfiguration.appleTargetPlatformEnv(
                targetPlatform, xcodeConfigInfo.getSdkVersionForPlatform(targetPlatform)))
        .putAll(AppleConfiguration.getXcodeVersionEnv(xcodeConfigInfo.getXcodeVersion()))
        .buildOrThrow();
  }

  /** Creates a new spawn action builder that requires a darwin architecture to run. */
  private static SpawnAction.Builder spawnOnDarwinActionBuilder(XcodeConfigInfo xcodeConfigInfo) {
    return new SpawnAction.Builder().setExecutionInfo(xcodeConfigInfo.getExecutionRequirements());
  }



  /** Iff a file matches this type, it is considered to use C++. */
  static final FileType CPP_SOURCES = FileType.of(".cc", ".cpp", ".mm", ".cxx", ".C");

  static final FileType NON_CPP_SOURCES = FileType.of(".m", ".c");

  static final FileType ASSEMBLY_SOURCES = FileType.of(".s", ".S", ".asm");

  static final FileType OBJECT_FILE_SOURCES = FileType.of(".o");

  /** Files that should actually be compiled. */
  static final FileTypeSet COMPILABLE_SRCS_TYPE =
      FileTypeSet.of(NON_CPP_SOURCES, CPP_SOURCES, ASSEMBLY_SOURCES);

  /**
   * Files that are already compiled.
   */
  static final FileTypeSet PRECOMPILED_SRCS_TYPE = FileTypeSet.of(OBJECT_FILE_SOURCES);
}
