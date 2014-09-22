// Copyright 2014 Google Inc. All rights reserved.
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

import static com.google.devtools.build.lib.rules.objc.ObjcProvider.FRAMEWORK_DIR;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.xcode.common.Platform;
import com.google.devtools.build.xcode.common.TargetDeviceFamily;
import com.google.devtools.build.xcode.util.Interspersing;

import java.util.List;

/**
 * Utility code for use when generating iOS SDK commands.
 */
public class IosSdkCommands {
  // TODO(bazel-team): Make these settings parameterizeable.
  public static final String MINIMUM_OS_VERSION = "7.0";
  public static final ImmutableList<TargetDeviceFamily> TARGET_DEVICE_FAMILIES =
      ImmutableList.of(TargetDeviceFamily.IPHONE);

  public static final String DEVELOPER_DIR = "/Applications/Xcode.app/Contents/Developer";
  public static final String BIN_DIR =
      DEVELOPER_DIR + "/Toolchains/XcodeDefault.xctoolchain/usr/bin";
  public static final String ACTOOL_PATH = DEVELOPER_DIR + "/usr/bin/actool";

  private IosSdkCommands() {
    throw new UnsupportedOperationException("static-only");
  }

  private static String platformDir(ObjcConfiguration configuration) {
    return DEVELOPER_DIR + "/Platforms/" + configuration.getPlatform().getNameInPlist()
        + ".platform";
  }

  public static String sdkDir(ObjcConfiguration configuration) {
    return platformDir(configuration) + "/Developer/SDKs/"
        + configuration.getPlatform().getNameInPlist() + configuration.getIosSdkVersion() + ".sdk";
  }

  private static Iterable<PathFragment> uniqueParentDirectories(Iterable<PathFragment> paths) {
    ImmutableSet.Builder<PathFragment> parents = new ImmutableSet.Builder<>();
    for (PathFragment path : paths) {
      parents.add(path.getParentDirectory());
    }
    return parents.build();
  }

  public static List<String> commonLinkAndCompileArgsForClang(
      ObjcProvider provider, ObjcConfiguration configuration) {
    ImmutableList.Builder<String> builder = new ImmutableList.Builder<>();
    if (configuration.getPlatform() == Platform.SIMULATOR) {
      builder.add("-mios-simulator-version-min=" + MINIMUM_OS_VERSION);
    }
    return builder
        .add("-arch", configuration.getIosCpu())
        .add("-isysroot", sdkDir(configuration))
        // TODO(bazel-team): Pass framework search paths to Xcodegen.
        .add("-F", sdkDir(configuration) + "/Developer/Library/Frameworks")
        // Add custom (non-SDK) framework search paths. For each framework foo/bar.framework,
        // include "foo" as a search path.
        .addAll(Interspersing.beforeEach(
            "-F",
            PathFragment.safePathStrings(uniqueParentDirectories(provider.get(FRAMEWORK_DIR)))))
        .build();
  }

  public static String momcPath(ObjcConfiguration configuration) {
    return platformDir(configuration) + "/Developer/usr/bin/momc";
  }

  public static List<String> compileArgsForClang(ObjcConfiguration configuration) {
    switch (configuration.getPlatform()) {
      case DEVICE:
        return ImmutableList.of();
      case SIMULATOR:
        // These are added by Xcode when building, because the simulator is built on OSX
        // frameworks so we aim compile to match the OSX objc runtime.
        return ImmutableList.of(
          "-fexceptions",
          "-fasm-blocks",
          "-fobjc-abi-version=2",
          "-fobjc-legacy-dispatch");
      default:
        throw new IllegalStateException("Unknown configuration type: "
            + configuration.getPlatform());
    }
  }
}
