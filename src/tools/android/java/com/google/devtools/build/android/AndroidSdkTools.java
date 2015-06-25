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
package com.google.devtools.build.android;

import com.google.common.base.Preconditions;

import com.android.builder.core.AndroidBuilder;
import com.android.builder.sdk.SdkInfo;
import com.android.builder.sdk.TargetInfo;
import com.android.sdklib.AndroidVersion;
import com.android.sdklib.BuildToolInfo;
import com.android.sdklib.IAndroidTarget;
import com.android.sdklib.repository.FullRevision;
import com.android.utils.StdLogger;

import java.io.File;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.nio.file.Path;

import javax.annotation.Nullable;

/**
 * Encapsulates the sdk related tools necessary for creating an AndroidBuilder.
 */
public class AndroidSdkTools {
  private final FullRevision apiVersion;

  private final Path aaptLocation;

  private final Path annotationJar;

  private final Path adbLocation;

  private final Path zipAlign;
  private final Path androidJar;

  private StdLogger stdLogger;

  public AndroidSdkTools(FullRevision apiVersion,
      Path aaptLocation,
      Path annotationJar,
      @Nullable Path adbLocation,
      @Nullable Path zipAlign,
      Path androidJar,
      StdLogger stdLogger) {
    this.stdLogger = stdLogger;
    this.apiVersion = Preconditions.checkNotNull(apiVersion, "apiVersion");
    this.aaptLocation = Preconditions.checkNotNull(aaptLocation, "aapt");
    this.annotationJar = Preconditions.checkNotNull(annotationJar, "annotationJar");
    this.adbLocation = adbLocation;
    this.zipAlign = zipAlign;
    this.androidJar = Preconditions.checkNotNull(androidJar, "androidJar");
  }

  /** Creates an AndroidBuilder from the provided sdk tools. */
  public AndroidBuilder createAndroidBuilder() {
    // BuildInfoTool contains the paths to all tools that the AndroidBuilder uses.
    BuildToolInfo buildToolInfo =
        new BuildToolInfoBuilder(apiVersion).setZipAlign(zipAlign).setAapt(aaptLocation).build();

    BazelPlatformTarget bazelPlatformTarget = new BazelPlatformTarget(androidJar,
        new AndroidVersion(apiVersion.getMajor(), ""), buildToolInfo);

    AndroidBuilder builder = new AndroidBuilder(
        "bazel",  /* project id */
        "bazel",  /* created by */
        stdLogger,
        false /* verbose */);
    TargetInfo targetInfo = createTargetInfo(buildToolInfo, bazelPlatformTarget);
    SdkInfo sdkInfo = createSdkInfo(annotationJar, adbLocation);

    // TargetInfo and sdk info provide links to all the tools.
    builder.setTargetInfo(sdkInfo, targetInfo);
    return builder;
  }

  private static SdkInfo createSdkInfo(Path annotationJar, Path adbLocation) {
    try {
      // necessary hack because SdkInfo doesn't declare a public constructor.
      Constructor<SdkInfo> sdkInfoConstructor =
          SdkInfo.class.getDeclaredConstructor(File.class, File.class);
      sdkInfoConstructor.setAccessible(true);
      return sdkInfoConstructor.newInstance(maybeToFile(annotationJar), maybeToFile(adbLocation));
    } catch (NoSuchMethodException
        | SecurityException
        | InstantiationException
        | IllegalAccessException
        | IllegalArgumentException
        | InvocationTargetException e) {
      throw new AssertionError(e);
    }
  }

  private static TargetInfo createTargetInfo(BuildToolInfo buildToolInfo,
      BazelPlatformTarget bazelPlatformTarget) {
    try {
      // necessary hack because TargetInfo doesn't declare a public constructor.
      Constructor<TargetInfo> targetInfoConstructor =
          TargetInfo.class.getDeclaredConstructor(IAndroidTarget.class, BuildToolInfo.class);
      targetInfoConstructor.setAccessible(true);
      return targetInfoConstructor.newInstance(bazelPlatformTarget, buildToolInfo);
    } catch (NoSuchMethodException
        | SecurityException
        | InstantiationException
        | IllegalAccessException
        | IllegalArgumentException
        | InvocationTargetException e) {
      throw new AssertionError(e);
    }
  }
  
  private static File maybeToFile(Path path) {
    if (path == null) {
      return null;
    }
    return path.toFile();
  }
}