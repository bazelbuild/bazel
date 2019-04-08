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
package com.google.devtools.build.lib.rules.cpp;

import com.google.common.annotations.VisibleForTesting;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.skyframe.PackageLookupValue;
import com.google.devtools.build.lib.skyframe.SkyframeBuildView.CcCrosstoolException;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CrosstoolRelease;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.protobuf.TextFormat;
import com.google.protobuf.TextFormat.ParseException;
import com.google.protobuf.UninitializedMessageException;
import java.io.IOException;
import java.io.InputStream;
import javax.annotation.Nullable;

/**
 * A {@link SkyFunction} that does things for CROSSTOOL that a regular configured target is not
 * allowed to (read a file from filesystem).
 */
public class CcSkyframeCrosstoolSupportFunction implements SkyFunction {

  static final String CROSSTOOL_CONFIGURATION_FILENAME = "CROSSTOOL";

  public CcSkyframeCrosstoolSupportFunction() {}

  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws InterruptedException, CcSkyframeCrosstoolSupportException {
    CcSkyframeCrosstoolSupportValue.Key key =
        (CcSkyframeCrosstoolSupportValue.Key) skyKey.argument();

    CrosstoolRelease crosstoolRelease = null;
    if (key.getPackageWithCrosstoolInIt() != null) {
      try {
        // 1. Lookup the package to handle multiple package roots (PackageLookupValue)
        PackageIdentifier packageIdentifier = key.getPackageWithCrosstoolInIt();
        PackageLookupValue crosstoolPackageValue =
            (PackageLookupValue) env.getValue(PackageLookupValue.key(packageIdentifier));
        if (env.valuesMissing()) {
          return null;
        }

        // 2. Get crosstool file (FileValue)
        PathFragment crosstool =
            packageIdentifier.getPackageFragment().getRelative(CROSSTOOL_CONFIGURATION_FILENAME);
        FileValue crosstoolFileValue =
            (FileValue)
                env.getValue(
                    FileValue.key(
                        RootedPath.toRootedPath(crosstoolPackageValue.getRoot(), crosstool)));
        if (env.valuesMissing()) {
          return null;
        }

        // 3. Parse the crosstool file the into CrosstoolRelease
        Path crosstoolFile = crosstoolFileValue.realRootedPath().asPath();
        if (!crosstoolFile.exists()) {
          throw new CcSkyframeCrosstoolSupportException(
              String.format(
                  "there is no CROSSTOOL file at %s, which is needed for this cc_toolchain",
                  crosstool.toString()),
              key);
        }
        try (InputStream inputStream = crosstoolFile.getInputStream()) {
          String crosstoolContent = new String(FileSystemUtils.readContentAsLatin1(inputStream));
          crosstoolRelease = toReleaseConfiguration(crosstoolContent);
        }
      } catch (IOException | InvalidConfigurationException e) {
        throw new CcSkyframeCrosstoolSupportException(e.getMessage(), key);
      }
    }

    return new CcSkyframeCrosstoolSupportValue(crosstoolRelease);
  }

  @Nullable
  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }

  /**
   * Reads the given <code>crosstoolContent</code>, which must be in ascii format, into a protocol
   * buffer.
   *
   * @param crosstoolContent for the error messages
   */
  @VisibleForTesting
  static CrosstoolRelease toReleaseConfiguration(String crosstoolContent)
      throws InvalidConfigurationException {
    CrosstoolRelease.Builder builder = CrosstoolRelease.newBuilder();
    try {
      TextFormat.merge(crosstoolContent, builder);
      return builder.build();
    } catch (ParseException e) {
      throw new InvalidConfigurationException(
          "Could not read the CROSSTOOL file because of a parser error (" + e.getMessage() + ")");
    } catch (UninitializedMessageException e) {
      throw new InvalidConfigurationException(
          "Could not read the CROSSTOOL file because of an incomplete protocol buffer ("
              + e.getMessage()
              + ")");
    }
  }

  /** Exception encapsulating IOExceptions thrown in {@link CcSkyframeCrosstoolSupportFunction} */
  public static class CcSkyframeCrosstoolSupportException extends SkyFunctionException {

    public CcSkyframeCrosstoolSupportException(
        String message, CcSkyframeCrosstoolSupportValue.Key key) {
      super(new CcCrosstoolException(message), key);
    }
  }
}
