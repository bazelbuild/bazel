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

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
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
import java.io.IOException;
import java.io.InputStream;
import javax.annotation.Nullable;

/**
 * A {@link SkyFunction} that does things for FDO that a regular configured target is not allowed
 * to.
 *
 * <p>This only exists because the value of {@code --fdo_optimize} can be a workspace-relative path
 * and thus we need to depend on {@link BlazeDirectories} somehow, which neither the configuration
 * nor the analysis phase can "officially" do.
 *
 * <p>The fix is probably to make it possible for {@link
 * com.google.devtools.build.lib.analysis.actions.SymlinkAction} to create workspace-relative
 * symlinks because action execution can hopefully depend on {@link BlazeDirectories}.
 *
 * <p>There is also the awful and incorrect {@link Path#exists()} call in {@link
 * com.google.devtools.build.lib.view.proto.CcProtoProfileProvider#getProfile(
 * com.google.devtools.build.lib.analysis.RuleContext)} which needs a {@link Path}.
 */
public class CcSkyframeSupportFunction implements SkyFunction {

  private final BlazeDirectories directories;

  public CcSkyframeSupportFunction(BlazeDirectories directories) {
    this.directories = Preconditions.checkNotNull(directories);
  }

  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws InterruptedException, CcSkyframeSupportException {
    CcSkyframeSupportValue.Key key = (CcSkyframeSupportValue.Key) skyKey.argument();
    Path fdoZipPath = null;
    if (key.getFdoZipPath() != null) {
      fdoZipPath = directories.getWorkspace().getRelative(key.getFdoZipPath());
    }

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
            packageIdentifier
                .getPackageFragment()
                .getRelative(CrosstoolConfigurationLoader.CROSSTOOL_CONFIGURATION_FILENAME);
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
          throw new CcSkyframeSupportException(
              String.format(
                  "there is no CROSSTOOL file at %s, which is needed for this cc_toolchain",
                  crosstool.toString()),
              key);
        }
        try (InputStream inputStream = crosstoolFile.getInputStream()) {
          String crosstoolContent = new String(FileSystemUtils.readContentAsLatin1(inputStream));
          crosstoolRelease =
              CrosstoolConfigurationLoader.toReleaseConfiguration(
                  "CROSSTOOL file " + crosstool, () -> crosstoolContent, /* digestOrNull= */ null);
        }
      } catch (IOException | InvalidConfigurationException e) {
        throw new CcSkyframeSupportException(e, key);
      }
    }

    return new CcSkyframeSupportValue(fdoZipPath, crosstoolRelease);
  }

  @Nullable
  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }

  /** Exception encapsulating IOExceptions thrown in {@link CcSkyframeSupportFunction} */
  public static class CcSkyframeSupportException extends SkyFunctionException {

    public CcSkyframeSupportException(Exception cause, CcSkyframeSupportValue.Key key) {
      super(cause, key);
    }

    public CcSkyframeSupportException(String message, CcSkyframeSupportValue.Key key) {
      super(new CcCrosstoolException(message), key);
    }
  }
}
