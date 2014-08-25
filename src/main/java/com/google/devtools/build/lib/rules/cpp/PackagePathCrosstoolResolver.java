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

package com.google.devtools.build.lib.rules.cpp;

import com.google.common.base.Function;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.syntax.Label.SyntaxException;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.view.RedirectChaser;
import com.google.devtools.build.lib.view.config.BuildOptions;
import com.google.devtools.build.lib.view.config.ConfigurationEnvironment;
import com.google.devtools.build.lib.view.config.InvalidConfigurationException;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig;

/**
 * A provider to load crosstool configuration from the package path.
 */
public final class PackagePathCrosstoolResolver
    implements CrosstoolConfigurationLoader.CrosstoolResolver {

  private static final String CROSSTOOL_CONFIGURATION_FILENAME = "CROSSTOOL";

  private final Function<String, String> cpuTransformer;

  public PackagePathCrosstoolResolver(Function<String, String> cpuTransformer) {
    this.cpuTransformer = cpuTransformer;
  }

  @Override
  public String resolveCrosstoolTop(ConfigurationEnvironment env, String crosstoolTop)
      throws InvalidConfigurationException {
    if (!crosstoolTop.startsWith("//")) {
      return crosstoolTop;
    }
    try {
      Label label = Label.parseAbsolute(crosstoolTop);
      Label resolvedLabel = RedirectChaser.followRedirects(env, label, "crosstool_top");
      return resolvedLabel.toString();
    } catch (SyntaxException e) {
      throw new InvalidConfigurationException(e);
    }
  }

  @Override
  public Path findConfiguration(ConfigurationEnvironment env, FileSystem fileSystem,
      String crosstoolTop) throws InvalidConfigurationException {
    // If the crosstool top is not a label, it must be an absolute path.
    if (!crosstoolTop.startsWith("//") && crosstoolTop.startsWith("/")) {
      // Not null only in tests
      if (fileSystem != null) {
        return fileSystem.getPath(crosstoolTop).getRelative(CROSSTOOL_CONFIGURATION_FILENAME);
      }
      return null;
    }
    try {
      Label label = Label.parseAbsolute(crosstoolTop);
      Package containingPackage = env.getTarget(label.getLocalTargetLabel("BUILD"))
          .getPackage();
      return env.getPath(containingPackage, CROSSTOOL_CONFIGURATION_FILENAME);
    } catch (SyntaxException e) {
      throw new InvalidConfigurationException(e);
    } catch (NoSuchThingException e) {
      return null;
    }
  }

  @Override
  public CrosstoolConfig.CToolchain selectToolchain(
      CrosstoolConfig.CrosstoolRelease release, BuildOptions options)
      throws InvalidConfigurationException {
    return CrosstoolConfigurationLoader.selectToolchain(release, options, cpuTransformer);
  }
}
