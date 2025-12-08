// Copyright 2020 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.runtime.commands.info;

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.base.Supplier;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.buildtool.BuildRequestOptions;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.InfoItem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.common.options.OptionsParsingResult;
import javax.annotation.Nullable;

/** Info item for the {blaze,bazel}-genfiles directory. */
public final class BlazeGenfilesInfoItem extends InfoItem {
  private final String productName;
  @Nullable private final OptionsParsingResult commandOptions;

  public BlazeGenfilesInfoItem(String productName, @Nullable OptionsParsingResult commandOptions) {
    super(
        productName + "-genfiles", "Configuration dependent directory for generated files.", false);
    this.productName = productName;
    this.commandOptions = commandOptions;
  }

  // Returns the convenience symlink path (e.g., <workspace>/bazel-genfiles) instead of the
  // configuration-specific path to avoid triggering BuildConfigurationValue creation, which
  // can invalidate the analysis cache. The symlink is created during the first build and points
  // to the actual genfiles directory. This is also the path users typically reference in scripts.
  //
  // The path respects the --symlink_prefix option. If --symlink_prefix=custom-, returns
  // <workspace>/custom-genfiles. If --symlink_prefix is an absolute path like /tmp/out/, returns
  // /tmp/out/genfiles.
  @Override
  public byte[] get(
      Supplier<BuildConfigurationValue> configurationSupplier, CommandEnvironment env) {
    checkNotNull(env);
    return print(getSymlinkPath(env, "genfiles"));
  }

  private Path getSymlinkPath(CommandEnvironment env, String suffix) {
    String symlinkPrefix = getSymlinkPrefix();
    if (symlinkPrefix.startsWith("/")) {
      // Absolute path prefix (e.g., --symlink_prefix=/tmp/out/)
      // Note: --symlink_prefix=/ disables symlinks entirely, but we still return a path
      return env.getRuntime().getFileSystem().getPath(symlinkPrefix + suffix);
    } else {
      // Relative prefix (e.g., --symlink_prefix=custom- or default bazel-)
      return env.getWorkspace().getRelative(symlinkPrefix + suffix);
    }
  }

  private String getSymlinkPrefix() {
    if (commandOptions == null) {
      return productName + "-";
    }
    BuildRequestOptions buildRequestOptions = commandOptions.getOptions(BuildRequestOptions.class);
    if (buildRequestOptions == null) {
      return productName + "-";
    }
    return buildRequestOptions.getSymlinkPrefix(productName);
  }
}
