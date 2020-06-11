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
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.InfoItem;

/** Info item for the {blaze,bazel}-bin directory. */
public final class BlazeBinInfoItem extends InfoItem {
  public BlazeBinInfoItem(String productName) {
    super(productName + "-bin", "Configuration dependent directory for binaries.", false);
  }

  // This is one of the three (non-hidden) info items that require a configuration, because the
  // corresponding paths contain the short name. Maybe we should recommend using the symlinks
  // or make them hidden by default?
  @Override
  public byte[] get(Supplier<BuildConfiguration> configurationSupplier, CommandEnvironment env) {
    checkNotNull(configurationSupplier);
    return print(configurationSupplier.get().getBinDirectory(RepositoryName.MAIN).getRoot());
  }
}
