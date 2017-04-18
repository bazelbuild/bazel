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

package com.google.devtools.build.lib.rules.cpp.transitions;

import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.PatchTransition;
import com.google.devtools.build.lib.rules.cpp.CppOptions;
import com.google.devtools.build.lib.rules.cpp.FdoSupport;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig;

/**
 * Dynamic transition that turns off LIPO/FDO settings.
 *
 * <p>This is suitable, for example, when visiting data dependencies of a C++ rule built with LIPO.
 */
public final class LipoDataTransition implements PatchTransition {
  public static final LipoDataTransition INSTANCE = new LipoDataTransition();

  private LipoDataTransition() {}

  @Override
  public BuildOptions apply(BuildOptions options) {
    if (options.get(BuildConfiguration.Options.class).isHost) {
      return options;
    }

    // If this target and its transitive closure don't have C++ options, there's no
    // LIPO context to change.
    if (!options.contains(CppOptions.class)) {
      return options;
    }

    CppOptions cppOptions = options.get(CppOptions.class);
    if (cppOptions.getLipoMode() == CrosstoolConfig.LipoMode.OFF) {
      return options;
    }

    options = options.clone();
    cppOptions = options.get(CppOptions.class);

    // Once autoFdoLipoData is on, it stays on (through all future transitions).
    if (!cppOptions.getAutoFdoLipoData() && cppOptions.getFdoOptimize() != null) {
      cppOptions.autoFdoLipoDataForBuild = FdoSupport.isAutoFdo(cppOptions.getFdoOptimize());
    }
    cppOptions.lipoModeForBuild = CrosstoolConfig.LipoMode.OFF;
    cppOptions.fdoInstrumentForBuild = null;
    cppOptions.fdoOptimizeForBuild = null;
    return options;
  }

  @Override
  public boolean defaultsToSelf() {
    return false;
  }
}
