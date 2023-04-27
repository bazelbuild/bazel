// Copyright 2018 The Bazel Authors. All rights reserved.
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


import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.rules.cpp.CcToolchain.AdditionalBuildVariablesComputer;
import com.google.devtools.build.lib.vfs.PathFragment;

/** Helper responsible for creating CcToolchainProvider */
public final class CcToolchainProviderHelper {

  /**
   * Returns {@link CcToolchainVariables} instance with build variables that only depend on the
   * toolchain.
   */
  static CcToolchainVariables getBuildVariables(
      BuildOptions buildOptions,
      CppConfiguration cppConfiguration,
      PathFragment sysroot,
      AdditionalBuildVariablesComputer additionalBuildVariablesComputer) {
    CcToolchainVariables.Builder variables = CcToolchainVariables.builder();

    String minOsVersion = cppConfiguration.getMinimumOsVersion();
    if (minOsVersion != null) {
      variables.addStringVariable(CcCommon.MINIMUM_OS_VERSION_VARIABLE_NAME, minOsVersion);
    }

    if (sysroot != null) {
      variables.addStringVariable(CcCommon.SYSROOT_VARIABLE_NAME, sysroot.getPathString());
    }

    variables.addAllNonTransitive(additionalBuildVariablesComputer.apply(buildOptions));

    return variables.build();
  }
}
