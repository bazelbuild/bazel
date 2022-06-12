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
package com.google.devtools.build.lib.rules.cpp;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CToolchain;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.CToolchain.Feature;
import com.google.protobuf.TextFormat;

class CcToolchainTestHelper {

  private CcToolchainTestHelper() {}

  /** Creates a CcToolchainFeatures from features described in the given toolchain fragment. */
  public static CcToolchainFeatures buildFeatures(String... toolchain) throws Exception {
    CToolchain.Builder toolchainBuilder = CToolchain.newBuilder();
    TextFormat.merge(Joiner.on("").join(toolchain), toolchainBuilder);
    return new CcToolchainFeatures(
        CcToolchainConfigInfo.fromToolchainForTestingOnly(toolchainBuilder.buildPartial()),
        PathFragment.create("crosstool/"));
  }

  /** Creates a CcToolchainFeatures from given features and action configs. */
  public static CcToolchainFeatures buildFeatures(
      ImmutableList<Feature> features, ImmutableList<CToolchain.ActionConfig> actionConfigs)
      throws Exception {
    return new CcToolchainFeatures(
        CcToolchainConfigInfo.fromToolchainForTestingOnly(
            CToolchain.newBuilder()
                .addAllFeature(features)
                .addAllActionConfig(actionConfigs)
                .buildPartial()),
        PathFragment.create("crosstool/"));
  }
}
