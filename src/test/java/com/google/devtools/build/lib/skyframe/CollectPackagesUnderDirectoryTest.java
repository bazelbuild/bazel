// Copyright 2019 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.bazel.bzlmod.ModuleFileFunction;
import com.google.devtools.build.lib.packages.BuildFileName;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionName;
import java.util.List;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test for {@link CollectPackagesUnderDirectoryFunction}. */
@RunWith(JUnit4.class)
public final class CollectPackagesUnderDirectoryTest
    extends AbstractCollectPackagesUnderDirectoryTest {
  @Override
  protected String getWorkspacePathString() {
    return "/workspace";
  }

  @Override
  protected List<BuildFileName> getBuildFileNamesByPriority() {
    return BazelSkyframeExecutorConstants.BUILD_FILES_BY_PRIORITY;
  }

  @Override
  protected ImmutableMap<SkyFunctionName, SkyFunction> getExtraSkyFunctions() {
    return ImmutableMap.of(
        SkyFunctions.MODULE_FILE,
        new ModuleFileFunction(
            ruleClassProvider.getBazelStarlarkEnvironment(),
            directories.getWorkspace(),
            ImmutableMap.of()));
  }

  @Override
  protected SkyframeExecutorFactory makeSkyframeExecutorFactory() {
    return new SequencedSkyframeExecutorFactory();
  }
}
