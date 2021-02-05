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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.analysis.buildinfo.BuildInfoCollection;
import com.google.devtools.build.lib.analysis.buildinfo.BuildInfoFactory;
import com.google.devtools.build.lib.analysis.buildinfo.BuildInfoKey;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayList;
import java.util.List;

/**
 * C++ build info creation - generates header files that contain the corresponding build-info data.
 */
public final class CppBuildInfo implements BuildInfoFactory {
  public static final BuildInfoKey KEY = new BuildInfoKey("C++");

  private static final PathFragment BUILD_INFO_NONVOLATILE_HEADER_NAME =
      PathFragment.create("build-info-nonvolatile.h");
  private static final PathFragment BUILD_INFO_VOLATILE_HEADER_NAME =
      PathFragment.create("build-info-volatile.h");
  // TODO(bazel-team): (2011) Get rid of the redacted build info. We should try to make
  // the linkstamping process handle the case where those values are undefined.
  private static final PathFragment BUILD_INFO_REDACTED_HEADER_NAME =
      PathFragment.create("build-info-redacted.h");

  @Override
  public BuildInfoCollection create(
      BuildInfoContext buildInfoContext,
      BuildConfiguration config,
      Artifact buildInfo,
      Artifact buildChangelist) {
    List<Action> actions = new ArrayList<>();
    WriteBuildInfoHeaderAction redactedInfo =
        getHeader(
            buildInfoContext,
            config,
            BUILD_INFO_REDACTED_HEADER_NAME,
            NestedSetBuilder.emptySet(Order.STABLE_ORDER),
            true,
            true);
    WriteBuildInfoHeaderAction nonvolatileInfo =
        getHeader(
            buildInfoContext,
            config,
            BUILD_INFO_NONVOLATILE_HEADER_NAME,
            NestedSetBuilder.create(Order.STABLE_ORDER, buildInfo),
            false,
            true);
    WriteBuildInfoHeaderAction volatileInfo =
        getHeader(
            buildInfoContext,
            config,
            BUILD_INFO_VOLATILE_HEADER_NAME,
            NestedSetBuilder.create(Order.STABLE_ORDER, buildChangelist),
            true,
            false);
    actions.add(redactedInfo);
    actions.add(nonvolatileInfo);
    actions.add(volatileInfo);
    return new BuildInfoCollection(
        actions,
        ImmutableList.of(nonvolatileInfo.getPrimaryOutput(), volatileInfo.getPrimaryOutput()),
        ImmutableList.of(redactedInfo.getPrimaryOutput()));
  }

  private WriteBuildInfoHeaderAction getHeader(
      BuildInfoContext buildInfoContext,
      BuildConfiguration config,
      PathFragment headerName,
      NestedSet<Artifact> inputs,
      boolean writeVolatileInfo,
      boolean writeNonVolatileInfo) {
    ArtifactRoot outputPath = config.getIncludeDirectory(RepositoryName.MAIN);
    final Artifact header =
        buildInfoContext.getBuildInfoArtifact(
            headerName,
            outputPath,
            writeVolatileInfo && !inputs.isEmpty()
                ? BuildInfoType.NO_REBUILD
                : BuildInfoType.FORCE_REBUILD_IF_CHANGED);
    return new WriteBuildInfoHeaderAction(
        inputs, header, writeVolatileInfo, writeNonVolatileInfo);
  }

  @Override
  public BuildInfoKey getKey() {
    return KEY;
  }

  @Override
  public boolean isEnabled(BuildConfiguration config) {
    return config.hasFragment(CppConfiguration.class);
  }
}
