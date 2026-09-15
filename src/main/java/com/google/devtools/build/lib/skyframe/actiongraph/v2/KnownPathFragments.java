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
package com.google.devtools.build.lib.skyframe.actiongraph.v2;

import com.google.devtools.build.lib.analysis.AnalysisProtosV2;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;

/** Cache for {@link PathFragment} in the action graph. */
public class KnownPathFragments extends BaseCache<PathFragment, AnalysisProtosV2.PathFragment> {
  KnownPathFragments(AqueryOutputHandler aqueryOutputHandler) {
    super(aqueryOutputHandler);
  }

  @Override
  AnalysisProtosV2.PathFragment createProto(PathFragment pathFragment, int id)
      throws IOException, InterruptedException {
    AnalysisProtosV2.PathFragment.Builder pathFragmentProtoBuilder =
        AnalysisProtosV2.PathFragment.newBuilder().setId(id).setLabel(pathFragment.getBaseName());

    // Recursively create the ancestor path fragments.
    // If pathFragment has no parent, leave parentId blank and avoid calling dataToId
    // to prevent the cache from being polluted with a null entry.
    if (hasParent(pathFragment)) {
      pathFragmentProtoBuilder.setParentId(
          dataToIdAndStreamOutputProto(pathFragment.getParentDirectory()));
    }

    return pathFragmentProtoBuilder.build();
  }

  @Override
  void toOutput(AnalysisProtosV2.PathFragment pathFragmentProto) throws IOException {
    aqueryOutputHandler.outputPathFragment(pathFragmentProto);
  }

  private static boolean hasParent(PathFragment pathFragment) {
    return pathFragment.getParentDirectory() != null
        && !pathFragment.getParentDirectory().getBaseName().isEmpty();
  }
}
