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
package com.google.devtools.build.lib.skyframe;

import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.pkgcache.LoadedPackageProvider;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.skyframe.CycleInfo;
import com.google.devtools.build.skyframe.NodeKey;

/**
 * Reports cycles between {@link ConfiguredTargetNode}s. As with
 * {@link TransitiveTargetCycleReporter}, these indicate cycles between targets. But in the current
 * target-parsing, loading, analysis, and execution phase distinction, such cycles would have been
 * caught during the loading phase. While we don't expect any of these cycles to occur in practice,
 * they do occur in unit tests of {@link BuildView}.
 */
class ConfiguredTargetCycleReporter extends AbstractLabelCycleReporter {

  private static final Predicate<NodeKey> IS_CONFIGURED_TARGET_NODE_KEY =
      NodeTypes.hasNodeType(NodeTypes.CONFIGURED_TARGET);

  ConfiguredTargetCycleReporter(LoadedPackageProvider loadedPackageProvider) {
    super(loadedPackageProvider);
  }

  @Override
  protected boolean canReportCycle(NodeKey topLevelKey, CycleInfo cycleInfo) {
    return Iterables.all(Iterables.concat(ImmutableList.of(topLevelKey),
        cycleInfo.getPathToCycle(), cycleInfo.getCycle()), IS_CONFIGURED_TARGET_NODE_KEY);
  }

  @Override
  public String prettyPrint(NodeKey key) {
    return ((LabelAndConfiguration) key.getNodeName()).prettyPrint();
  }

  @Override
  public Label getLabel(NodeKey key) {
    return ((LabelAndConfiguration) key.getNodeName()).getLabel();
  }
}
