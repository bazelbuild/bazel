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
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.events.ErrorEventListener;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.CycleInfo;
import com.google.devtools.build.skyframe.CyclesReporter;
import com.google.devtools.build.skyframe.NodeKey;

import javax.annotation.Nullable;

/**
 * Reports cycles between {@link FileNode}s. These indicate file symlink cycles (e.g. 'a' is a
 * symlink to 'b' and 'b' is a symlink to 'a').
 */
class FileSymlinkCycleReporter implements CyclesReporter.SingleCycleReporter {

  private static final Predicate<NodeKey> IS_FILE_NODE_KEY = NodeTypes.hasNodeType(NodeTypes.FILE);

  @Override
  public boolean maybeReportCycle(NodeKey topLevelKey, CycleInfo cycleInfo, boolean alreadyReported,
      @Nullable ErrorEventListener listener) {
    if (!Iterables.all(cycleInfo.getCycle(), IS_FILE_NODE_KEY)) {
      return false;
    }

    if (listener == null || alreadyReported) {
      return true;
    }

    StringBuilder cycleMessage = new StringBuilder("circular symlinks detected\n");
    cycleMessage.append("[start of symlink cycle]\n");
    for (NodeKey nodeKey : cycleInfo.getCycle()) {
      RootedPath rootedPath = (RootedPath) nodeKey.getNodeName();
      cycleMessage.append(rootedPath.asPath() + "\n");
    }
    cycleMessage.append("[end of symlink cycle]");

    listener.error(null, cycleMessage.toString());
    return true;
  }
}
