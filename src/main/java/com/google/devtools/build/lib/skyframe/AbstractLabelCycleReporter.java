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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.events.ErrorEventListener;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.pkgcache.LoadedPackageProvider;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.skyframe.CycleInfo;
import com.google.devtools.build.skyframe.CyclesReporter;
import com.google.devtools.build.skyframe.NodeKey;

import javax.annotation.Nullable;

/** Reports cycles between skyframe nodes whose keys contains {@link Label}s. */
abstract class AbstractLabelCycleReporter implements CyclesReporter.SingleCycleReporter {

  private final LoadedPackageProvider loadedPackageProvider;

  AbstractLabelCycleReporter(LoadedPackageProvider loadedPackageProvider) {
    this.loadedPackageProvider = loadedPackageProvider;
  }

  /** Returns the String representation of the {@code NodeKey}. */
  protected abstract String prettyPrint(NodeKey key);

  /** Returns the associated Label of the NodeKey. */
  protected abstract Label getLabel(NodeKey key);

  protected abstract boolean canReportCycle(NodeKey topLevelKey, CycleInfo cycleInfo);

  @Override
  public boolean maybeReportCycle(NodeKey topLevelKey, CycleInfo cycleInfo,
      boolean alreadyReported, @Nullable ErrorEventListener listener) {
    if (!canReportCycle(topLevelKey, cycleInfo)) {
      return false;
    }

    if (listener == null) {
      return true;
    }

    if (alreadyReported) {
      Label label = getLabel(topLevelKey);
      Target target = getTargetForLabel(label);
      listener.error(target.getLocation(),
          "in " + target.getTargetKind() + " " + label +
              ": cycle in dependency graph: target depends on an already-reported cycle");
    } else {
      StringBuilder cycleMessage = new StringBuilder("cycle in dependency graph:");
      ImmutableList<NodeKey> pathToCycle = cycleInfo.getPathToCycle();
      ImmutableList<NodeKey> cycle = cycleInfo.getCycle();
      for (NodeKey node : pathToCycle) {
        cycleMessage.append("\n    ");
        cycleMessage.append(prettyPrint(node));
      }

      Iterable<NodeKey> nodesToPrint = cycle.size() > 1 ?
          Iterables.concat(cycle, ImmutableList.of(cycle.get(0))) : cycle;
      NodeKey cycleNode = null;
      for (NodeKey node : nodesToPrint) {
        if (cycleNode == null) {
          cycleNode = node;
        }
        if (node == cycleNode) {
          cycleMessage.append("\n  * ");
        } else {
          cycleMessage.append("\n    ");
        }
        cycleMessage.append(prettyPrint(node));
      }

      if (cycle.size() == 1) {
        cycleMessage.append(" [self-edge]");
      }

      Label label = getLabel(cycleNode);
      Target target = getTargetForLabel(label);
      listener.error(target.getLocation(), "in " + target.getTargetKind() + " " + label
          + ": " + cycleMessage.toString());
    }

    return true;
  }

  private Target getTargetForLabel(Label label) {
    try {
      return loadedPackageProvider.getLoadedTarget(label);
    } catch (NoSuchThingException e) {
      // This method is used for getting the target from a label in a circular dependency.
      // If we have a cycle that means that we need to have accessed the target (to get its
      // dependencies). So all the labels in a dependency cycle need to exist.
      throw new IllegalStateException(e);
    }
  }
}
