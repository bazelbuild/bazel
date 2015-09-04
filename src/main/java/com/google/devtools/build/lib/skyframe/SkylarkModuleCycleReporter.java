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

import com.google.common.base.Function;
import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.skyframe.CycleInfo;
import com.google.devtools.build.skyframe.CyclesReporter;
import com.google.devtools.build.skyframe.SkyKey;

/**
 * Reports cycles of recursive import of Skylark files.
 */
public class SkylarkModuleCycleReporter implements CyclesReporter.SingleCycleReporter {

  private static final Predicate<SkyKey> IS_SKYLARK_MODULE_SKY_KEY =
      SkyFunctions.isSkyFunction(SkyFunctions.SKYLARK_IMPORTS_LOOKUP);

  private static final Predicate<SkyKey> IS_PACKAGE_SKY_KEY =
      SkyFunctions.isSkyFunction(SkyFunctions.PACKAGE);

  @Override
  public boolean maybeReportCycle(SkyKey topLevelKey, CycleInfo cycleInfo, boolean alreadyReported,
      EventHandler eventHandler) {
    ImmutableList<SkyKey> pathToCycle = cycleInfo.getPathToCycle();
    if (pathToCycle.isEmpty()) {
      return false;
    }
    SkyKey lastPathElement = cycleInfo.getPathToCycle().get(pathToCycle.size() - 1);
    if (alreadyReported) {
      return true;
    } else if (Iterables.all(cycleInfo.getCycle(), IS_SKYLARK_MODULE_SKY_KEY)
        // The last element of the path to the cycle has to be a PackageFunction.
        && IS_PACKAGE_SKY_KEY.apply(lastPathElement)) {
      StringBuilder cycleMessage =
          new StringBuilder()
              .append(lastPathElement.argument()).append("/BUILD: ")
              .append("cycle in referenced extension files: ");

      AbstractLabelCycleReporter.printCycle(cycleInfo.getCycle(), cycleMessage,
          new Function<SkyKey, String>() {
        @Override
        public String apply(SkyKey input) {
          return ((PackageIdentifier) input.argument()).toString();
        }
      });

      // TODO(bazel-team): it would be nice to pass the Location of the load Statement in the
      // BUILD file.
      eventHandler.handle(Event.error(null, cycleMessage.toString()));
      return true;
    }
    return false;
  }
}
