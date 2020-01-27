// Copyright 2016 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.util.Pair;
import java.util.LinkedHashSet;

/**
 * A class that, when being told about start and end of a package being loaded, keeps track of the
 * loading progress and provides it as a human-readable string intended for the progress bar.
 */
public class PackageProgressReceiver {

  private int packagesCompleted;
  private LinkedHashSet<PackageIdentifier> pendingSet = new LinkedHashSet<>();

  /** Register that loading a package has started. */
  synchronized void startReadPackage(PackageIdentifier packageId) {
    pendingSet.add(packageId);
  }

  /** Register that loding a package has completed. */
  synchronized void doneReadPackage(PackageIdentifier packageId) {
    packagesCompleted++;
    pendingSet.remove(packageId);
  }

  /**
   * Reset all instance variables of this object to a state equal to that of a newly
   * constructed object.
   */
  public synchronized void reset() {
    packagesCompleted = 0;
    pendingSet = new LinkedHashSet<>();
  }

  /**
   * Return the ordered pair of a consistent snapshot of the state, consisting of a human-readable
   * description of the progress achieved so far and a human readable description of the currently
   * running activities. The later always include the oldest loading package not finished loading.
   */
  public synchronized Pair<String, String> progressState() {
    String progress = "" + packagesCompleted + " packages loaded";
    StringBuffer activity = new StringBuffer();
    if (pendingSet.size() > 0) {
      activity
          .append("currently loading: ")
          .append(Iterables.getFirst(pendingSet, null).toString());
      if (pendingSet.size() > 1) {
        activity.append(" ... (" + pendingSet.size() + " packages)");
      }
    }
    return new Pair<String, String>(progress, activity.toString());
  }
}
