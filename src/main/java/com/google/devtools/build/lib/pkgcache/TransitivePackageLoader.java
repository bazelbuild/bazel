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
package com.google.devtools.build.lib.pkgcache;

import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import java.util.Set;

/**
 * Visits a set of Targets and Labels transitively.
 */
public interface TransitivePackageLoader {

  /**
   * Visit the specified labels and follow the transitive closure of their outbound dependencies. If
   * the targets have previously been visited, may do an up-to-date check which will not trigger any
   * of the observers.
   *
   * @param eventHandler the error and warnings eventHandler; must be thread-safe
   * @param labelsToVisit the labels to visit in addition to the targets
   * @param keepGoing if false, stop visitation upon first error
   * @param parallelThreads number of threads to use in the visitation
   */
  boolean sync(
      ExtendedEventHandler eventHandler,
      Set<Label> labelsToVisit,
      boolean keepGoing,
      int parallelThreads)
      throws InterruptedException;
}
