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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.ExtendedEventHandler;

/**
 * This event is fired after the loading phase is complete.
 */
public final class LoadingPhaseCompleteEvent implements ExtendedEventHandler.Postable {
  private final ImmutableSet<Label> labels;
  private final ImmutableSet<Label> filteredLabels;

  /**
   * Construct the event.
   *
   * @param labels the set of active targets that remain
   * @param filteredLabels the set of filtered targets
   */
  public LoadingPhaseCompleteEvent(
      ImmutableSet<Label> labels,
      ImmutableSet<Label> filteredLabels) {
    this.labels = Preconditions.checkNotNull(labels);
    this.filteredLabels = Preconditions.checkNotNull(filteredLabels);
  }

  /**
   * @return The set of active target labels remaining, which is a subset of the
   *         targets we attempted to load.
   */
  public ImmutableSet<Label> getLabels() {
    return labels;
  }

  /**
   * @return The set of filtered targets.
   */
  public ImmutableSet<Label> getFilteredLabels() {
    return filteredLabels;
  }
}
