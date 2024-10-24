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
package com.google.devtools.build.lib.analysis.config;

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.devtools.build.lib.events.ExtendedEventHandler.Postable;

/**
 * Event which indicates a configuration change besides build configuration-affecting options.
 *
 * <p>Posting this event will result in the analysis cache being discarded. It is expected to be
 * posted in {@link com.google.devtools.build.lib.runtime.BlazeModule#beforeCommand} so that it can
 * be received in time to clear the analysis cache.
 */
public final class AdditionalConfigurationChangeEvent implements Postable {
  private final String changeDescription;

  public AdditionalConfigurationChangeEvent(String changeDescription) {
    this.changeDescription = checkNotNull(changeDescription);
  }

  /** Returns a description of the changed configuration. */
  public String getChangeDescription() {
    return changeDescription;
  }
}
