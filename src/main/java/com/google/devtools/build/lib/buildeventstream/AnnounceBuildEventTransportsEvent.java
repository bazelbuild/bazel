// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.buildeventstream;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.events.ExtendedEventHandler.Postable;
import java.util.Collection;
import java.util.Set;

/**
 * An event announcing a list of all active {@link BuildEventTransport}s.
 */
public class AnnounceBuildEventTransportsEvent implements Postable {

  private final Set<BuildEventTransport> transports;

  public AnnounceBuildEventTransportsEvent(Collection<BuildEventTransport> transports) {
    this.transports = ImmutableSet.copyOf(transports);
  }

  /**
   * Returns a list of all active build event transports.
   */
  public Set<BuildEventTransport> transports() {
    return transports;
  }
}
