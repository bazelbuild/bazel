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

package com.google.devtools.build.lib.events;

/**
 * Interface for reporting events during the build. It extends the {@link EventHandler} by also
 * allowing posting more structured information.
 */
public interface ExtendedEventHandler extends EventHandler {

  /** Interface for declaring events that can be posted via the extended event handler */
  interface Postable {}

  /** Post an postable object with more refined information about an important build event */
  void post(Postable obj);

  /**
   * Interface for declaring postable events that report about progress (as opposed to success or
   * failure) and hence should not be stored and replayed.
   */
  interface ProgressLike extends Postable {}

  /** Interface for progress events that report about fetching from a remote site */
  interface FetchProgress extends ProgressLike {

    /**
     * The resource that was originally requested and uniquely determines the fetch source. The
     * actual fetching may use mirrors, proxies, or similar. The resource need not be an URL, but it
     * has to uniquely identify the particular fetch among all fetch events.
     */
    String getResourceIdentifier();

    /** Human readable description of the progress */
    String getProgress();

    /** Wether the fetch progress reported about is finished already */
    boolean isFinished();
  }

  /** Interface for events reporting information to be added to a resolved file. */
  interface ResolvedEvent extends ProgressLike {

    /** The name of the resolved entity, e.g., the name of an external repository */
    String getName();

    /** The entry for the list of resolved Information. */
    Object getResolvedInformation();
  }
}
