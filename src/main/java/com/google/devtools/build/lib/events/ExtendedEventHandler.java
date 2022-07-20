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

  /** An event that can be posted via the extended event handler. */
  interface Postable {

    /**
     * If this post originated from {@link
     * com.google.devtools.build.skyframe.SkyFunction.Environment#getListener}, whether it should be
     * stored in the corresponding Skyframe node to be replayed on incremental builds when the node
     * is deemed up-to-date.
     *
     * <p>Posts which are crucial to the correctness of the evaluation should return {@code true} so
     * that they are replayed when the {@link com.google.devtools.build.skyframe.SkyFunction}
     * invocation is cached. On the other hand, posts that are merely informational (such as a
     * progress update) should return {@code false} to avoid taking up memory.
     *
     * <p>This method is not relevant for posts which do not originate from {@link
     * com.google.devtools.build.skyframe.SkyFunction} evaluation.
     */
    default boolean storeForReplay() {
      return false;
    }
  }

  /** Posts a {@link Postable} object about an important build event. */
  void post(Postable obj);

  /** A progress event that reports about fetching from a remote site. */
  interface FetchProgress extends Postable {

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

}
