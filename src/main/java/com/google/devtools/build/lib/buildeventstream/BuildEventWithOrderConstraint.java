// Copyright 2016 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.buildeventstream;

import java.util.Collection;

/** Interface for {@link BuildEvent}s with order constraints. */
public interface BuildEventWithOrderConstraint extends BuildEvent {
  /**
   * Specify events that need to come first.
   *
   * <p>Specify events by their {@link BuildEventId} that need to be posted on the build event
   * stream before this event. In doing so, the event promises that the events to be waited for are
   * already generated, so that the event does not have to be buffered for an extended time.
   */
  Collection<BuildEventId> postedAfter();
}
