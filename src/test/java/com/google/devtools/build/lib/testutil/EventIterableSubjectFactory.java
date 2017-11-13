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
package com.google.devtools.build.lib.testutil;

import com.google.common.truth.FailureMetadata;
import com.google.common.truth.IterableSubject;
import com.google.common.truth.Subject;
import com.google.common.truth.Truth;
import com.google.devtools.build.lib.events.Event;

/**
 * {@link Subject.Factory} for {@code Iterable<Event>} objects, providing {@link IterableSubject}s
 * of {@link String} objects for easy asserting.
 */
public class EventIterableSubjectFactory
    implements Subject.Factory<EventIterableSubject, Iterable<Event>> {
  public static IterableSubject assertThatEvents(Iterable<Event> events) {
    return Truth.assertAbout(new EventIterableSubjectFactory()).that(events).hasEventsThat();
  }

  @Override
  public EventIterableSubject createSubject(
      FailureMetadata failureMetadata, Iterable<Event> eventCollector) {
    return new EventIterableSubject(failureMetadata, eventCollector);
  }
}
