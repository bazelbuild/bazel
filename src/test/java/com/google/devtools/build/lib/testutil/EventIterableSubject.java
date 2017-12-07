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

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.Iterables;
import com.google.common.truth.FailureMetadata;
import com.google.common.truth.IterableSubject;
import com.google.common.truth.Subject;
import com.google.devtools.build.lib.events.Event;
import javax.annotation.Nullable;

/**
 * {@link Subject} for {@code Iterable<Event>} that provides an {@link IterableSubject} of {@link
 * String} objects as opposed to the harder-to-assert-on {@link Event} objects.
 */
class EventIterableSubject extends Subject<EventIterableSubject, Iterable<Event>> {
  EventIterableSubject(FailureMetadata failureMetadata, @Nullable Iterable<Event> actual) {
    super(failureMetadata, actual);
  }

  IterableSubject hasEventsThat() {
    return assertThat(Iterables.transform(actual(), Event::getMessage));
  }
}
