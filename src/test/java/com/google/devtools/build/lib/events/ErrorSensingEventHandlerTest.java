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

import static com.google.common.truth.Truth.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests {@link ErrorSensingEventHandler}. */
@RunWith(JUnit4.class)
public class ErrorSensingEventHandlerTest {

  @Test
  public void delegation() {
    ExtendedEventHandler delegate = mock(ExtendedEventHandler.class);
    ErrorSensingEventHandler<Void> subject =
        ErrorSensingEventHandler.withoutPropertyValueTracking(delegate);
    Event event = Event.of(EventKind.INFO, "message");

    subject.handle(event);

    verify(delegate).handle(event);
  }

  @Test
  public void rememberError() {
    ExtendedEventHandler delegate = mock(ExtendedEventHandler.class);
    ErrorSensingEventHandler<Void> subject =
        ErrorSensingEventHandler.withoutPropertyValueTracking(delegate);

    subject.handle(Event.of(EventKind.INFO, "message"));

    assertThat(subject.hasErrors()).isFalse();

    subject.handle(Event.of(EventKind.ERROR, "anError"));

    assertThat(subject.hasErrors()).isTrue();
  }

  @Test
  public void rememberErrorProperty() {
    ExtendedEventHandler delegate = mock(ExtendedEventHandler.class);

    ErrorSensingEventHandler<Void> withoutTracking =
        ErrorSensingEventHandler.withoutPropertyValueTracking(delegate);
    ErrorSensingEventHandler<String> withTracking =
        new ErrorSensingEventHandler<>(delegate, String.class);

    Event nonerrorEvent = Event.info("nonerror").withProperty(String.class, "propertyValue");
    withoutTracking.handle(nonerrorEvent);
    withTracking.handle(nonerrorEvent);

    assertThat(withoutTracking.getErrorProperty()).isNull();
    assertThat(withTracking.getErrorProperty()).isNull();

    Event errorEvent = Event.error("anError").withProperty(String.class, "propertyValue");
    withoutTracking.handle(errorEvent);
    withTracking.handle(errorEvent);

    assertThat(withoutTracking.getErrorProperty()).isNull();
    assertThat(withTracking.getErrorProperty()).isEqualTo("propertyValue");

    withTracking.handle(Event.error("anotherError").withProperty(String.class, "ignoredValue"));
    assertThat(withTracking.getErrorProperty()).isEqualTo("propertyValue");
  }
}
