// Copyright 2014 Google Inc. All rights reserved.
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

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests the {@link StoredEventHandler} class.
 */
@RunWith(JUnit4.class)
public class WarningsAsErrorsEventHandlerTest {

  @Test
  public void hasErrors() {
    ErrorSensingEventHandler delegate =
        new ErrorSensingEventHandler(NullEventHandler.INSTANCE);
    WarningsAsErrorsEventHandler eventHandler =
        new WarningsAsErrorsEventHandler(delegate);

    eventHandler.handle(Event.info("info"));
    assertFalse(delegate.hasErrors());

    eventHandler.handle(Event.warn("warning"));
    assertTrue(delegate.hasErrors());

    delegate.resetErrors();

    eventHandler.handle(Event.error("error"));
    assertTrue(delegate.hasErrors());
  }
}
