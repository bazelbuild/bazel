/*
 * Copyright 2020 The Bazel Authors. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.devtools.build.android.desugar.testing.junit;

import com.google.errorprone.annotations.FormatMethod;
import java.util.ArrayList;
import java.util.List;

/** A messenger that manages desugar configuration errors. */
class ErrorMessenger {

  private final List<String> errorMessages = new ArrayList<>();

  @FormatMethod
  ErrorMessenger addError(String recipe, Object... args) {
    errorMessages.add(String.format(recipe, args));
    return this;
  }

  boolean containsAnyError() {
    return !errorMessages.isEmpty();
  }

  List<String> getAllMessages() {
    return errorMessages;
  }

  @Override
  public String toString() {
    return getAllMessages().toString();
  }
}
