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

package com.google.devtools.build.lib.pkgcache;

import com.google.devtools.build.lib.events.ExtendedEventHandler;

/**
 * This event is fired when a target or target pattern fails to parse.
 * In some cases (not all) this happens before targets are created,
 * and thus in these cases there are no status lines.
 * Therefore, the parse failure is reported separately.
 */
public class ParsingFailedEvent implements ExtendedEventHandler.Postable {
  private final String targetPattern;
  private final String message;

  /**
   * Creates a new parsing failed event with the given pattern and message.
   */
  public ParsingFailedEvent(String targetPattern, String message) {
    this.targetPattern = targetPattern;
    this.message = message;
  }

  public String getPattern() {
    return targetPattern;
  }

  public String getMessage() {
    return message;
  }
}
