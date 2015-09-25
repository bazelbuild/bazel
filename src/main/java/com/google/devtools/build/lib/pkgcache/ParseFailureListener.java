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

import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.events.EventHandler;

/**
 * Represents a listener which reports parse errors to the underlying
 * {@link EventHandler} and {@link EventBus} (if non-null).
 */
public interface ParseFailureListener {

  /** Reports a parsing failure. */
  void parsingError(String badPattern, String message);
}
