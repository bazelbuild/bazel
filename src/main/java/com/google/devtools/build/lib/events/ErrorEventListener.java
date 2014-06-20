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

/**
 * The ErrorEventListener is the primary means of reporting error and warning events. It is a subset
 * of the functionality of the {@link Reporter}. In most cases, you should use this interface
 * instead of the final {@code Reporter} class.
 */
public interface ErrorEventListener {

  /**
   * Reports a warning.
   */
  void warn(Location location, String message);

  /**
   * Reports an error.
   */
  void error(Location location, String message);

  /**
   * Reports atemporal statements about the build, i.e. they're true for the duration of execution.
   */
  void info(Location location, String message);

  /**
   * Reports a temporal statement about the build.
   */
  void progress(Location location, String message);

  /**
   * Reports and arbitrary event.
   */
  void report(EventKind kind, Location location, String message);

  /**
   * Returns true iff the given tag matches the output filter.
   */
  // TODO(bazel-team): We probably don't need this; when we have one instance per configured
  // target, we can filter on that instead.
  boolean showOutput(String tag);
}
