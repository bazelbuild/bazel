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

package com.google.devtools.build.lib.profiler;

/**
 * Allows class to implement profiler-friendly (and user-friendly)
 * textual description of the object that would uniquely identify an object in
 * the profiler data dump.
 */
public interface Describable {

  /**
   * Returns textual description that will uniquely identify an object.
   */
  String describe();

}
