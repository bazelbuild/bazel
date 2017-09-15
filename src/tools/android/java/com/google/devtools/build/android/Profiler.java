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

package com.google.devtools.build.android;

/** Provides a simple api for profiling task duration. */
public interface Profiler {

  Profiler startTask(String taskName);

  Profiler recordEndOf(String taskName);

  static Profiler empty() {
    return new Profiler() {
      @Override
      public Profiler startTask(String taskName) {
        return this;
      }

      @Override
      public Profiler recordEndOf(String taskName) {
        return this;
      }
    };
  }
}
