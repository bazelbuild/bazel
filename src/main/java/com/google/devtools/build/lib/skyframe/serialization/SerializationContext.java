// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skyframe.serialization;

/** Stateful class for providing additional context to a single serialization "session". */
// TODO(bazel-team): This class is just a shell, fill in.
public class SerializationContext {
  // TODO(bazel-team): Replace with real stateless implementation when we start adding
  // functionality.
  private static final SerializationContext EMPTY_STATELESS = new SerializationContext();

  public static SerializationContext create() {
    return stateless();
  }

  /** Returns an empty instance which doesn't retain any state. */
  public static SerializationContext stateless() {
    return EMPTY_STATELESS;
  }

  private SerializationContext() {}
}
