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
package com.google.devtools.build.lib.query2;

import static java.util.stream.Collectors.joining;

import com.google.common.collect.Streams;
import com.google.devtools.build.lib.query2.engine.ThreadSafeOutputFormatterCallback;

/** A {@link ThreadSafeOutputFormatterCallback} that has a name to select on. */
public abstract class NamedThreadSafeOutputFormatterCallback<T>
    extends ThreadSafeOutputFormatterCallback<T> {
  public abstract String getName();

  public static <T> String callbackNames(
      Iterable<NamedThreadSafeOutputFormatterCallback<T>> callbacks) {
    return Streams.stream(callbacks)
        .map(NamedThreadSafeOutputFormatterCallback::getName)
        .collect(joining(", "));
  }

  public static <T> NamedThreadSafeOutputFormatterCallback<T> selectCallback(
      String type, Iterable<NamedThreadSafeOutputFormatterCallback<T>> callbacks) {
    for (NamedThreadSafeOutputFormatterCallback<T> callback : callbacks) {
      if (callback.getName().equals(type)) {
        return callback;
      }
    }
    return null;
  }
}
