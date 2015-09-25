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

package com.google.devtools.build.xcode.util;

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.base.Optional;

import java.io.File;
import java.nio.file.Path;
import java.util.List;
import java.util.Set;

/**
 * Provides utility methods that make equality comparison type safe. The
 * {@link Object#equals(Object)} method usually returns false when the other object is {@code null}
 * or a different runtime class. These utility methods try to force each object to be of the same
 * class (with the method signatures) and non-null (and throwing a {@link NullPointerException} if
 * either one is null.
 */
public class Equaling {
  private Equaling() {
    throw new UnsupportedOperationException("static-only");
  }

  // Note that we always checkNotNull(b) on a separate line from a.equals(b). This is to make it so
  // the stack trace will tell you exactly which reference is null.

  public static <T extends Value<T>> boolean of(Value<T> a, Value<T> b) {
    checkNotNull(b);
    return a.equals(b);
  }

  public static boolean of(String a, String b) {
    checkNotNull(b);
    return a.equals(b);
  }

  public static <T> boolean of(Optional<T> a, Optional<T> b) {
    checkNotNull(b);
    return a.equals(b);
  }

  public static <T> boolean of(Set<T> a, Set<T> b) {
    checkNotNull(b);
    return a.equals(b);
  }

  public static boolean of(File a, File b) {
    checkNotNull(b);
    return a.equals(b);
  }

  public static boolean of(Path a, Path b) {
    checkNotNull(b);
    return a.equals(b);
  }

  public static <T> boolean of(List<T> a, List<T> b) {
    checkNotNull(b);
    return a.equals(b);
  }

  public static <T extends Enum<T>> boolean of(Enum<T> a, Enum<T> b) {
    checkNotNull(b);
    return a.equals(b);
  }
}
