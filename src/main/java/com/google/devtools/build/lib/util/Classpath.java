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
package com.google.devtools.build.lib.util;

import com.google.common.reflect.ClassPath;
import com.google.common.reflect.ClassPath.ClassInfo;
import java.io.IOException;
import java.util.LinkedHashSet;
import java.util.Set;

/**
 * A helper class to find all classes on the current classpath. This is used to automatically create
 * JUnit 3 and 4 test suites.
 */
public final class Classpath {

  /**
   * Base exception for any classpath related errors.
   */
  public static final class ClassPathException extends Exception {
    public ClassPathException(String format, Object... args) {
      super(String.format(format, args));
    }
  }

  /** Finds all classes that live in or below the given package. */
  public static Set<Class<?>> findClasses(String packageName) throws ClassPathException {
    Set<Class<?>> result = new LinkedHashSet<>();
    String packagePrefix = (packageName + '.').replace('/', '.');
    try {
      for (ClassInfo ci : ClassPath.from(Classpath.class.getClassLoader()).getAllClasses()) {
        if (ci.getName().startsWith(packagePrefix)) {
          try {
            result.add(ci.load());
          } catch (UnsatisfiedLinkError | NoClassDefFoundError unused) {
            // Ignore: we're most likely running on a different platform.
          }
        }
      }
    } catch (IOException e) {
      throw new ClassPathException(e.getMessage());
    }
    return result;
  }
}
