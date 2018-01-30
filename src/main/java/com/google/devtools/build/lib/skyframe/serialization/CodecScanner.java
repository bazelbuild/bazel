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

import com.google.common.reflect.ClassPath;
import com.google.common.reflect.ClassPath.ClassInfo;
import java.io.IOException;
import java.util.stream.Stream;

/** Scans the classpath to find codec instances. */
public class CodecScanner {

  /**
   * Returns a stream of likely codec implementations.
   *
   * <p>Caller should do additional checks as this method only performs string matching.
   *
   * @param packagePrefix emits only classes in packages having this prefix
   */
  public static Stream<Class<?>> scanCodecs(String packagePrefix) throws IOException {
    return ClassPath.from(ClassLoader.getSystemClassLoader())
        .getResources()
        .stream()
        .filter(r -> r instanceof ClassInfo)
        .map(r -> (ClassInfo) r)
        .filter(c -> c.getPackageName().startsWith(packagePrefix))
        .filter(c -> c.getSimpleName().endsWith("Codec"))
        .map(c -> c.load());
  }
}
