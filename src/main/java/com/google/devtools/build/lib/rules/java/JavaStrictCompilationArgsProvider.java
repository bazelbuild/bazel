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

package com.google.devtools.build.lib.rules.java;

import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import java.util.Collection;
import java.util.stream.Collectors;

/**
 * This class just wraps a JavaCompilationArgsProvider.
 *
 * <p>It is used to enforce strict_deps from a downstream java_library to an upstream
 * java_proto_library.
 *
 * <p>TODO(twerth): Remove this class once we finished migration.
 */
public class JavaStrictCompilationArgsProvider implements TransitiveInfoProvider {
  private final JavaCompilationArgsProvider javaCompilationArgsProvider;

  public JavaStrictCompilationArgsProvider(
      JavaCompilationArgsProvider javaCompilationArgsProvider) {
    this.javaCompilationArgsProvider = javaCompilationArgsProvider;
  }

  public JavaCompilationArgsProvider getJavaCompilationArgsProvider() {
    return javaCompilationArgsProvider;
  }

  public static JavaStrictCompilationArgsProvider merge(
      Collection<JavaStrictCompilationArgsProvider> providers) {
    Collection<JavaCompilationArgsProvider> javaCompilationArgsProviders =
        providers.stream()
            .map(JavaStrictCompilationArgsProvider::getJavaCompilationArgsProvider)
            .collect(Collectors.toList());
    return new JavaStrictCompilationArgsProvider(
        JavaCompilationArgsProvider.merge(javaCompilationArgsProviders));
  }
}
