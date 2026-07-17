// Copyright 2022 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.vfs.util;

import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.UnixGlobPathDiscriminator;
import com.google.errorprone.annotations.CheckReturnValue;
import java.util.function.BiPredicate;
import java.util.function.Predicate;

/**
 * Test version of UnixGlobPathDiscriminator that accepts predicate/bipredicate for handling
 * specific use-cases without creating a new class.
 */
@CheckReturnValue
public final class TestUnixGlobPathDiscriminator implements UnixGlobPathDiscriminator {

  private final Predicate<Path> traversalPredicate;
  private final BiPredicate<Path, Boolean> resultPredicate;

  public TestUnixGlobPathDiscriminator(
      Predicate<Path> traversalPredicate, BiPredicate<Path, Boolean> resultPredicate) {
    this.traversalPredicate = traversalPredicate;
    this.resultPredicate = resultPredicate;
  }

  @Override
  public boolean shouldTraverseDirectory(Path path) {
    return traversalPredicate.test(path);
  }

  @Override
  public boolean shouldIncludePathInResult(Path path, boolean isDirectory) {
    return resultPredicate.test(path, isDirectory);
  }
}
