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
package com.google.devtools.build.lib.rules.java;

import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;

/**
 * Provides the fully qualified name of the primary class to invoke for java targets.
 */
@Immutable
public final class JavaPrimaryClassProvider implements TransitiveInfoProvider {
  private final String primaryClass;

  public JavaPrimaryClassProvider(String primaryClass) {
    this.primaryClass = primaryClass;
  }

  /**
   * Returns either the Java class whose main() method is to be invoked (when
   * use_testrunner=0) or the Java subclass of junit.framework.Test that
   * is to be tested by the test runner class (when use_testrunner=1).
   *
   * @return a fully qualified Java class name, or null if none could be
   *   determined.
   */
  public String getPrimaryClass() {
    return primaryClass;
  }
}
