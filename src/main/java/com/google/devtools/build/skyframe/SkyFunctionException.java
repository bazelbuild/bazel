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
package com.google.devtools.build.skyframe;


import com.google.common.base.Preconditions;

/**
 * Base class of exceptions thrown by {@link SkyFunction#build} on failure.
 *
 * SkyFunctions should declare a subclass {@code C} of {@link SkyFunctionException} whose
 * constructors forward fine-grained exception types (e.g. {@link IOException}) to
 * {@link SkyFunctionException}'s constructor, and they should also declare
 * {@link SkyFunction#build} to throw {@code C}. This way the type system checks that no unexpected
 * exceptions are thrown by the {@link SkyFunction}.
 *
 * We took this approach over using a generic exception class Java since disallows it because of
 * type erasure
 * (see http://docs.oracle.com/javase/tutorial/java/generics/restrictions.html#cannotCatch).
 */
public abstract class SkyFunctionException extends Exception {
  private final SkyKey rootCause;
  private final boolean isTransient;

  public SkyFunctionException(SkyKey rootCause, Throwable cause) {
    super(Preconditions.checkNotNull(cause));
    this.rootCause = Preconditions.checkNotNull(rootCause, cause);
    // TODO(bazel-team): We may want to switch the default to false at some point.
    this.isTransient = true;
  }

  public SkyFunctionException(SkyKey rootCause, Throwable cause, boolean isTransient) {
    super(Preconditions.checkNotNull(cause));
    this.rootCause = Preconditions.checkNotNull(rootCause, cause);
    this.isTransient = isTransient;
  }

  final SkyKey getRootCauseSkyKey() {
    return rootCause;
  }

  final boolean isTransient() {
    return isTransient;
  }

  /**
   * Catastrophic failures halt the build even when in keepGoing mode.
   */
  public boolean isCatastrophic() {
    return false;
  }
}
