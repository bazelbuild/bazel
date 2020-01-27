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

package com.google.devtools.build.lib.buildtool;

import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.ExitCode;

/**
 * An exception that signals that something is wrong with the user's environment
 * that they can fix. Used to report the problem of having no free space left in
 * the blaze output directory.
 *
 * <p>Note that this is a much higher level exception then the similarly named
 * EnvironmentExecException, which is thrown from the base Client and Strategy
 * layers of Blaze.
 *
 * <p>This exception is only thrown when we've decided that the build has, in
 * fact, failed and we should exit.
 */
public class LocalEnvironmentException extends AbruptExitException {

  public LocalEnvironmentException(String message) {
    super(message, ExitCode.LOCAL_ENVIRONMENTAL_ERROR);
  }

  public LocalEnvironmentException(Throwable cause) {
    super(ExitCode.LOCAL_ENVIRONMENTAL_ERROR, cause);
  }

  public LocalEnvironmentException(String message, Throwable cause) {
    super(message, ExitCode.LOCAL_ENVIRONMENTAL_ERROR, cause);
  }
}
