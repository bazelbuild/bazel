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
package com.google.devtools.build.lib.analysis;

/**
 * An exception indicating that there was a problem during the view
 * construction (loading and analysis phases) for one or more targets, that the
 * configured target graph could not be successfully constructed, and that
 * a build cannot be started.
 */
public class ViewCreationFailedException extends Exception {

  public ViewCreationFailedException(String message) {
    super(message);
  }

  public ViewCreationFailedException(String message, Throwable cause) {
    super(message + ": " + cause.getMessage(), cause);
  }

  public ViewCreationFailedException(Throwable cause) {
    super(cause);
  }
}
