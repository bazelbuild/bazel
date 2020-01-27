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
package com.google.devtools.build.android;

/**
 * An exception triggered by a user error (including problems with input).
 *
 * <p>In Bazel, users tend to assume that a stack trace indicates a bug in underlying Bazel code and
 * ignore the content of the exception. If we know that the exception was actually their fault, we
 * should just exit immediately rather than print a stack trace.
 */
public class UserException extends RuntimeException {
  UserException(String message, Throwable e) {
    super(message, e);
  }

  UserException(String message) {
    super(message);
  }
}
