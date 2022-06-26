// Copyright 2021 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.actions;

/**
 * Ancestor for exceptions thrown because of a bad input: an input that is not allowed for the given
 * spawn/execution platform (like a relative symlink in a Fileset or a directory for a platform that
 * does not support directory inputs). Indicates a user error, rather than an I/O error.
 */
public class ForbiddenActionInputException extends Exception {
  protected ForbiddenActionInputException(String message) {
    super(message);
  }
}
