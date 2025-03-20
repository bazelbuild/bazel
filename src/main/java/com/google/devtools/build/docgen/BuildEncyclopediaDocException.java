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
package com.google.devtools.build.docgen;

/** An exception in Build Encyclopedia generation. */
public class BuildEncyclopediaDocException extends Exception {

  private final String location;
  private final String errorMsg;

  BuildEncyclopediaDocException(String location, String errorMsg) {
    this.location = location;
    this.errorMsg = errorMsg;
  }

  BuildEncyclopediaDocException(String file, int lineNumber, String errorMsg) {
    this.location = formatLocation(file, lineNumber);
    this.errorMsg = errorMsg;
  }

  static String formatLocation(String file, int lineNumber) {
    return String.format("%s:%d", file, lineNumber);
  }

  /** Returns the location (filename or label, possibly with a line number) of the error. */
  public String getLocation() {
    return location;
  }

  /** Returns the error message text. */
  public String getErrorMsg() {
    return errorMsg;
  }

  @Override
  public String getMessage() {
    return String.format("Error in %s: %s", location, errorMsg);
  }
}
