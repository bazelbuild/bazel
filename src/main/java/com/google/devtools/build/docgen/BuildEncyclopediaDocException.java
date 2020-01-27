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

/**
 * An exception for Build Encyclopedia generation implementing the common BLAZE
 * error formatting, i.e. displaying file name and line number.
 */
public class BuildEncyclopediaDocException extends Exception {

  private String fileName;
  private int lineNumber;
  private String errorMsg;

  public BuildEncyclopediaDocException(String fileName, int lineNumber, String errorMsg) {
    this.fileName = fileName;
    this.lineNumber = lineNumber;
    this.errorMsg = errorMsg;
  }

  public String getFileName() {
    return fileName;
  }

  public int getLineNumber() {
    return lineNumber;
  }

  public String getErrorMsg() {
    return errorMsg;
  }

  @Override
  public String getMessage() {
    return "Error in " + fileName + ":" + lineNumber + ": " + errorMsg;
  }
}
