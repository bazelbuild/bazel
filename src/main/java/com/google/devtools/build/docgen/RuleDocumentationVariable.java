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
 * Rule documentation variables for modular rule documentation, e.g.
 * separate section for Implicit outputs.
 */
public class RuleDocumentationVariable {

  private String ruleName;
  private String variableName;
  private String value;
  private int startLineCnt;

  public RuleDocumentationVariable(
      String ruleName, String variableName, String value, int startLineCnt) {
    this.ruleName = ruleName;
    this.variableName = variableName;
    this.value = value;
    this.startLineCnt = startLineCnt;
  }

  public String getRuleName() {
    return ruleName;
  }

  public String getVariableName() {
    return variableName;
  }

  public String getValue() {
    return value;
  }

  public int getStartLineCnt() {
    return startLineCnt;
  }
}
