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

package com.google.devtools.build.lib.analysis.test;

import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skylarkbuildapi.test.AnalysisFailureApi;
import com.google.devtools.build.lib.skylarkinterface.SkylarkPrinter;

/**
 * Encapsulates information about an analysis-phase error which would have occurred during a build.
 */
public class AnalysisFailure implements AnalysisFailureApi {

  private final Label label;
  private final String message;

  public AnalysisFailure(
      Label label,
      String message) {
    this.label = label;
    this.message = message;
  }

  @Override
  public Label getLabel() {
    return label;
  }

  @Override
  public String getMessage() {
    return message;
  }

  @Override
  public void repr(SkylarkPrinter printer) {
    printer.append("<AnalyisFailure object>");
  }

  @Override
  public String toString() {
    return "AnalysisFailure(" + label + ", " + message + ")";
  }
}
