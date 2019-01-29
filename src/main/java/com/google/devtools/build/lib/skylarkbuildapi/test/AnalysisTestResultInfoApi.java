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

package com.google.devtools.build.lib.skylarkbuildapi.test;

import com.google.devtools.build.lib.skylarkbuildapi.ProviderApi;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkConstructor;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkValue;

/**
 * Encapsulates information about an analysis-phase error which would have occurred during a build.
 */
@SkylarkModule(
    name = "AnalysisTestResultInfo",
    doc = "<b>Experimental. This API is experimental and subject to change at any time</b><p> "
        + "Encapsulates the result of analyis-phase testing. Build targets which return an "
        + "instance of this provider signal to the build system that it should generate a "
        + "'stub' test executable which generates the equivalent test result. Analysis-phase "
        + "('in-build') test rules <b>must</b> return an instance of this provider, and "
        + "non-analysis-phase test rules <b>cannot</b> return this provider.",
    documented = false)
public interface AnalysisTestResultInfoApi extends SkylarkValue {

  @SkylarkCallable(
      name = "success",
      doc =
          "If true, then the analysis-phase test represented by this target passed. If "
              + "false, the test failed.",
      documented = false,
      structField = true)
  public Boolean getSuccess();

  @SkylarkCallable(
      name = "message",
      doc = "A descriptive message containing information about the test and its success/failure.",
      documented = false,
      structField = true)
  public String getMessage();

  /** Provider class for {@link AnalysisTestResultInfoApi} objects. */
  @SkylarkModule(name = "Provider", documented = false, doc = "")
  public interface AnalysisTestResultInfoProviderApi extends ProviderApi {

    @SkylarkCallable(
        name = "AnalysisTestResultInfo",
        doc = "The <code>AnalysisTestResultInfo</code> constructor.",
        parameters = {
          @Param(
              name = "success",
              type = Boolean.class,
              named = true,
              doc =
                  "If true, then the analysis-phase test represented by this target should "
                      + "pass. If false, the test should fail."),
          @Param(
              name = "message",
              type = String.class,
              named = true,
              doc =
                  "A descriptive message containing information about the test and its "
                      + "success/failure.")
        },
        selfCall = true)
    @SkylarkConstructor(
        objectType = AnalysisTestResultInfoApi.class,
        receiverNameForDoc = "AnalysisTestResultInfo")
    public AnalysisTestResultInfoApi testResultInfo(Boolean success, String message);
  }
}
