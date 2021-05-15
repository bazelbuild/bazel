// Copyright 2020 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis.testing;

import static com.google.common.truth.Truth.assertAbout;

import com.google.common.truth.FailureMetadata;
import com.google.common.truth.Subject;
import com.google.devtools.build.lib.analysis.platform.ToolchainInfo;
import net.starlark.java.eval.EvalException;

/** A Truth {@link Subject} for {@link ToolchainInfo}. */
public class ToolchainInfoSubject extends Subject {
  // Static data.

  /** Entry point for test assertions related to {@link ToolchainInfo}. */
  public static ToolchainInfoSubject assertThat(ToolchainInfo toolchainInfo) {
    return assertAbout(ToolchainInfoSubject::new).that(toolchainInfo);
  }

  /** Static method for getting the subject factory (for use with assertAbout()). */
  public static Factory<ToolchainInfoSubject, ToolchainInfo> toolchainInfos() {
    return ToolchainInfoSubject::new;
  }

  // Instance fields.

  private final ToolchainInfo actual;

  private ToolchainInfoSubject(FailureMetadata failureMetadata, ToolchainInfo subject) {
    super(failureMetadata, subject);
    this.actual = subject;
  }

  public Subject getValue(String name) throws EvalException {
    return check("getValue(%s)", name).that(actual.getValue(name));
  }
}
