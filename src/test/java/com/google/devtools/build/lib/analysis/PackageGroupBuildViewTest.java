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
package com.google.devtools.build.lib.analysis;

import com.google.devtools.build.lib.analysis.configuredtargets.PackageGroupConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link PackageGroupConfiguredTarget}.
 */
@RunWith(JUnit4.class)
public final class PackageGroupBuildViewTest extends BuildViewTestCase {
  /** Regression test for bug #3445835. */
  @Test
  public void testPackageGroupInDeps() throws Exception {
    checkError("foo", "bar", "in deps attribute of cc_library rule //foo:bar: "
        + "package group '//foo:foo' is misplaced here ",
        "package_group(name = 'foo', packages = ['//none'])",
        "cc_library(name = 'bar', deps = [':foo'])");
  }

  @Test
  public void testPackageGroupInData() throws Exception {
    checkError(
        "foo",
        "bar",
        "in data attribute of cc_library rule //foo:bar: "
            + "package group '//foo:foo' is misplaced here ",
        "package_group(name = 'foo', packages = ['//none'])",
        "cc_library(name = 'bar', data = [':foo'])");
  }
}
