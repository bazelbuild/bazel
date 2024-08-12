// Copyright 2016 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for deprecation warnings in Bazel. */
@RunWith(JUnit4.class)
public final class DeprecationValidatorTest extends BuildViewTestCase {

  @Test
  public void noDeprecationWarningForTopLevelTarget() throws Exception {
    scratchConfiguredTarget(
        "a",
        "a",
        "filegroup(name='a', deprecation='ignored because this target is on the top level')");
    assertNoEvents();
  }

  @Test
  public void noDeprecationWarningWithinPackage() throws Exception {
    scratchConfiguredTarget(
        "a",
        "a",
        "filegroup(name='a', srcs=[':b'])",
        "filegroup(name='b', deprecation='ignored because depending target is in same package')");
    assertNoEvents();
  }

  @Test
  public void noDeprecationWarningForDeprecatedTarget() throws Exception {
    scratchConfiguredTarget(
        "b",
        "b",
        "filegroup(name='b', deprecation='ignored because depending target is deprecated')");
    scratchConfiguredTarget(
        "a",
        "a",
        "filegroup(name='a', srcs=['//b:b'], deprecation='ignored for a top level target')");
    assertNoEvents();
  }

  @Test
  public void deprecationWarningForJavaCompanionOfJavatestsPackage() throws Exception {
    scratchConfiguredTarget(
        "javatests/a",
        "b",
        "filegroup(name='b', deprecation='deprecation warning printed', testonly=0)");
    checkWarning(
        "java/a",
        "a",
        "target '//java/a:a' depends on deprecated target '//javatests/a:b': "
            + "deprecation warning printed",
        "filegroup(name='a', srcs=['//javatests/a:b'])");
  }

  @Test
  public void deprecationWarningForDifferentPackage() throws Exception {
    scratchConfiguredTarget(
        "b", "b", "filegroup(name='b', deprecation='deprecation warning printed')");
    checkWarning(
        "a",
        "a",
        "target '//a:a' depends on deprecated target '//b:b': deprecation warning printed",
        "filegroup(name='a', srcs=['//b:b'])");
  }

  @Test
  public void deprecationWarningForSamePackageInDifferentRepository() throws Exception {
    scratch.appendFile("MODULE.bazel",
        "bazel_dep(name = 'r')",
        "local_path_override(module_name = 'r', path = '/r')");
    scratch.file("/r/MODULE.bazel", "module(name = 'r')");
    scratch.file("/r/a/BUILD", "filegroup(name='b', deprecation='deprecation warning printed')");
    invalidatePackages();
    checkWarning(
        "a",
        "a",
        "target '//a:a' depends on deprecated target '@@r+//a:b': deprecation warning printed",
        "filegroup(name='a', srcs=['@r//a:b'])");
  }

  @Test
  public void deprecationWarningForJavatestsCompanionOfJavaPackageInDifferentRepository()
      throws Exception {
    scratch.appendFile("MODULE.bazel",
        "bazel_dep(name = 'r')",
        "local_path_override(module_name = 'r', path = '/r')");
    scratch.file("/r/MODULE.bazel", "module(name = 'r')");
    scratch.file(
        "/r/java/a/BUILD", "filegroup(name='b', deprecation='deprecation warning printed')");
    invalidatePackages();
    checkWarning(
        "javatests/a",
        "a",
        "target '//javatests/a:a' depends on deprecated target '@@r+//java/a:b': "
            + "deprecation warning printed",
        "filegroup(name='a', srcs=['@r//java/a:b'])");
  }
}
