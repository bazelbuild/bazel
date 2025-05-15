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
package com.google.devtools.build.lib.buildtool;

import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.analysis.ViewCreationFailedException;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Integration test for package groups and visibility. */
@RunWith(JUnit4.class)
public class PackageGroupIntegrationTest extends BuildIntegrationTestCase {
  private static final String LOAD_FOO_LIBRARY =
      "load('//test_defs:foo_library.bzl', 'foo_library')";

  @Before
  public final void setUpToolsConfigMock() throws Exception {
    AnalysisMock.get().pySupport().setup(mockToolsConfig);
  }

  @Test
  public void testSimpleDeny() throws Exception {
    write("z/BUILD", "package_group(name='bs', packages=['//z/c'])");
    write("z/a/BUILD", LOAD_FOO_LIBRARY, "foo_library(name='a', visibility=['//z:bs'])");
    write("z/b/BUILD", LOAD_FOO_LIBRARY, "foo_library(name='b', deps=['//z/a:a'])");
    assertThrows(ViewCreationFailedException.class, () -> buildTarget("//z/b:b"));
  }

  @Test
  public void testSimpleAllow() throws Exception {
    write("z/BUILD", "package_group(name='bs', packages=['//z/b'])");
    write("z/a/BUILD", LOAD_FOO_LIBRARY, "foo_library(name='a', visibility=['//z:bs'])");
    write("z/b/BUILD", LOAD_FOO_LIBRARY, "foo_library(name='b', deps=['//z/a:a'])");
    buildTarget("//z/b:b");
  }

  @Test
  public void testNoticesPackageGroupChangedToOk() throws Exception {
    write("z/BUILD", "package_group(name='bs', packages=['//z/c'])");
    write("z/a/BUILD", LOAD_FOO_LIBRARY, "foo_library(name='a', visibility=['//z:bs'])");
    write("z/b/BUILD", LOAD_FOO_LIBRARY, "foo_library(name='b', deps=['//z/a:a'])");
    assertThrows(ViewCreationFailedException.class, () -> buildTarget("//z/b:b"));

    waitForTimestampGranularity();

    write("z/BUILD", "package_group(name='bs', packages=['//z/b'])");
    buildTarget("//z/b:b");
  }

  @Test
  public void testNoticesPackageGroupChangedToBad() throws Exception {
    write("z/BUILD", "package_group(name='bs', packages=['//z/b'])");
    write("z/a/BUILD", LOAD_FOO_LIBRARY, "foo_library(name='a', visibility=['//z:bs'])");
    write("z/b/BUILD", LOAD_FOO_LIBRARY, "foo_library(name='b', deps=['//z/a:a'])");
    buildTarget("//z/b:b");

    waitForTimestampGranularity();

    write("z/BUILD", "package_group(name='bs', packages=['//z/c'])");
    assertThrows(ViewCreationFailedException.class, () -> buildTarget("//z/b:b"));
  }

  @Test
  public void testNoticesChangeInDefaultVisibility() throws Exception {
    write("z/BUILD", "package_group(name='bs', packages=['//z/c'])");
    write(
        "z/a/BUILD",
        String.format(
            """
            %s
            package(default_visibility = ["//z:bs"])

            foo_library(name = "a")
            """,
            LOAD_FOO_LIBRARY));
    write("z/b/BUILD", LOAD_FOO_LIBRARY, "foo_library(name='b', deps=['//z/a:a'])");
    assertThrows(ViewCreationFailedException.class, () -> buildTarget("//z/b:b"));

    waitForTimestampGranularity();

    write("z/BUILD", "package_group(name='bs', packages=['//z/b'])");
    buildTarget("//z/b:b");
  }

  // Regression test for bug #16303057: Building a package_group directly results in NPE
  @Test
  public void testPackageGroupBuildDirectly() throws Exception {
    write("npe/BUILD", "package_group(name = 'npe', packages = ['//npe'])");
    buildTarget("//npe");
  }
}
