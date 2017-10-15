// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.objc;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.rules.objc.ObjcCommon.BUNDLE_CONTAINER_TYPE;
import static com.google.devtools.build.lib.rules.objc.ObjcCommon.NOT_IN_CONTAINER_ERROR_FORMAT;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.BUNDLE_FILE;

import com.google.common.collect.ImmutableList;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test case for objc_bundle. */
@RunWith(JUnit4.class)
public class ObjcBundleTest extends ObjcRuleTestCase {
  @Test
  public void testErrorForImportArtifactNotInDotBundleDir() throws Exception {
    scratch.file("x/foo/notinbundledir");
    scratch.file("x/bar/x.bundle/isinbundledir");
    checkError("x", "x",
        String.format(NOT_IN_CONTAINER_ERROR_FORMAT,
            "x/foo/notinbundledir",
            ImmutableList.of(BUNDLE_CONTAINER_TYPE)),
        "objc_bundle(",
        "    name = 'x',",
        "    bundle_imports = ['bar/x.bundle/isinbundledir', 'foo/notinbundledir'],",
        ")");
  }

  @Test
  public void testBundleFilesProvided() throws Exception {
    scratch.file("bundle/bar/x.bundle/1");
    scratch.file("bundle/bar/x.bundle/subdir/2");
    scratch.file("bundle/bar/y.bundle/subdir/1");
    scratch.file("bundle/bar/y.bundle/2");
    scratch.file("bundle/BUILD",
        "objc_bundle(",
        "    name = 'bundle',",
        "    bundle_imports = glob(['bar/**']),",
        ")");
    ObjcProvider provider = providerForTarget("//bundle:bundle");
    assertThat(provider.get(BUNDLE_FILE)).containsExactly(
        new BundleableFile(getSourceArtifact("bundle/bar/x.bundle/1"), "x.bundle/1"),
        new BundleableFile(getSourceArtifact("bundle/bar/x.bundle/subdir/2"), "x.bundle/subdir/2"),
        new BundleableFile(getSourceArtifact("bundle/bar/y.bundle/subdir/1"), "y.bundle/subdir/1"),
        new BundleableFile(getSourceArtifact("bundle/bar/y.bundle/2"), "y.bundle/2"));
  }

  @Test
  public void testBundleImportsUsesOuterMostDotBundleDirAsRoot() throws Exception {
    scratch.file("bundle/bar/x.bundle/foo/y.bundle/baz");
    scratch.file("bundle/BUILD",
        "objc_bundle(",
        "    name = 'bundle',",
        "    bundle_imports = glob(['bar/**']),",
        ")");
    ObjcProvider provider = providerForTarget("//bundle:bundle");
    assertThat(provider.get(BUNDLE_FILE))
        .containsExactly(new BundleableFile(
            getSourceArtifact("bundle/bar/x.bundle/foo/y.bundle/baz"), "x.bundle/foo/y.bundle/baz"))
        .inOrder();
  }
}
