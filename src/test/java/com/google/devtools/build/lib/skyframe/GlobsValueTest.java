// Copyright 2023 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableSet;
import com.google.common.testing.EqualsTester;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.packages.Globber.Operation;
import com.google.devtools.build.lib.skyframe.GlobsValue.GlobRequest;
import com.google.devtools.build.lib.skyframe.serialization.testutils.FsUtils;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class GlobsValueTest {

  @Test
  public void testSerialization() throws Exception {
    PackageIdentifier packageId = PackageIdentifier.create("foo", PathFragment.create("//bar"));
    Root packageRoot = Root.fromPath(FsUtils.TEST_FILESYSTEM.getPath("/packageRoot"));

    GlobRequest globRequest1 = GlobRequest.create("*", Operation.FILES_AND_DIRS);
    GlobRequest globRequest2 = GlobRequest.create("foo/**", Operation.SUBPACKAGES);
    GlobRequest globRequest3 = GlobRequest.create("**/*", Operation.FILES);

    SerializationTester serializationTester =
        new SerializationTester(
                GlobsValue.key(packageId, packageRoot, ImmutableSet.of(globRequest1, globRequest2)),
                GlobsValue.key(packageId, packageRoot, ImmutableSet.of(globRequest2, globRequest3)))
            .setVerificationFunction(GlobsValueTest::verifyEquivalent);
    FsUtils.addDependencies(serializationTester);
    serializationTester.runTests();
  }

  private static void verifyEquivalent(GlobsValue.Key orig, GlobsValue.Key deserialized) {
    assertThat(deserialized).isSameInstanceAs(orig);
  }

  @Test
  public void testPrintingDeterministic() throws Exception {
    PackageIdentifier packageId = PackageIdentifier.create("foo", PathFragment.create("//bar"));
    Root packageRoot = Root.fromPath(FsUtils.TEST_FILESYSTEM.getPath("/packageRoot"));

    GlobRequest globRequest1 = GlobRequest.create("*", Operation.FILES_AND_DIRS);
    GlobRequest globRequest2 = GlobRequest.create("foo/**", Operation.SUBPACKAGES);
    GlobRequest globRequest3 = GlobRequest.create("**/*", Operation.FILES);

    GlobsValue.Key key1 =
        GlobsValue.key(
            packageId, packageRoot, ImmutableSet.of(globRequest1, globRequest2, globRequest3));
    GlobsValue.Key key2 =
        GlobsValue.key(
            packageId, packageRoot, ImmutableSet.of(globRequest1, globRequest3, globRequest2));
    GlobsValue.Key key3 =
        GlobsValue.key(
            packageId, packageRoot, ImmutableSet.of(globRequest2, globRequest1, globRequest3));
    GlobsValue.Key key4 =
        GlobsValue.key(
            packageId, packageRoot, ImmutableSet.of(globRequest2, globRequest3, globRequest1));
    GlobsValue.Key key5 =
        GlobsValue.key(
            packageId, packageRoot, ImmutableSet.of(globRequest3, globRequest1, globRequest2));
    GlobsValue.Key key6 =
        GlobsValue.key(
            packageId, packageRoot, ImmutableSet.of(globRequest3, globRequest2, globRequest1));
    new EqualsTester()
        .addEqualityGroup(
            key1.toString(),
            key2.toString(),
            key3.toString(),
            key4.toString(),
            key5.toString(),
            key6.toString(),
            "<GlobsKey packageRoot = /packageRoot, packageIdentifier = @@foo///bar,"
                + " globRequests = [GlobRequest: * FILES_AND_DIRS,GlobRequest: **/* FILES,"
                + "GlobRequest: foo/** SUBPACKAGES]>")
        .testEquals();
  }
}
