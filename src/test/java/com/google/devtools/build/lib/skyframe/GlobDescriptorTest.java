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
package com.google.devtools.build.lib.skyframe;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.skyframe.serialization.testutils.FsUtils;
import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link GlobDescriptor}. */
@RunWith(JUnit4.class)
public class GlobDescriptorTest {

  @Test
  public void testSerialization() throws Exception {
    SerializationTester serializationTester =
        new SerializationTester(
                GlobDescriptor.create(
                    PackageIdentifier.create("@foo", PathFragment.create("//bar")),
                    Root.fromPath(FsUtils.TEST_FILESYSTEM.getPath("/packageRoot")),
                    PathFragment.create("subdir"),
                    "pattern",
                    /*excludeDirs=*/ false),
                GlobDescriptor.create(
                    PackageIdentifier.create("@bar", PathFragment.create("//foo")),
                    Root.fromPath(FsUtils.TEST_FILESYSTEM.getPath("/anotherPackageRoot")),
                    PathFragment.create("anotherSubdir"),
                    "pattern",
                    /*excludeDirs=*/ true))
            .setVerificationFunction(GlobDescriptorTest::verifyEquivalent);
    FsUtils.addDependencies(serializationTester);
    serializationTester.runTests();
  }

  private static void verifyEquivalent(GlobDescriptor orig, GlobDescriptor deserialized) {
    assertThat(deserialized).isSameInstanceAs(orig);
  }

  @Test
  public void testCreateReturnsInternedInstances() throws LabelSyntaxException {
    GlobDescriptor original =
        GlobDescriptor.create(
            PackageIdentifier.create("@foo", PathFragment.create("//bar")),
            Root.fromPath(FsUtils.TEST_FILESYSTEM.getPath("/packageRoot")),
            PathFragment.create("subdir"),
            "pattern",
            /*excludeDirs=*/ false);

    GlobDescriptor sameCopy = GlobDescriptor.create(
        original.getPackageId(),
        original.getPackageRoot(),
        original.getSubdir(),
        original.getPattern(),
        original.excludeDirs());
    assertThat(sameCopy).isSameInstanceAs(original);

    GlobDescriptor diffCopy = GlobDescriptor.create(
        original.getPackageId(),
        original.getPackageRoot(),
        original.getSubdir(),
        original.getPattern(),
        !original.excludeDirs());
    assertThat(diffCopy).isNotEqualTo(original);
  }

}
