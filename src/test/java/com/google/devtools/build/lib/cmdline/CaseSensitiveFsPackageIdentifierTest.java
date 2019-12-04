// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.cmdline;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.testing.EqualsTester;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Unit tests for {@link PackageIdentifier}.
 */
@RunWith(JUnit4.class)
public final class CaseSensitiveFsPackageIdentifierTest {

  @Test
  public void testSemantics() throws Exception {
    PathFragment pf1 = PathFragment.create("foo/bar");
    PathFragment pf2 = PathFragment.create("FoO/BaR");
    assertThat(pf1).isNotEqualTo(pf2);

    RepositoryName rn1 = RepositoryName.create("@foo");
    RepositoryName rn2 = RepositoryName.create("@FoO");
    assertThat(rn1).isNotEqualTo(rn2);

    PackageIdentifier id1 = PackageIdentifier.parse("@foo//bar/baz");
    PackageIdentifier id2 = PackageIdentifier.parse("@foo//BAR/baz");
    PackageIdentifier id3 = PackageIdentifier.parse("@FOO//bar/baz");
    PackageIdentifier id4 = PackageIdentifier.parse("@FOO//BAR/baz");
    new EqualsTester()
        .addEqualityGroup(id1)
        .addEqualityGroup(id2)
        .addEqualityGroup(id3)
        .addEqualityGroup(id4)
        .testEquals();
    assertThat(id1).isNotSameInstanceAs(id2);
    assertThat(id1).isNotSameInstanceAs(id3);
    assertThat(id1).isNotSameInstanceAs(id4);
    assertThat(id2).isNotSameInstanceAs(id3);
    assertThat(id2).isNotSameInstanceAs(id4);
    assertThat(id3).isNotSameInstanceAs(id4);
  }
}
