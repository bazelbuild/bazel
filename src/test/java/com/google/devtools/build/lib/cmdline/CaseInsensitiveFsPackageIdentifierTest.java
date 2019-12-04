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
public final class CaseInsensitiveFsPackageIdentifierTest {

  @Test
  public void testSemantics() throws Exception {
    PathFragment pf1 = PathFragment.create("foo/bar");
    PathFragment pf2 = PathFragment.create("FoO/BaR");
    assertThat(pf1).isEqualTo(pf2);
    // Fortunately, PathFragments are not interned so we get different instances. The ability to
    // create equal, but not-same instances allows storing them in wrapper objects with their
    // toString() value, thus distinguish differently cased versions of otherwise equal paths.
    assertThat(pf1).isNotSameInstanceAs(pf2);
    assertThat(pf1.toString()).isNotEqualTo(pf2.toString());

    RepositoryName rn1 = RepositoryName.create("@foo");
    RepositoryName rn2 = RepositoryName.create("@FoO");
    // Fortunately, RepositoryNames are not interned so we get different instances. The ability to
    // create equal, but not-same instances allows storing them in wrapper objects with their
    // toString() value, thus distinguish differently cased versions of otherwise equal paths.
    assertThat(rn1).isEqualTo(rn2);
    assertThat(rn1).isNotSameInstanceAs(rn2);
    assertThat(rn1.toString()).isNotEqualTo(rn2.toString());

    PackageIdentifier id1 = PackageIdentifier.parse("@foo//bar/baz");
    PackageIdentifier id2 = PackageIdentifier.parse("@foo//BAR/baz");
    PackageIdentifier id3 = PackageIdentifier.parse("@FOO//bar/baz");
    PackageIdentifier id4 = PackageIdentifier.parse("@FOO//BAR/baz");
    // On a case-insensitive filesystem, RepositoryName ("foo" and "FOO") and PathFragment
    // ("bar/baz" and "BAR/baz") are compared case-insensitively, so all PackageIdentifiers are
    // equal.
    new EqualsTester().addEqualityGroup(id1, id2, id3, id4).testEquals();
    // PackageIdentifier.create() interns objects, and it returns CaseTrustingPackageIdentifier
    // because label case checking is disnabled, so the interned objects are all the same instance.
    assertThat(id1).isSameInstanceAs(id2);
    assertThat(id1).isSameInstanceAs(id3);
    assertThat(id1).isSameInstanceAs(id4);
  }
}
