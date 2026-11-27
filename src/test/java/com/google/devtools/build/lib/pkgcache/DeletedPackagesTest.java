// Copyright 2026 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.pkgcache;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.pkgcache.PackageOptions.CommaSeparatedPackagePatternListConverter;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link DeletedPackages} and {@link CommaSeparatedPackagePatternListConverter}. */
@RunWith(JUnit4.class)
public class DeletedPackagesTest {

  private static final CommaSeparatedPackagePatternListConverter CONVERTER =
      new CommaSeparatedPackagePatternListConverter();

  private static DeletedPackages parse(String input) throws Exception {
    return DeletedPackages.fromPatterns(CONVERTER.convert(input));
  }

  private static boolean deleted(DeletedPackages d, String pkg) throws Exception {
    return d.matches(PackageIdentifier.parse(pkg));
  }

  @Test
  public void exactPattern() throws Exception {
    DeletedPackages d = parse("foo/bar");
    assertThat(deleted(d, "//foo/bar")).isTrue();
    assertThat(deleted(d, "//foo")).isFalse();
    assertThat(deleted(d, "//foo/bar/baz")).isFalse();
  }

  @Test
  public void subtreePattern() throws Exception {
    DeletedPackages d = parse("foo/bar/...");
    assertThat(deleted(d, "//foo/bar")).isTrue();
    assertThat(deleted(d, "//foo/bar/baz/qux")).isTrue();
    assertThat(deleted(d, "//foo")).isFalse();
    assertThat(deleted(d, "//foo/barbaz")).isFalse();
    assertThat(deleted(d, "@repo//foo/bar")).isFalse();
  }

  @Test
  public void subtreePatternAtRoot() throws Exception {
    for (String input : new String[] {"...", "//..."}) {
      DeletedPackages d = parse(input);
      assertThat(deleted(d, "//")).isTrue();
      assertThat(deleted(d, "//anything/below")).isTrue();
      assertThat(deleted(d, "@repo//anything")).isFalse();
    }
  }

  @Test
  public void subtreePatternInExternalRepo() throws Exception {
    assertThat(deleted(parse("@repo//foo/..."), "@repo//foo/bar")).isTrue();
    assertThat(deleted(parse("@repo//foo/..."), "//foo/bar")).isFalse();
    assertThat(deleted(parse("@repo//..."), "@repo//foo")).isTrue();
  }

  @Test
  public void mixedPatterns() throws Exception {
    DeletedPackages d = parse("foo,bar/...,@repo//baz");
    assertThat(d.exact())
        .containsExactly(PackageIdentifier.parse("//foo"), PackageIdentifier.parse("@repo//baz"));
    assertThat(d.subtrees()).containsExactly(PackageIdentifier.parse("//bar"));
  }

  @Test
  public void packageOptions() throws Exception {
    OptionsParser parser = OptionsParser.builder().optionsClasses(PackageOptions.class).build();
    parser.parse("--deleted_packages=foo", "--deleted_packages=bar/...");
    DeletedPackages d = parser.getOptions(PackageOptions.class).getDeletedPackages();
    assertThat(deleted(d, "//foo")).isTrue();
    assertThat(deleted(d, "//bar/baz")).isTrue();
  }

  @Test
  public void emptyInput() throws Exception {
    assertThat(CONVERTER.convert("")).isEmpty();
  }

  @Test
  public void invalidPatterns() {
    for (String input : new String[] {"foo:bar", "/...", "a//...", "foo//...", "foo/.../bar"}) {
      assertThrows(input, OptionsParsingException.class, () -> CONVERTER.convert(input));
    }
  }
}
