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

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.pkgcache.PackageOptions.CommaSeparatedPackageNameListConverter;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link DeletedPackages} and {@link CommaSeparatedPackageNameListConverter}. */
@RunWith(JUnit4.class)
public class DeletedPackagesTest {

  private static final CommaSeparatedPackageNameListConverter CONVERTER =
      new CommaSeparatedPackageNameListConverter();

  private static PackageIdentifier mainPkg(String path) {
    return PackageIdentifier.createInMainRepo(path);
  }

  @Test
  public void matches_exactPackage() {
    DeletedPackages d = DeletedPackages.exact(ImmutableSet.of(mainPkg("foo/bar")));
    assertThat(d.matches(mainPkg("foo/bar"))).isTrue();
    assertThat(d.matches(mainPkg("foo"))).isFalse();
    assertThat(d.matches(mainPkg("foo/bar/baz"))).isFalse();
    assertThat(d.matches(mainPkg("foo/baz"))).isFalse();
  }

  @Test
  public void matches_subtree() {
    DeletedPackages d =
        DeletedPackages.of(/* exact= */ ImmutableSet.of(), ImmutableSet.of(mainPkg("foo/bar")));
    assertThat(d.matches(mainPkg("foo/bar"))).isTrue();
    assertThat(d.matches(mainPkg("foo/bar/baz"))).isTrue();
    assertThat(d.matches(mainPkg("foo/bar/baz/qux"))).isTrue();
    assertThat(d.matches(mainPkg("foo"))).isFalse();
    assertThat(d.matches(mainPkg("foo/barbaz"))).isFalse();
    assertThat(d.matches(mainPkg("foo/baz"))).isFalse();
  }

  @Test
  public void matches_subtreeAtRoot() {
    DeletedPackages d =
        DeletedPackages.of(/* exact= */ ImmutableSet.of(), ImmutableSet.of(mainPkg("")));
    assertThat(d.matches(mainPkg(""))).isTrue();
    assertThat(d.matches(mainPkg("anything"))).isTrue();
    assertThat(d.matches(mainPkg("anything/below"))).isTrue();
  }

  @Test
  public void matches_subtreeRespectsRepository() throws Exception {
    PackageIdentifier external = PackageIdentifier.parse("@repo//foo");
    DeletedPackages d =
        DeletedPackages.of(/* exact= */ ImmutableSet.of(), ImmutableSet.of(external));
    assertThat(d.matches(external)).isTrue();
    // Same path in main repo isn't deleted.
    assertThat(d.matches(mainPkg("foo"))).isFalse();
  }

  @Test
  public void converter_exactPackage() throws Exception {
    assertThat(CONVERTER.convert("foo/bar"))
        .containsExactly(new DeletedPackages.Pattern(mainPkg("foo/bar"), false));
  }

  @Test
  public void converter_subtreeWithSlashSuffix() throws Exception {
    assertThat(CONVERTER.convert("foo/bar/..."))
        .containsExactly(new DeletedPackages.Pattern(mainPkg("foo/bar"), true));
  }

  @Test
  public void converter_subtreeAlone() throws Exception {
    assertThat(CONVERTER.convert("..."))
        .containsExactly(new DeletedPackages.Pattern(mainPkg(""), true));
  }

  @Test
  public void converter_subtreeAtMainRepoRoot() throws Exception {
    assertThat(CONVERTER.convert("//..."))
        .containsExactly(new DeletedPackages.Pattern(mainPkg(""), true));
  }

  @Test
  public void converter_subtreeInExternalRepo() throws Exception {
    assertThat(CONVERTER.convert("@repo//foo/..."))
        .containsExactly(new DeletedPackages.Pattern(PackageIdentifier.parse("@repo//foo"), true));
  }

  @Test
  public void converter_subtreeAtExternalRepoRoot() throws Exception {
    assertThat(CONVERTER.convert("@repo//..."))
        .containsExactly(new DeletedPackages.Pattern(PackageIdentifier.parse("@repo//"), true));
  }

  @Test
  public void converter_mixed() throws Exception {
    assertThat(CONVERTER.convert("foo,bar/...,@repo//baz"))
        .containsExactly(
            new DeletedPackages.Pattern(mainPkg("foo"), false),
            new DeletedPackages.Pattern(mainPkg("bar"), true),
            new DeletedPackages.Pattern(PackageIdentifier.parse("@repo//baz"), false))
        .inOrder();
  }

  @Test
  public void converter_emptyInputProducesEmptyList() throws Exception {
    assertThat(CONVERTER.convert("")).isEmpty();
  }

  @Test
  public void converter_invalidPackageThrows() {
    assertThrows(OptionsParsingException.class, () -> CONVERTER.convert("foo:bar"));
  }

  @Test
  public void getDeletedPackages_combinesAllPatterns() throws Exception {
    OptionsParser parser =
        OptionsParser.builder().optionsClasses(PackageOptions.class).build();
    parser.parse("--deleted_packages=foo,bar/...,@repo//baz/...");
    PackageOptions options = parser.getOptions(PackageOptions.class);

    DeletedPackages deleted = options.getDeletedPackages();
    assertThat(deleted.exact()).containsExactly(mainPkg("foo"));
    assertThat(deleted.subtrees())
        .containsExactly(mainPkg("bar"), PackageIdentifier.parse("@repo//baz"));
  }
}
