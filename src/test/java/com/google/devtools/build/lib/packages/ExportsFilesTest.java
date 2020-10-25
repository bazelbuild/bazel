// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.packages;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.base.Joiner;
import com.google.devtools.build.lib.packages.util.PackageLoadingTestCase;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** A test for the {@code exports_files} function. */
@RunWith(JUnit4.class)
public class ExportsFilesTest extends PackageLoadingTestCase {

  private Package pkg() throws Exception {
    scratch.file("pkg/BUILD", "exports_files(['foo.txt', 'bar.txt'])");
    return getTarget("//pkg:BUILD").getPackage();
  }

  @Test
  public void testExportsFilesRegistersFilesWithPackage() throws Exception {
    List<String> names = getFileNamesOf(pkg());
    String expected = "//pkg:BUILD //pkg:bar.txt //pkg:foo.txt";
    assertThat(Joiner.on(' ').join(names)).isEqualTo(expected);
  }

  /**
   * Returns the names of the input files that are known to pkg.
   */
  private static List<String> getFileNamesOf(Package pkg) {
    List<String> names = new ArrayList<>();
    for (FileTarget target : pkg.getTargets(FileTarget.class)) {
      names.add(target.getLabel().toString());
    }
    Collections.sort(names);
    return names;
  }

  @Test
  public void testFileThatsNotRegisteredYieldsUnknownTargetException() throws Exception {
    NoSuchTargetException e =
        assertThrows(NoSuchTargetException.class, () -> pkg().getTarget("baz.txt"));
    assertThat(e)
        .hasMessageThat()
        .isEqualTo(
            "no such target '//pkg:baz.txt':"
                + " target 'baz.txt' not declared in package 'pkg' (did you mean 'bar.txt'?)"
                + " defined by /workspace/pkg/BUILD");
  }

  @Test
  public void testRegisteredFilesAreRetrievable() throws Exception {
    Package pkg = pkg();
    assertThat(pkg.getTarget("foo.txt").getName()).isEqualTo("foo.txt");
    assertThat(pkg.getTarget("bar.txt").getName()).isEqualTo("bar.txt");
  }

  @Test
  public void testExportsFilesAndRuleNameConflict() throws Exception {
    reporter.removeHandler(failFastHandler);
    scratch.file(
        "pkg2/BUILD",
        "exports_files(['foo'])",
        "genrule(name = 'foo', srcs = ['bar'], outs = [], cmd = '/bin/true')");
    assertThat(getTarget("//pkg2:foo")).isInstanceOf(InputFile.class);
    assertContainsEvent("rule 'foo' in package 'pkg2' conflicts with existing source file");
  }
}
