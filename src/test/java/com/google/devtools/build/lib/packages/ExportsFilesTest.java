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
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import com.google.common.base.Joiner;
import com.google.devtools.build.lib.events.util.EventCollectionApparatus;
import com.google.devtools.build.lib.packages.util.PackageFactoryApparatus;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.vfs.Path;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * A test for the {@code exports_files} function defined in
 * {@link PackageFactory}.
 */
@RunWith(JUnit4.class)
public class ExportsFilesTest {

  private Scratch scratch = new Scratch("/workspace");
  private EventCollectionApparatus events = new EventCollectionApparatus();
  private PackageFactoryApparatus packages = new PackageFactoryApparatus(events.reporter());

  private Package pkg() throws Exception {
    Path buildFile = scratch.file("pkg/BUILD",
                                  "exports_files(['foo.txt', 'bar.txt'])");
    return packages.createPackage("pkg", buildFile);
  }

  @Test
  public void testExportsFilesRegistersFilesWithPackage() throws Exception {
    List<String> names = getFileNamesOf(pkg());
    String expected = "//pkg:BUILD //pkg:bar.txt //pkg:foo.txt";
    assertEquals(expected, Joiner.on(' ').join(names));
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
    try {
      pkg().getTarget("baz.txt");
      fail();
    } catch (NoSuchTargetException e) {
      assertThat(e).hasMessage("no such target '//pkg:baz.txt':"
          + " target 'baz.txt' not declared in package 'pkg' (did you mean 'bar.txt'?)"
          + " defined by /workspace/pkg/BUILD");
    }
  }

  @Test
  public void testRegisteredFilesAreRetrievable() throws Exception {
    Package pkg = pkg();
    assertEquals("foo.txt", pkg.getTarget("foo.txt").getName());
    assertEquals("bar.txt", pkg.getTarget("bar.txt").getName());
  }

  @Test
  public void testExportsFilesAndRuleNameConflict() throws Exception {
    events.setFailFast(false);

    Path buildFile = scratch.file("pkg2/BUILD",
        "exports_files(['foo'])",
        "genrule(name = 'foo', srcs = ['bar'], outs = [],",
        "        cmd = '/bin/true')");
    Package pkg = packages.createPackage("pkg2", buildFile);
    events.assertContainsError("rule 'foo' in package 'pkg2' conflicts with "
                               + "existing source file");
    assertTrue(pkg.getTarget("foo") instanceof InputFile);
  }

}
