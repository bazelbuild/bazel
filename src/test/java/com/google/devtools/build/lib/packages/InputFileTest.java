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

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;

import com.google.devtools.build.lib.events.util.EventCollectionApparatus;
import com.google.devtools.build.lib.packages.util.PackageFactoryApparatus;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.vfs.Path;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * A test for {@link InputFile}.
 */
@RunWith(JUnit4.class)
public class InputFileTest {

  private Path pathX;
  private Path pathY;
  private Package pkg;

  private EventCollectionApparatus events = new EventCollectionApparatus();
  private Scratch scratch = new Scratch("/workspace");
  private PackageFactoryApparatus packages = new PackageFactoryApparatus(events.reporter());

  @Before
  public void setUp() throws Exception {
    Path buildfile =
        scratch.file(
            "pkg/BUILD",
            "genrule(name = 'dummy', ",
            "        cmd = '', ",
            "        outs = [], ",
            "        srcs = ['x', 'subdir/y'])");
    pkg = packages.createPackage("pkg", buildfile);
    events.assertNoWarningsOrErrors();

    this.pathX = scratch.file("pkg/x", "blah");
    this.pathY = scratch.file("pkg/subdir/y", "blah blah");
  }

  private static void checkPathMatches(InputFile input, Path expectedPath) {
    assertEquals(expectedPath, input.getPath());
  }

  private static void checkName(InputFile input, String expectedName) {
    assertEquals(expectedName, input.getName());
  }

  private static void checkLabel(InputFile input, String expectedLabelString) {
    assertEquals(expectedLabelString, input.getLabel().toString());
  }

  @Test
  public void testGetAssociatedRule() throws Exception {
    assertNull(null, pkg.getTarget("x").getAssociatedRule());
  }

  @Test
  public void testInputFileInPackageDirectory() throws NoSuchTargetException {
    InputFile inputFileX = (InputFile) pkg.getTarget("x");
    checkPathMatches(inputFileX, pathX);
    checkName(inputFileX, "x");
    checkLabel(inputFileX, "//pkg:x");
    assertEquals("source file", inputFileX.getTargetKind());
  }

  @Test
  public void testInputFileInSubdirectory() throws NoSuchTargetException {
    InputFile inputFileY = (InputFile) pkg.getTarget("subdir/y");
    checkPathMatches(inputFileY, pathY);
    checkName(inputFileY, "subdir/y");
    checkLabel(inputFileY, "//pkg:subdir/y");
  }

  @Test
  public void testEquivalenceRelation() throws NoSuchTargetException {
    InputFile inputFileX = (InputFile) pkg.getTarget("x");
    assertSame(pkg.getTarget("x"), inputFileX);
    InputFile inputFileY = (InputFile) pkg.getTarget("subdir/y");
    assertSame(pkg.getTarget("subdir/y"), inputFileY);
    assertEquals(inputFileX, inputFileX);
    assertFalse(inputFileX.equals(inputFileY));
    assertFalse(inputFileY.equals(inputFileX));
  }
}
