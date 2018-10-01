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

import com.google.devtools.build.lib.events.Location.LineAndColumn;
import com.google.devtools.build.lib.events.util.EventCollectionApparatus;
import com.google.devtools.build.lib.packages.util.PackageFactoryApparatus;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link Rule}.
 */
@RunWith(JUnit4.class)
public class RuleTest {
  private Scratch scratch = new Scratch("/workspace");
  private EventCollectionApparatus events = new EventCollectionApparatus();
  private PackageFactoryApparatus packages = new PackageFactoryApparatus(events.reporter());
  private Root root;

  @Before
  public void setUp() throws Exception {
    root = Root.fromPath(scratch.dir(""));
  }

  @Test
  public void testAttributeLocation() throws Exception {
    Path buildFile = scratch.file("x/BUILD",
        "cc_binary(name = 'x',",
        "          srcs = ['a', 'b', 'c'],",
        "          defines = ['-Da', '-Db'])");
    Package pkg = packages.createPackage("x", RootedPath.toRootedPath(root, buildFile));
    Rule rule = pkg.getRule("x");

    assertThat(rule.getLocation().getStartLineAndColumn()).isEqualTo(new LineAndColumn(1, 1));

    // Special "name" attribute always has same location as rule:
    assertThat(rule.getAttributeLocation("name").getStartLineAndColumn())
        .isEqualTo(new LineAndColumn(1, 1));

    // User-provided attributes have precise locations:
    assertThat(rule.getAttributeLocation("srcs").getStartLineAndColumn())
        .isEqualTo(new LineAndColumn(2, 18));
    assertThat(rule.getAttributeLocation("defines").getStartLineAndColumn())
        .isEqualTo(new LineAndColumn(3, 21));

    // Default attributes have same location as rule:
    assertThat(rule.getAttributeLocation("malloc").getStartLineAndColumn())
        .isEqualTo(new LineAndColumn(1, 1));

    // Attempts to locate non-existent attributes don't fail;
    // the rule location is returned:
    assertThat(rule.getAttributeLocation("no-such-attr").getStartLineAndColumn())
        .isEqualTo(new LineAndColumn(1, 1));
  }

  @Test
  public void testOutputNameError() throws Exception {
    events.setFailFast(false);
    Path buildFile = scratch.file("namecollide/BUILD",
        "genrule(name = 'hello_world',",
                "srcs = ['ignore_me.txt'],",
                "outs = ['message.txt', 'hello_world'],",
                "cmd  = 'echo \"Hello, world.\" >$(location message.txt)')");

    Package pkg = packages.createPackage("namecollide", RootedPath.toRootedPath(root, buildFile));
    Rule genRule = pkg.getRule("hello_world");
    assertThat(genRule.containsErrors()).isFalse(); // TODO: assertTrue
    events.assertContainsWarning("target 'hello_world' is both a rule and a file; please choose "
                               + "another name for the rule");
  }

  @Test
  public void testIsLocalTestRuleForLocalEquals1() throws Exception {
    Path buildFile = scratch.file("x/BUILD",
        "cc_test(name = 'y',",
        "          srcs = ['a'],",
        "          local = 0)",
        "cc_test(name = 'z',",
        "          srcs = ['a'],",
        "          local = 1)");
    Package pkg = packages.createPackage("x", RootedPath.toRootedPath(root, buildFile));
    Rule y = pkg.getRule("y");
    assertThat(TargetUtils.isLocalTestRule(y)).isFalse();
    Rule z = pkg.getRule("z");
    assertThat(TargetUtils.isLocalTestRule(z)).isTrue();
  }

  @Test
  public void testDeprecation() throws Exception {
    Path buildFile = scratch.file("x/BUILD",
        "cc_test(name = 'y')",
        "cc_test(name = 'z', deprecation = 'Foo')");
    Package pkg = packages.createPackage("x", RootedPath.toRootedPath(root, buildFile));
    Rule y = pkg.getRule("y");
    assertThat(TargetUtils.getDeprecation(y)).isNull();
    Rule z = pkg.getRule("z");
    assertThat(TargetUtils.getDeprecation(z)).isEqualTo("Foo");
  }

  @Test
  public void testVisibilityValid() throws Exception {
    Package pkg =
        packages.createPackage(
            "x",
            RootedPath.toRootedPath(
                root,
                scratch.file(
                    "x/BUILD",
                    "cc_binary(name = 'pr',",
                    "          visibility = ['//visibility:private'])",
                    "cc_binary(name = 'pu',",
                    "          visibility = ['//visibility:public'])",
                    "cc_binary(name = 'cu',",
                    "          visibility = ['//a:b'])")));

    assertThat(pkg.getRule("pu").getVisibility()).isEqualTo(ConstantRuleVisibility.PUBLIC);
    assertThat(pkg.getRule("pr").getVisibility()).isEqualTo(ConstantRuleVisibility.PRIVATE);
  }
}
