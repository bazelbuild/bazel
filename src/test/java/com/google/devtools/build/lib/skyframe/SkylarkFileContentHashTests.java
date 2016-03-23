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
package com.google.devtools.build.lib.skyframe;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.packages.ConstantRuleVisibility;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.skyframe.util.SkyframeExecutorTestUtils;
import com.google.devtools.build.lib.util.BlazeClock;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyKey;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.Collection;
import java.util.UUID;

/**
 * Tests for the hash code calculated for Skylark RuleClasses based on the transitive closure
 * of the imports of their respective definition SkylarkEnvironments.
 */
@RunWith(JUnit4.class)
public class SkylarkFileContentHashTests extends BuildViewTestCase {

  @Before
  public final void createFiles() throws Exception  {
    scratch.file("foo/BUILD");
    scratch.file("bar/BUILD");
    scratch.file("helper/BUILD");

    scratch.file("helper/ext.bzl", "def rule_impl(ctx):", "  return None");

    scratch.file(
        "foo/ext.bzl",
        "load('/helper/ext', 'rule_impl')",
        "",
        "foo1 = rule(implementation = rule_impl)",
        "foo2 = rule(implementation = rule_impl)");

    scratch.file(
        "bar/ext.bzl",
        "load('/helper/ext', 'rule_impl')",
        "",
        "bar1 = rule(implementation = rule_impl)");

    scratch.file(
        "pkg/BUILD",
        "load('/foo/ext', 'foo1')",
        "load('/foo/ext', 'foo2')",
        "load('/bar/ext', 'bar1')",
        "",
        "foo1(name = 'foo1')",
        "foo2(name = 'foo2')",
        "bar1(name = 'bar1')");
  }

  @Test
  public void testHashInvariance() throws Exception {
    assertEquals(getHash("pkg", "foo1"), getHash("pkg", "foo1"));
  }

  @Test
  public void testHashInvarianceAfterOverwritingFileWithSameContents() throws Exception {
    String bar1 = getHash("pkg", "bar1");
    scratch.overwriteFile(
        "bar/ext.bzl",
        "load('/helper/ext', 'rule_impl')",
        "",
        "bar1 = rule(implementation = rule_impl)");
    invalidatePackages();
    assertEquals(bar1, getHash("pkg", "bar1"));
  }

  @Test
  public void testHashSameForRulesDefinedInSameFile() throws Exception {
    assertEquals(getHash("pkg", "foo1"), getHash("pkg", "foo2"));
  }

  @Test
  public void testHashNotSameForRulesDefinedInDifferentFiles() throws Exception {
    assertNotEquals(getHash("pkg", "foo1"), getHash("pkg", "bar1"));
  }

  @Test
  public void testImmediateFileChangeChangesHash() throws Exception {
    String bar1 = getHash("pkg", "bar1");
    scratch.overwriteFile(
        "bar/ext.bzl",
        "load('/helper/ext', 'rule_impl')",
        "# Some comments to change file hash",
        "",
        "bar1 = rule(implementation = rule_impl)");
    invalidatePackages();
    assertNotEquals(bar1, getHash("pkg", "bar1"));
  }

  @Test
  public void testTransitiveFileChangeChangesHash() throws Exception {
    String bar1 = getHash("pkg", "bar1");
    String foo1 = getHash("pkg", "foo1");
    String foo2 = getHash("pkg", "foo2");
    scratch.overwriteFile(
        "helper/ext.bzl",
        "# Some comments to change file hash",
        "def rule_impl(ctx):",
        "  return None");
    invalidatePackages();
    assertNotEquals(bar1, getHash("pkg", "bar1"));
    assertNotEquals(foo1, getHash("pkg", "foo1"));
    assertNotEquals(foo2, getHash("pkg", "foo2"));
  }

  @Test
  public void testFileChangeDoesNotAffectRulesDefinedOutsideOfTransitiveClosure() throws Exception {
    String foo1 = getHash("pkg", "foo1");
    String foo2 = getHash("pkg", "foo2");
    scratch.overwriteFile(
        "bar/ext.bzl",
        "load('/helper/ext', 'rule_impl')",
        "# Some comments to change file hash",
        "",
        "bar1 = rule(implementation = rule_impl)");
    invalidatePackages();
    assertEquals(foo1, getHash("pkg", "foo1"));
    assertEquals(foo2, getHash("pkg", "foo2"));
  }

  private void assertNotEquals(String hash, String hash2) {
    assertFalse(hash.equals(hash2));
  }

  /**
   * Returns the hash code of the rule target defined by the pkg and the target name parameters.
   * Asserts that the targets and it's Skylark dependencies were loaded properly.
   */
  private String getHash(String pkg, String name) throws Exception {
    getSkyframeExecutor()
        .preparePackageLoading(
            new PathPackageLocator(outputBase, ImmutableList.of(rootDirectory)),
            ConstantRuleVisibility.PUBLIC,
            true,
            7,
            "",
            UUID.randomUUID(),
            new TimestampGranularityMonitor(BlazeClock.instance()));
    SkyKey pkgLookupKey = PackageValue.key(PackageIdentifier.parse("@//" + pkg));
    EvaluationResult<PackageValue> result =
        SkyframeExecutorTestUtils.evaluate(
            getSkyframeExecutor(), pkgLookupKey, /*keepGoing=*/ false, reporter);
    assertFalse(result.hasError());
    Collection<Target> targets = result.get(pkgLookupKey).getPackage().getTargets();
    for (Target target : targets) {
      if (target.getName().equals(name)) {
        return ((Rule) target)
            .getRuleClassObject()
            .getRuleDefinitionEnvironment()
            .getTransitiveContentHashCode();
      }
    }
    throw new IllegalStateException("target not found: " + name);
  }
}
