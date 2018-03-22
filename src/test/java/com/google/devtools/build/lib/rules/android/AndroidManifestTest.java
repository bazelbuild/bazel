// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules.android;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;


import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import java.io.IOException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests the AndroidManifest class */
@RunWith(JUnit4.class)
public class AndroidManifestTest extends BuildViewTestCase {
  private static final String DEFAULT_PATH = "prefix/java/com/google/foo/bar";
  private static final String DEFAULT_PACKAGE = "com.google.foo.bar";

  @Test
  public void testGetDefaultPackage() throws Exception {
    RuleContext ctx = makeContext();
    assertThat(AndroidManifest.getDefaultPackage(ctx)).isEqualTo(DEFAULT_PACKAGE);
    ctx.assertNoErrors();
  }

  @Test
  public void testGetDefaultPackage_NoJavaDir() throws Exception {
    RuleContext ctx = makeContext("notjava/com/google/foo/bar");
    AndroidManifest.getDefaultPackage(ctx);
    assertThrows(RuleErrorException.class, ctx::assertNoErrors);
  }

  @Test
  public void testIsDummy() throws Exception {
    RuleContext ctx = makeContext();
    assertThat(AndroidManifest.of(ctx, /* manifest = */ null, DEFAULT_PACKAGE).isDummy()).isTrue();
    assertThat(AndroidManifest.of(ctx, ctx.createOutputArtifact(), DEFAULT_PACKAGE).isDummy())
        .isFalse();
  }

  @Test
  public void testStampAndMergeWith_NoDeps() throws Exception {
    AndroidManifest manifest = AndroidManifest.empty(makeContext());

    StampedAndroidManifest stamped = manifest.stampAndMergeWith(ImmutableList.of());
    assertThat(stamped).isNotEqualTo(manifest);
    assertThat(stamped.getPackage()).isEqualTo(DEFAULT_PACKAGE);
    assertThat(stamped.getManifest()).isNotNull();
    assertThat(stamped.isDummy()).isTrue();

    // The merge should still do a stamp, so future stamping should be a no-op.
    assertThat(stamped).isSameAs(stamped.stamp());
  }

  @Test
  public void testStampAndMergeWith_NoProviders() throws Exception {
    AndroidManifest manifest = AndroidManifest.empty(makeContext());

    AndroidManifest merged = manifest.stampAndMergeWith(getDeps("[]"));

    assertThat(merged.getManifest()).isNotNull();
    assertThat(merged.isDummy()).isTrue();
    assertThat(merged.getPackage()).isEqualTo(DEFAULT_PACKAGE);

    // The merge should still do a stamp, so future stamping should be a no-op.
    assertThat(merged).isSameAs(merged.stamp());
  }

  @Test
  public void testStampAndMergeWith_DummyProviders() throws Exception {
    AndroidManifest manifest = AndroidManifest.empty(makeContext());

    AndroidManifest merged =
        manifest.stampAndMergeWith(
            getDeps("[AndroidManifestInfo(manifest=manifest, package='com.pkg', is_dummy=True)]"));

    assertThat(merged.getManifest()).isNotNull();
    assertThat(merged.isDummy()).isTrue();
    assertThat(merged.getPackage()).isEqualTo(DEFAULT_PACKAGE);

    // The merge should still do a stamp, so future stamping should be a no-op.
    assertThat(merged).isSameAs(merged.stamp());
  }

  @Test
  public void testStampAndMergeWith() throws Exception {
    AndroidManifest manifest = AndroidManifest.empty(makeContext());

    AndroidManifest merged =
        manifest.stampAndMergeWith(
            getDeps("[AndroidManifestInfo(manifest=manifest, package='com.pkg', is_dummy=False)]"));

    // Merging results in a new, non-dummy manifest with the same package
    assertThat(merged).isNotEqualTo(manifest);
    assertThat(merged.getManifest()).isNotNull();
    assertThat(merged.getPackage()).isEqualTo(manifest.getPackage());
    assertThat(merged.isDummy()).isFalse();

    // Merging should implicitly stamp
    assertThat(merged).isSameAs(merged.stamp());
  }

  private ImmutableList<ConfiguredTarget> getDeps(String returnList)
      throws IOException, LabelSyntaxException {
    scratch.file(
        "skylark/rule.bzl",
        "def impl(ctx):",
        "  manifest = ctx.actions.declare_file(ctx.attr.name + 'AndroidManifest.xml')",
        "  ctx.actions.write(manifest, 'some values')",
        "  return " + returnList,
        "test_rule = rule(implementation=impl)");

    scratch.file(
        "skylark/BUILD",
        "load(':rule.bzl', 'test_rule')",
        "test_rule(name='dep1')",
        "test_rule(name='dep2')",
        "test_rule(name='dep3')");

    return ImmutableList.of(
        getConfiguredTarget("//skylark:dep1"),
        getConfiguredTarget("//skylark:dep2"),
        getConfiguredTarget("//skylark:dep3"));
  }

  private RuleContext makeContext() throws Exception {
    return makeContext(DEFAULT_PATH);
  }

  private RuleContext makeContext(String pkg) throws Exception {
    // Use BuildView's getRuleContextForTesting method, rather than BuildViewTestCase's
    // getRuleContext method, to avoid the StubEventHandler used by the latter, which prevents us
    // from declaring new artifacts within code called from the test.
    return view.getRuleContextForTesting(
        scratchConfiguredTarget(pkg, "lib", "android_library(name = 'lib')"),
        new StoredEventHandler(),
        masterConfig);
  }
}
