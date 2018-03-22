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

import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests that the AndroidManifestInfo can be moved between Native and Skylark code */
@RunWith(JUnit4.class)
public class AndroidManifestInfoTest extends BuildViewTestCase {
  @Test
  public void testGetProvider() throws Exception {
    AndroidManifestInfo info =
        getInfoFromSkylark("manifest=manifest, package='some.pkg', is_dummy=True");

    assertThat(info.getManifest()).isNotNull();
    assertThat(info.getManifest().getFilename()).isEqualTo("AndroidManifest.xml");
    assertThat(info.getPackage()).isEqualTo("some.pkg");
    assertThat(info.isDummy()).isTrue();
  }

  @Test
  public void testGetProvider_isDummyOptional() throws Exception {
    AndroidManifestInfo info = getInfoFromSkylark("manifest=manifest, package='some.pkg'");

    assertThat(info.getManifest()).isNotNull();
    assertThat(info.getManifest().getFilename()).isEqualTo("AndroidManifest.xml");
    assertThat(info.getPackage()).isEqualTo("some.pkg");
    assertThat(info.isDummy()).isFalse();
  }

  @Test
  public void testGetProviderInSkylark() throws Exception {
    scratch.file(
        "java/skylark/test/rule.bzl",
        "def impl(ctx):",
        "  manifest = ctx.actions.declare_file('AndroidManifest.xml')",
        "  ctx.actions.write(manifest, 'some values')",
        "  info = AndroidManifestInfo(manifest=manifest, package='some.pkg', is_dummy=True)",
        // Prove that we can access the fields of the provider within Skylark by making a new
        // provider from the contents of the current one.
        "  return [AndroidManifestInfo(",
        "    manifest=info.manifest, package=info.package, is_dummy=info.is_dummy)]",
        "test_rule = rule(implementation=impl)");

    AndroidManifestInfo copied =
        scratchConfiguredTarget(
                "java/skylark/test",
                "rule",
                "load(':rule.bzl', 'test_rule')",
                "test_rule(name='rule')")
            .get(AndroidManifestInfo.PROVIDER);

    assertThat(copied).isNotNull();
    assertThat(copied.getManifest()).isNotNull();
    assertThat(copied.getManifest().getFilename()).isEqualTo("AndroidManifest.xml");
    assertThat(copied.getPackage()).isEqualTo("some.pkg");
    assertThat(copied.isDummy()).isTrue();
  }

  private AndroidManifestInfo getInfoFromSkylark(String infoArgs) throws Exception {
    scratch.file(
        "java/skylark/test/rule.bzl",
        "def impl(ctx):",
        "  manifest = ctx.actions.declare_file('AndroidManifest.xml')",
        "  ctx.actions.write(manifest, 'some values')",
        "  return [AndroidManifestInfo(" + infoArgs + ")]",
        "test_rule = rule(implementation=impl)");

    AndroidManifestInfo info =
        scratchConfiguredTarget(
                "java/skylark/test",
                "rule",
                "load(':rule.bzl', 'test_rule')",
                "test_rule(name='rule')")
            .get(AndroidManifestInfo.PROVIDER);
    assertThat(info).isNotNull();

    return info;
  }
}
