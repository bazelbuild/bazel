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
package com.google.devtools.build.lib.rules.android;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.skyframe.BzlLoadValue.keyForBuild;

import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Provider;
import com.google.devtools.build.lib.packages.StarlarkProvider;
import com.google.devtools.build.lib.packages.StructImpl;
import net.starlark.java.eval.Starlark;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@code android_common.resource_source_directory} in {@link AndroidStarlarkCommon}. */
@RunWith(JUnit4.class)
public final class AndroidResourceSourceDirectoryTest extends BuildViewTestCase {

  private Provider.Key infoKey;

  @Before
  public void setUpProbeRule() throws Exception {
    scratch.file(
        "java/android/defs.bzl",
        """
        Info = provider(fields = ["source_dir"])

        def _impl(ctx):
            return [Info(source_dir = android_common.resource_source_directory(ctx.file.src))]

        probe = rule(
            implementation = _impl,
            attrs = {"src": attr.label(allow_single_file = True)},
        )
        """);
    infoKey =
        new StarlarkProvider.Key(
            keyForBuild(Label.parseCanonical("//java/android:defs.bzl")), "Info");
  }

  @Test
  public void resourceSourceDirectory_validResourcePath_returnsSourceDirectory() throws Exception {
    scratch.file("java/android/res/values/strings.xml", "<resources></resources>");
    scratch.file("java/android/res/drawable-hdpi-v4/icon.png", "");
    scratch.file(
        "java/android/BUILD",
        """
        load(":defs.bzl", "probe")

        probe(
            name = "values_res",
            src = "res/values/strings.xml",
        )

        probe(
            name = "qualified_res",
            src = "res/drawable-hdpi-v4/icon.png",
        )
        """);

    ConfiguredTarget valuesTarget = getConfiguredTarget("//java/android:values_res");
    StructImpl valuesInfo = (StructImpl) valuesTarget.get(infoKey);
    assertThat(valuesInfo.getValue("source_dir")).isEqualTo("java/android/res");

    ConfiguredTarget qualifiedTarget = getConfiguredTarget("//java/android:qualified_res");
    StructImpl qualifiedInfo = (StructImpl) qualifiedTarget.get(infoKey);
    assertThat(qualifiedInfo.getValue("source_dir")).isEqualTo("java/android/res");
  }

  @Test
  public void resourceSourceDirectory_nonResourcePath_returnsNone() throws Exception {
    scratch.file("java/android/src/not_a_res_dir/file.txt", "hello");
    scratch.file("short.txt", "hello");
    scratch.file(
        "BUILD",
        """
        exports_files(["short.txt"])
        """);
    scratch.file(
        "java/android/BUILD",
        """
        load(":defs.bzl", "probe")

        probe(
            name = "non_resource_dir",
            src = "src/not_a_res_dir/file.txt",
        )

        probe(
            name = "short_path",
            src = "//:short.txt",
        )
        """);

    ConfiguredTarget nonResourceTarget = getConfiguredTarget("//java/android:non_resource_dir");
    StructImpl nonResourceInfo = (StructImpl) nonResourceTarget.get(infoKey);
    assertThat(nonResourceInfo.getValue("source_dir")).isEqualTo(Starlark.NONE);

    ConfiguredTarget shortPathTarget = getConfiguredTarget("//java/android:short_path");
    StructImpl shortPathInfo = (StructImpl) shortPathTarget.get(infoKey);
    assertThat(shortPathInfo.getValue("source_dir")).isEqualTo(Starlark.NONE);
  }
}
