// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.objc;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.rules.objc.ObjcCommon.FRAMEWORK_CONTAINER_TYPE;
import static com.google.devtools.build.lib.rules.objc.ObjcCommon.NOT_IN_CONTAINER_ERROR_FORMAT;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.SDK_DYLIB;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.SDK_FRAMEWORK;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.STATIC_FRAMEWORK_DIR;
import static com.google.devtools.build.lib.rules.objc.ObjcProvider.WEAK_SDK_FRAMEWORK;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.xcode.bundlemerge.proto.BundleMergeProtos;
import com.google.devtools.build.xcode.bundlemerge.proto.BundleMergeProtos.BundleFile;
import java.util.Set;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test case for objc_framework. */
@RunWith(JUnit4.class)
public class ObjcFrameworkTest extends ObjcRuleTestCase {
  @Test
  public void testErrorForImportArtifactNotInDotFrameworkDir() throws Exception {
    scratch.file("x/foo/notinframeworkdir");
    scratch.file("x/bar/x.framework/isinframeworkdir");
    checkError("x", "x",
        String.format(NOT_IN_CONTAINER_ERROR_FORMAT,
            "x/foo/notinframeworkdir",
            ImmutableList.of(FRAMEWORK_CONTAINER_TYPE)),
        "objc_framework(",
        "    name = 'x',",
        "    framework_imports = ['bar/x.framework/isinframeworkdir', 'foo/notinframeworkdir'],",
        ")");
  }

  @Test
  public void testProvidesFilesAndDirs_static() throws Exception {
    addBinWithTransitiveDepOnFrameworkImport();
    ObjcProvider provider = providerForTarget("//fx:fx");
    assertThat(provider.get(STATIC_FRAMEWORK_DIR))
        .containsExactly(
            PathFragment.create("fx/fx1.framework"),
            PathFragment.create("fx/fx2.framework"));
    assertThat(provider.get(ObjcProvider.STATIC_FRAMEWORK_FILE))
        .containsExactly(
            getSourceArtifact("fx/fx1.framework/a"),
            getSourceArtifact("fx/fx1.framework/b"),
            getSourceArtifact("fx/fx2.framework/c"),
            getSourceArtifact("fx/fx2.framework/d"));
    assertThat(provider.get(ObjcProvider.DYNAMIC_FRAMEWORK_DIR)).isEmpty();
    assertThat(provider.get(ObjcProvider.DYNAMIC_FRAMEWORK_FILE)).isEmpty();
  }
  
  @Test
  public void testProvidesFilesAndDirs_dynamic() throws Exception {
    scratch.file("fx/fx1.framework/a");
    scratch.file("fx/fx1.framework/b");
    scratch.file("fx/fx2.framework/c");
    scratch.file("fx/fx2.framework/d");
    scratch.file("fx/BUILD",
        "objc_framework(",
        "    name = 'fx',",
        "    framework_imports = glob(['fx1.framework/*', 'fx2.framework/*']),",
        "    is_dynamic = 1,",
        ")");
    createLibraryTargetWriter("//lib:lib")
        .setAndCreateFiles("srcs", "a.m", "b.m", "private.h")
        .setList("deps", "//fx:fx")
        .write();
    
    ObjcProvider provider = providerForTarget("//fx:fx");
    assertThat(provider.get(ObjcProvider.DYNAMIC_FRAMEWORK_DIR))
        .containsExactly(
            PathFragment.create("fx/fx1.framework"),
            PathFragment.create("fx/fx2.framework"));
    assertThat(provider.get(ObjcProvider.DYNAMIC_FRAMEWORK_FILE))
        .containsExactly(
            getSourceArtifact("fx/fx1.framework/a"),
            getSourceArtifact("fx/fx1.framework/b"),
            getSourceArtifact("fx/fx2.framework/c"),
            getSourceArtifact("fx/fx2.framework/d"));
    assertThat(provider.get(ObjcProvider.STATIC_FRAMEWORK_DIR)).isEmpty();
    assertThat(provider.get(ObjcProvider.STATIC_FRAMEWORK_FILE)).isEmpty();
  }

  @Test
  public void testSdkFrameworks_objcProvider() throws Exception {
    ConfiguredTarget configuredTarget = addLibWithDepOnFrameworkImport();
    ObjcProvider provider = providerForTarget(configuredTarget.getLabel().toString());

    Set<SdkFramework> sdkFrameworks = ImmutableSet.of(new SdkFramework("CoreLocation"));

    assertThat(provider.get(SDK_FRAMEWORK)).containsExactlyElementsIn(sdkFrameworks);
  }

  @Test
  public void testWeakSdkFrameworks_objcProvider() throws Exception {
    ConfiguredTarget configuredTarget = addLibWithDepOnFrameworkImport();
    ObjcProvider provider = providerForTarget(configuredTarget.getLabel().toString());

    assertThat(provider.get(WEAK_SDK_FRAMEWORK))
        .containsExactly(new SdkFramework("MediaAccessibility"));
  }

  @Test
  public void testDylibs_objcProvider() throws Exception {
    ConfiguredTarget configuredTarget = addLibWithDepOnFrameworkImport();
    ObjcProvider provider = providerForTarget(configuredTarget.getLabel().toString());

    assertThat(provider.get(SDK_DYLIB)).containsExactly("libdy1");
  }

  @Test
  public void testRequiresNonEmptyFrameworkImports()
      throws Exception {
    scratch.file("x/dir/x.framework/isinframeworkdir");
    checkError("x", "empty_with_configuration",
        getErrorMsgNonEmptyList(
            "framework_imports", "objc_framework", "//x:empty_with_configuration"),
        "objc_framework(",
        "    name = 'empty_with_configuration',",
        "    framework_imports = [],",
        ")");
  }

  // This also serves as a regression test for non-empty attributes with configurable values. Please
  // don't delete this.
  @Test
  public void testRequiresNonEmptyFrameworkImports_Configurable_EmptyWithConfiguration()
      throws Exception {
    scratch.file("x/dir/x.framework/isinframeworkdir");
    useConfiguration("--test_arg=a");
    checkError("x", "empty_with_configuration",
        getErrorMsgNonEmptyList(
            "framework_imports", "objc_framework", "//x:empty_with_configuration"),
        "config_setting(",
        "    name = 'a',",
        "    values = {'test_arg': 'a'},",
        ")",
        "objc_framework(",
        "    name = 'empty_with_configuration',",
        "    framework_imports = select({",
        "        ':a': [],",
        "        '//conditions:default': ['dir/x.framework/isinframeworkdir']",
        "    })",
        ")");
  }

  // This also serves as a regression test for non-empty attributes with configurable values. Please
  // don't delete this.
  @Test
  public void testRequiresNonEmptyFrameworkImports_Configurable_NonEmptyWithConfiguration()
      throws Exception {
    scratch.file("x/dir/x.framework/isinframeworkdir");
    useConfiguration("--test_arg=a");
    scratchConfiguredTarget("x", "empty_with_configuration",
        "config_setting(",
        "    name = 'a',",
        "    values = {'test_arg': 'a'},",
        ")",
        "objc_framework(",
        "    name = 'empty_with_configuration',",
        "    framework_imports = select({",
        "        ':a': ['dir/x.framework/isinframeworkdir'],",
        "        '//conditions:default': []",
        "    })",
        ")");
  }

  // This also serves as a regression test for non-empty attributes with configurable values. Please
  // don't delete this.
  @Test
  public void testRequiresNonEmptyFrameworkImports_Configurable_NonEmptyWithDefault()
      throws Exception {
    scratch.file("x/dir/x.framework/isinframeworkdir");
    scratchConfiguredTarget("x", "empty_with_configuration",
        "config_setting(",
        "    name = 'a',",
        "    values = {'test_arg': 'a'},",
        ")",
        "objc_framework(",
        "    name = 'empty_with_configuration',",
        "    framework_imports = select({",
        "        ':a': [],",
        "        '//conditions:default': ['dir/x.framework/isinframeworkdir']",
        "    })",
        ")");
  }

  // This also serves as a regression test for non-empty attributes with configurable values. Please
  // don't delete this.
  @Test
  public void testRequiresNonEmptyFrameworkImports_Configurable_EmptyWithDefault()
      throws Exception {
    scratch.file("x/dir/x.framework/isinframeworkdir");
    checkError("x", "empty_with_configuration",
        getErrorMsgNonEmptyList(
            "framework_imports", "objc_framework", "//x:empty_with_configuration"),
        "config_setting(",
        "    name = 'a',",
        "    values = {'test_arg': 'a'},",
        ")",
        "objc_framework(",
        "    name = 'empty_with_configuration',",
        "    framework_imports = select({",
        "        ':a': ['dir/x.framework/isinframeworkdir'],",
        "        '//conditions:default': []",
        "    })",
        ")");
  }

  @Test
  public void testDynamicFrameworkInFinalBundle() throws Exception {
    scratch.file("x/Foo.framework/Foo");
    scratch.file("x/Foo.framework/Info.plist");
    scratch.file("x/Foo.framework/Headers/Foo.h");
    scratch.file("x/Foo.framework/Resources/bar.png");
    scratch.file(
        "x/BUILD",
        "objc_framework(",
        "    name = 'foo_framework',",
        "    framework_imports = glob(['Foo.framework/**']),",
        "    is_dynamic = 1,",
        ")",
        "",
        "objc_binary(",
        "    name = 'bin',",
        "    srcs = [ 'a.m' ],",
        "    deps = [ ':foo_framework' ],",
        ")",
        "",
        "ios_application(",
        "    name = 'x',",
        "    binary = ':bin',",
        ")");

    BundleMergeProtos.Control mergeControl = bundleMergeControl("//x:x");

    assertThat(mergeControl.getBundleFileList())
        .containsAllOf(
            BundleFile.newBuilder()
                .setBundlePath("Frameworks/Foo.framework/Foo")
                .setSourceFile(getSourceArtifact("x/Foo.framework/Foo").getExecPathString())
                .setExternalFileAttribute(BundleableFile.EXECUTABLE_EXTERNAL_FILE_ATTRIBUTE)
                .build(),
            BundleFile.newBuilder()
                .setBundlePath("Frameworks/Foo.framework/Info.plist")
                .setSourceFile(getSourceArtifact("x/Foo.framework/Info.plist").getExecPathString())
                .setExternalFileAttribute(BundleableFile.EXECUTABLE_EXTERNAL_FILE_ATTRIBUTE)
                .build(),
            BundleFile.newBuilder()
                .setBundlePath("Frameworks/Foo.framework/Resources/bar.png")
                .setSourceFile(
                    getSourceArtifact("x/Foo.framework/Resources/bar.png").getExecPathString())
                .setExternalFileAttribute(BundleableFile.DEFAULT_EXTERNAL_FILE_ATTRIBUTE)
                .build());

    assertThat(mergeControl.getBundleFileList())
        .doesNotContain(
            BundleFile.newBuilder()
                .setBundlePath("Frameworks/Foo.framework/Headers/Foo.h")
                .setSourceFile(
                    getSourceArtifact("x/Foo.framework/Headers/Foo.h").getExecPathString())
                .setExternalFileAttribute(BundleableFile.DEFAULT_EXTERNAL_FILE_ATTRIBUTE)
                .build());
  }

  @Test
  public void testDynamicFrameworkSigned() throws Exception {
    useConfiguration("--ios_cpu=arm64");

    scratch.file("x/Foo.framework/Foo");
    scratch.file("x/Foo.framework/Info.plist");
    scratch.file("x/Foo.framework/Headers/Foo.h");
    scratch.file("x/Foo.framework/Resources/bar.png");
    scratch.file(
        "x/BUILD",
        "objc_framework(",
        "    name = 'foo_framework',",
        "    framework_imports = glob(['Foo.framework/**']),",
        "    is_dynamic = 1,",
        ")",
        "",
        "objc_binary(",
        "    name = 'bin',",
        "    srcs = [ 'a.m' ],",
        "    deps = [ ':foo_framework' ],",
        ")",
        "",
        "ios_application(",
        "    name = 'x',",
        "    binary = ':bin',",
        ")");

    SpawnAction signingAction = (SpawnAction) ipaGeneratingAction();

    assertThat(normalizeBashArgs(signingAction.getArguments()))
        .containsAllOf("--sign", "${t}/Payload/x.app/Frameworks/*", "--sign", "${t}/Payload/x.app")
        .inOrder();
  }
}
