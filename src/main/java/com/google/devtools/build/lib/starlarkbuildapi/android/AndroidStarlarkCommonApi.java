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
package com.google.devtools.build.lib.starlarkbuildapi.android;

import com.google.devtools.build.lib.analysis.config.transitions.StarlarkExposedRuleTransitionFactory;
import com.google.devtools.build.lib.packages.Info;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.starlarkbuildapi.FileApi;
import com.google.devtools.build.lib.starlarkbuildapi.FilesToRunProviderApi;
import com.google.devtools.build.lib.starlarkbuildapi.StarlarkRuleContextApi;
import com.google.devtools.build.lib.starlarkbuildapi.java.JavaInfoApi;
import com.google.devtools.build.lib.starlarkbuildapi.platform.ConstraintValueInfoApi;
import javax.annotation.Nullable;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.ParamType;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.StarlarkInt;
import net.starlark.java.eval.StarlarkValue;

/** Common utilities for Starlark rules related to Android. */
@StarlarkBuiltin(
    name = "android_common",
    doc =
        "Do not use this module. It is intended for migration purposes only. If you depend on it, "
            + "you will be broken when it is removed."
            + "Common utilities and functionality related to Android rules.",
    documented = false)
public interface AndroidStarlarkCommonApi<
        FileT extends FileApi,
        JavaInfoT extends JavaInfoApi<?, ?, ?>,
        FilesToRunProviderT extends FilesToRunProviderApi<FileT>,
        ConstraintValueT extends ConstraintValueInfoApi,
        StarlarkRuleContextT extends StarlarkRuleContextApi<ConstraintValueT>>
    extends StarlarkValue {

  @StarlarkMethod(
      name = "create_device_broker_info",
      documented = false,
      parameters = {@Param(name = "type")})
  AndroidDeviceBrokerInfoApi createDeviceBrokerInfo(String deviceBrokerType);

  @StarlarkMethod(
      name = "resource_source_directory",
      allowReturnNones = true,
      doc =
          "Returns a source directory for Android resource file. "
              + "The source directory is a prefix of resource's relative path up to "
              + "a directory that designates resource kind (cf. "
              + "http://developer.android.com/guide/topics/resources/providing-resources.html).",
      documented = false,
      parameters = {
        @Param(
            name = "resource",
            doc = "The android resource file.",
            positional = true,
            named = false)
      })
  @Nullable
  String getSourceDirectoryRelativePathFromResource(FileT resource);

  @StarlarkMethod(
      name = "multi_cpu_configuration",
      doc =
          "A configuration for rule attributes that compiles native code according to "
              + "the --android_platforms flag.",
      documented = false,
      structField = true)
  AndroidSplitTransitionApi getAndroidSplitTransition();

  @StarlarkMethod(
      name = "android_platforms_transition",
      doc =
          "A configuration for rules that uses the --android_platforms flag instead of --platforms."
              + " This should only be used by Android rules during migration and is not for"
              + " general use.",
      documented = false,
      structField = true)
  StarlarkExposedRuleTransitionFactory getAndroidPlatformsTransition();

  @StarlarkMethod(
      name = "enable_implicit_sourceless_deps_exports_compatibility",
      doc = "Takes a JavaInfo and converts it to an implicit exportable JavaInfo.",
      documented = false,
      enableOnlyWithFlag = BuildLanguageOptions.EXPERIMENTAL_ENABLE_ANDROID_MIGRATION_APIS,
      parameters = {
        @Param(
            name = "dep",
            doc =
                "A JavaInfo that will be used as an implicit export for sourceless deps exports"
                    + " compatibility.",
            positional = true,
            named = false),
        @Param(
            name = "neverlink",
            doc = "Whether to only use this JavaInfo at compile time and not at runtime.",
            positional = true,
            defaultValue = "False")
      })
  JavaInfoT enableImplicitSourcelessDepsExportsCompatibility(Info javaInfo, boolean neverlink)
      throws RuleErrorException;

  @StarlarkMethod(
      name = "create_dex_merger_actions",
      doc =
          "Creates a list of DexMerger actions to be run in parallel, each action taking one shard"
              + " from the input directory, merging all the dex archives inside the shard to a"
              + " single dexarchive under the output directory.",
      documented = false,
      enableOnlyWithFlag = BuildLanguageOptions.EXPERIMENTAL_ENABLE_ANDROID_MIGRATION_APIS,
      parameters = {
        @Param(name = "ctx", doc = "The rule context.", positional = true, named = false),
        @Param(
            name = "output",
            doc = "The output directory.",
            positional = false,
            named = true,
            allowedTypes = {@ParamType(type = FileApi.class)}),
        @Param(
            name = "input",
            doc = "The input directory.",
            positional = false,
            named = true,
            allowedTypes = {@ParamType(type = FileApi.class)}),
        @Param(
            name = "dexopts",
            doc = "A list of additional command-line flags for the dx tool. Optional",
            positional = false,
            named = true,
            allowedTypes = {@ParamType(type = Sequence.class, generic1 = String.class)},
            defaultValue = "[]"),
        @Param(
            name = "dexmerger",
            doc = "A FilesToRunProvider to be used for dex merging.",
            positional = false,
            named = true,
            allowedTypes = {@ParamType(type = FilesToRunProviderApi.class)}),
        @Param(
            name = "min_sdk_version",
            doc = "The minSdkVersion the dexes were built for.",
            positional = false,
            named = true,
            defaultValue = "0",
            allowedTypes = {
              @ParamType(type = StarlarkInt.class),
            })
      })
  void createDexMergerActions(
      StarlarkRuleContextT starlarkRuleContext,
      FileT output,
      FileT input,
      Sequence<?> dexopts, // <String> expected.
      FilesToRunProviderT dexmerger,
      StarlarkInt minSdkVersion)
      throws EvalException, RuleErrorException;
}
