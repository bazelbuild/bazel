// Copyright 2016 The Bazel Authors. All rights reserved.
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

import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.SpawnActionTemplate;
import com.google.devtools.build.lib.analysis.actions.SpawnActionTemplate.OutputPathMapper;
import com.google.devtools.build.lib.analysis.starlark.StarlarkRuleContext;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import java.io.Serializable;
import java.util.List;
import javax.annotation.Nullable;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.ParamType;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.NoneType;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.Starlark;
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
public class AndroidStarlarkCommon implements StarlarkValue {
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
  public String getSourceDirectoryRelativePathFromResource(Artifact resource) {
    return AndroidCommon.getSourceDirectoryRelativePathFromResource(resource).toString();
  }

  @StarlarkMethod(
      name = "create_dex_merger_actions",
      doc =
          "Creates a list of DexMerger actions to be run in parallel, each action taking one shard"
              + " from the input directory, merging all the dex archives inside the shard to a"
              + " single dexarchive under the output directory.",
      documented = false,
      parameters = {
        @Param(name = "ctx", doc = "The rule context.", positional = true, named = false),
        @Param(
            name = "output",
            doc = "The output directory.",
            positional = false,
            named = true,
            allowedTypes = {@ParamType(type = Artifact.class)}),
        @Param(
            name = "input",
            doc = "The input directory.",
            positional = false,
            named = true,
            allowedTypes = {@ParamType(type = Artifact.class)}),
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
            allowedTypes = {@ParamType(type = FilesToRunProvider.class)}),
        @Param(
            name = "min_sdk_version",
            doc = "The minSdkVersion the dexes were built for.",
            positional = false,
            named = true,
            defaultValue = "0",
            allowedTypes = {
              @ParamType(type = StarlarkInt.class),
            }),
        @Param(
            name = "desugar_globals",
            doc = "The D8 desugar globals file.",
            positional = false,
            named = true,
            defaultValue = "None",
            allowedTypes = {
              @ParamType(type = Artifact.class),
              @ParamType(type = NoneType.class),
            }),
      })
  public void createDexMergerActions(
      StarlarkRuleContext starlarkRuleContext,
      Artifact output,
      Artifact input,
      Sequence<?> dexopts, // <String> expected.
      FilesToRunProvider dexmerger,
      StarlarkInt minSdkVersion,
      Object desugarGlobals)
      throws EvalException, RuleErrorException {
    createTemplatedMergerActions(
        starlarkRuleContext.getRuleContext(),
        (SpecialArtifact) output,
        (SpecialArtifact) input,
        Sequence.cast(dexopts, String.class, "dexopts"),
        dexmerger,
        minSdkVersion.toInt("min_sdk_version"),
        desugarGlobals);
  }

  /**
   * Sets up a monodex {@code $dexmerger} actions for each dex archive in the given tree artifact
   * and puts the outputs in a tree artifact.
   */
  private static void createTemplatedMergerActions(
      RuleContext ruleContext,
      SpecialArtifact outputTree,
      SpecialArtifact inputTree,
      List<String> dexopts,
      FilesToRunProvider executable,
      int minSdkVersion,
      Object desugarGlobals) {
    SpawnActionTemplate.Builder dexmerger =
        new SpawnActionTemplate.Builder(inputTree, outputTree)
            .setExecutable(executable)
            .setMnemonics("DexShardsToMerge", "DexMerger")
            .setOutputPathMapper(
                (OutputPathMapper & Serializable) TreeFileArtifact::getParentRelativePath);
    CustomCommandLine.Builder commandLine =
        CustomCommandLine.builder()
            .addPlaceholderTreeArtifactExecPath("--input", inputTree)
            .addPlaceholderTreeArtifactExecPath("--output", outputTree)
            .add("--multidex=given_shard")
            .addAll(
                AndroidCommon.mergerDexopts(
                    ruleContext,
                    Iterables.filter(
                        dexopts, Predicates.not(Predicates.equalTo("--minimal-main-dex")))));
    if (minSdkVersion > 0) {
      commandLine.add("--min_sdk_version", Integer.toString(minSdkVersion));
    }
    Artifact desugarGlobalsArtifact = fromNoneable(desugarGlobals, Artifact.class);
    if (desugarGlobalsArtifact != null) {
      dexmerger.addCommonInputs(ImmutableList.of(desugarGlobalsArtifact));
      commandLine.addPath("--global_synthetics_path", desugarGlobalsArtifact.getExecPath());
    }
    dexmerger.setCommandLineTemplate(commandLine.build());
    ruleContext.registerAction(dexmerger.build(ruleContext.getActionOwner()));
  }

  /**
   * Checks if a "Noneable" object passed by Starlark is "None", which Java should treat as null.
   */
  private static boolean isNone(Object object) {
    return object == Starlark.NONE;
  }

  /**
   * Converts a "Noneable" Object passed by Starlark to an nullable object of the appropriate type.
   *
   * <p>Starlark "Noneable" types are passed in as an Object that may be either the correct type or
   * a Starlark.NONE object. Starlark will handle type checking, based on the appropriate @param
   * annotation, but we still need to do the actual cast (or conversion to null) ourselves.
   *
   * @param object the Noneable object
   * @param clazz the correct class, as defined in the @Param annotation
   * @param <T> the type to cast to
   * @return {@code null}, if the noneable argument was None, or the cast object, otherwise.
   */
  @Nullable
  private static <T> T fromNoneable(Object object, Class<T> clazz) {
    if (isNone(object)) {
      return null;
    }

    return clazz.cast(object);
  }
}
