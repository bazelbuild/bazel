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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.TreeArtifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.SpawnActionTemplate;
import com.google.devtools.build.lib.analysis.actions.SpawnActionTemplate.OutputPathMapper;
import com.google.devtools.build.lib.analysis.starlark.StarlarkRuleContext;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.vfs.PathFragment;
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
  // Based on com.android.resources.ResourceFolderType
  private static final ImmutableSet<String> RESOURCE_DIRECTORY_TYPES =
      ImmutableSet.of(
          "anim",
          "animator",
          "color",
          "drawable",
          "font",
          "interpolator",
          "layout",
          "menu",
          "mipmap",
          "navigation",
          "raw",
          "transition",
          "values",
          "xml");

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
    PathFragment resourceDir = findResourceDir(resource);
    if (resourceDir == null) {
      return null;
    }
    return trimTo(resource.getRootRelativePath(), resourceDir).toString();
  }

  /**
   * Finds and validates the resource directory PathFragment from the artifact Path.
   *
   * <p>If the artifact is not a Fileset, the resource directory is presumed to be the second
   * directory from the end. Filesets are expect to have the last directory as the resource
   * directory.
   */
  @Nullable
  private static PathFragment findResourceDir(Artifact artifact) {
    PathFragment fragment = artifact.getExecPath();
    int segmentCount = fragment.segmentCount();
    if (segmentCount < 3) {
      return null;
    }
    // TODO(bazel-team): Expand Fileset to verify, or remove Fileset as an option for resources.
    if (artifact.isFileset() || artifact.isTreeArtifact()) {
      return fragment.subFragment(segmentCount - 1);
    }

    // Check the resource folder type layout.
    // get the prefix of the parent folder of the fragment.
    String parentDirectory = fragment.getSegment(segmentCount - 2);
    int dashIndex = parentDirectory.indexOf('-');
    String androidFolder =
        dashIndex == -1 ? parentDirectory : parentDirectory.substring(0, dashIndex);
    if (!RESOURCE_DIRECTORY_TYPES.contains(androidFolder)) {
      return null;
    }

    return fragment.subFragment(segmentCount - 3, segmentCount - 2);
  }

  /**
   * Finds the rightmost occurrence of the needle and returns subfragment of the haystack from left
   * to the end of the occurrence inclusive of the needle.
   *
   * <pre>
   * Example:
   *   Given the haystack:
   *     res/research/handwriting/res/values/strings.xml
   *   And the needle:
   *     res
   *   Returns:
   *     res/research/handwriting/res
   * </pre>
   */
  private static PathFragment trimTo(PathFragment haystack, PathFragment needle) {
    if (needle.equals(PathFragment.EMPTY_FRAGMENT)) {
      return haystack;
    }
    ImmutableList<String> needleSegments = needle.splitToListOfSegments();
    // Compute the overlap offset for duplicated parts of the needle.
    int[] overlap = new int[needleSegments.size() + 1];
    // Start overlap at -1, as it will cancel out the increment in the search.
    // See http://en.wikipedia.org/wiki/Knuth%E2%80%93Morris%E2%80%93Pratt_algorithm for the
    // details.
    overlap[0] = -1;
    for (int i = 0, j = -1; i < needleSegments.size(); j++, i++, overlap[i] = j) {
      while (j >= 0 && !needleSegments.get(i).equals(needleSegments.get(j))) {
        // Walk the overlap until the bound is found.
        j = overlap[j];
      }
    }
    // TODO(corysmith): reverse the search algorithm.
    // Keep the index of the found so that the rightmost index is taken.
    ImmutableList<String> haystackSegments = haystack.splitToListOfSegments();
    int found = -1;
    for (int i = 0, j = 0; i < haystackSegments.size(); i++) {

      while (j >= 0 && !haystackSegments.get(i).equals(needleSegments.get(j))) {
        // Not matching, walk the needle index to attempt another match.
        j = overlap[j];
      }
      j++;
      // Needle index is exhausted, so the needle must match.
      if (j == needleSegments.size()) {
        // Record the found index + 1 to be inclusive of the end index.
        found = i + 1;
        // Subtract one from the needle index to restart the search process
        j = j - 1;
      }
    }
    if (found != -1) {
      // Return the subsection of the haystack.
      return haystack.subFragment(0, found);
    }
    throw new IllegalArgumentException(String.format("%s was not found in %s", needle, haystack));
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
    if (!(output instanceof TreeArtifact outputTree)) {
      throw Starlark.errorf(
          "Expected 'output' to be a directory artifact, but got %s", output.prettyPrint());
    }
    if (!(input instanceof TreeArtifact inputTree)) {
      throw Starlark.errorf(
          "Expected 'input' to be a directory artifact, but got %s", input.prettyPrint());
    }
    createTemplatedMergerActions(
        starlarkRuleContext.getRuleContext(),
        outputTree,
        inputTree,
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
      TreeArtifact outputTree,
      TreeArtifact inputTree,
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
            .addAll(dexopts);
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
