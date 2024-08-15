// Copyright 2024 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe.serialization;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSortedMap;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.config.OutputDirectories.OutputDirectory;
import com.google.devtools.build.lib.analysis.test.TestConfiguration;
import com.google.devtools.build.lib.packages.BazelStarlarkEnvironment;
import com.google.devtools.build.lib.packages.StructProvider;
import com.google.devtools.build.lib.vfs.Root;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import net.starlark.java.eval.Starlark;
import net.starlark.java.syntax.Location;

/** Objects that should be serialized as constants in Skyframe serialization. */
public final class CommonSerializationConstants {

  private CommonSerializationConstants() {}

  private static final ImmutableList<String> OUTPUT_PATHS =
      ImmutableList.of("k8-opt", "k8-fastbuild", "k8-debug");

  public static ImmutableList<Object> makeReferenceConstants(
      BlazeDirectories directories,
      ConfiguredRuleClassProvider ruleClassProvider,
      String workspaceName) {
    ImmutableList.Builder<Object> referenceConstants =
        ImmutableList.builder()
            .add(directories)
            .add(directories.getExecRootBase().getFileSystem())
            .add(directories.getBuildDataDirectory(workspaceName))
            .add(ruleClassProvider.getFragmentRegistry().getAllFragments())
            // Commonly referenced if --trim_test_configuration is enabled.
            .add(
                ruleClassProvider
                    .getFragmentRegistry()
                    .getAllFragments()
                    .trim(TestConfiguration.class));

    Root virtualRoot = directories.getVirtualSourceRoot();
    if (virtualRoot != null) {
      referenceConstants.add(ArtifactRoot.asSourceRoot(virtualRoot));
    }

    // The builtins bzl root (if it exists) lives on a separate InMemoryFileSystem.
    Root builtinsRoot = ruleClassProvider.getBundledBuiltinsRoot();
    if (builtinsRoot != null) {
      referenceConstants.add(builtinsRoot);
    }

    for (OutputDirectory outputDirectory : OutputDirectory.values()) {
      for (String outputPath : OUTPUT_PATHS) {
        referenceConstants.add(outputDirectory.getRoot(outputPath, directories, workspaceName));
      }
    }
    return referenceConstants.build();
  }

  @CanIgnoreReturnValue
  public static ObjectCodecRegistry.Builder addStarlarkFunctionality(
      ObjectCodecRegistry.Builder builder, ConfiguredRuleClassProvider ruleClassProvider) {
    BazelStarlarkEnvironment starlarkEnv = ruleClassProvider.getBazelStarlarkEnvironment();
    builder
        .addReferenceConstant(StructProvider.STRUCT)
        .addReferenceConstant(Starlark.NONE)
        .addReferenceConstant(Location.BUILTIN)
        .addReferenceConstants(
            ImmutableSortedMap.copyOf(starlarkEnv.getUninjectedBuildBzlEnv()).values());

    // Make reference constants for all the native module's potential elements, so that something
    // like myvar = native.cc_test in a .bzl file doesn't cause problems (otherwise we'd have to
    // know how to serialize native.cc_test).
    //
    // Some of these elements may be overridden to Starlark values by builtins injection; see
    // StarlarkBuiltinsFunction. The native module object itself is not registered because it is
    // constructed during builtins injection.
    //
    // TODO(b/111564291): how do we get access to all other Starlark built-ins (ones in
    // apple_common, for instance) and register those? Currently most of those objects are fairly
    // simple to serialize, but that may change in the future. Also be mindful of whether
    // StarlarkSemantics (i.e., incompatible/experimental flags) can affect the bindings we see
    // here. [brandjon: May be able to use the new method
    // BazelStarlarkEnvironment#getUninjectedBuildBzlEnv.]
    builder
        .addReferenceConstants(
            ImmutableSortedMap.copyOf(starlarkEnv.getUninjectedBuildBzlNativeBindings()).values())
        .addReferenceConstants(
            ImmutableSortedMap.copyOf(starlarkEnv.getWorkspaceBzlNativeBindings()).values());

    return builder;
  }
}
