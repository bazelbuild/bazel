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
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.config.OutputDirectories.OutputDirectory;
import com.google.devtools.build.lib.analysis.test.TestConfiguration;
import com.google.devtools.build.lib.vfs.Root;

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
}
