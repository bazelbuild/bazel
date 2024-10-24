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
//

package com.google.devtools.build.lib.bazel.bzlmod;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableTable;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.rules.repository.RepoRecordedInput;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import java.util.Optional;
import javax.annotation.Nullable;
import net.starlark.java.eval.StarlarkSemantics;

/**
 * An internal abstraction to support the two "flavors" of module extensions: the "regular", which
 * is declared using {@code module_extension} in a .bzl file; and the "innate", which is fabricated
 * from usages of {@code use_repo_rule} in MODULE.bazel files.
 *
 * <p>The general idiom is to "load" such a {@link RunnableExtension} object by getting as much
 * information about it as needed to determine whether it can be reused from the lockfile (hence
 * methods such as {@link #getEvalFactors()}, {@link #getBzlTransitiveDigest()}, {@link
 * #getStaticEnvVars()}). Then the {@link #run} method can be called if it's determined that we
 * can't reuse the cached results in the lockfile and have to re-run this extension.
 */
interface RunnableExtension {
  ModuleExtensionEvalFactors getEvalFactors();

  byte[] getBzlTransitiveDigest();

  ImmutableMap<String, Optional<String>> getStaticEnvVars();

  /** Runs the extension. Returns null if a Skyframe restart is required. */
  @Nullable
  RunModuleExtensionResult run(
      Environment env,
      SingleExtensionUsagesValue usagesValue,
      StarlarkSemantics starlarkSemantics,
      ModuleExtensionId extensionId,
      RepositoryMapping mainRepositoryMapping)
      throws InterruptedException, ExternalDepsException;

  /* Holds the result data from running a module extension */
  record RunModuleExtensionResult(
      ImmutableMap<RepoRecordedInput.File, String> recordedFileInputs,
      ImmutableMap<RepoRecordedInput.Dirents, String> recordedDirentsInputs,
      ImmutableMap<RepoRecordedInput.EnvVar, Optional<String>> recordedEnvVarInputs,
      ImmutableMap<String, RepoSpec> generatedRepoSpecs,
      Optional<ModuleExtensionMetadata> moduleExtensionMetadata,
      ImmutableTable<RepositoryName, String, RepositoryName> recordedRepoMappingEntries) {}
}
