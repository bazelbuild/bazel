// Copyright 2018 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.blackbox.bazel;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.blackbox.framework.BlackBoxTestContext;
import com.google.devtools.build.lib.blackbox.framework.ToolsSetup;
import com.google.devtools.build.lib.util.OS;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;

/** Setup for Bazel default tools */
public class DefaultToolsSetup implements ToolsSetup {

  private static ImmutableList<String> repos =
      ImmutableList.<String>builder()
          .add("bazel_skylib")
          .add("rules_cc")
          .add("rules_proto")
          .add("rules_java")
          .add("rules_java_builtin_for_testing")
          .add("rules_python")
          .build();

  private ImmutableList<String> getRepositoryOverrides() {
    String sharedRepoHome = System.getenv("TEST_REPOSITORY_HOME");
    if (sharedRepoHome == null) {
      return ImmutableList.of();
    }

    Path sharedRepoHomePath = Paths.get(sharedRepoHome);
    if (!Files.exists(sharedRepoHomePath)) {
      return ImmutableList.of();
    }

    ImmutableList.Builder<String> lines = ImmutableList.builder();
    for (String repo : repos) {
      Path sharedRepoPath = sharedRepoHomePath.resolve(repo);
      String suffix = "_for_testing";
      repo = repo.endsWith(suffix) ? repo.substring(0, repo.length() - suffix.length()) : repo;
      lines.add(
          "common --override_repository="
              + repo
              + "="
              + sharedRepoPath.toString().replace('\\', '/'));
    }

    return lines.build();
  }

  @Override
  public void setup(BlackBoxTestContext context) throws IOException {
    Path outputRoot = Files.createTempDirectory(context.getTmpDir(), "root").toAbsolutePath();
    ArrayList<String> lines = new ArrayList<>();
    lines.add("startup --output_user_root=" + outputRoot.toString().replace('\\', '/'));
    lines.addAll(getRepositoryOverrides());

    String sharedInstallBase = System.getenv("TEST_INSTALL_BASE");
    if (sharedInstallBase != null) {
      lines.add("startup --install_base=" + sharedInstallBase);
    }

    String sharedRepoCache = System.getenv("REPOSITORY_CACHE");
    if (sharedRepoCache != null) {
      lines.add("common --repository_cache=" + sharedRepoCache);
      // TODO: Remove this flag once all dependencies are mirrored.
      // See https://github.com/bazelbuild/bazel/pull/19549 for more context.
      lines.add("common --repo_env=BAZEL_HTTP_RULES_URLS_AS_DEFAULT_CANONICAL_ID=0");
      if (OS.getCurrent() == OS.DARWIN) {
        // For reducing SSD usage on our physical Mac machines.
        lines.add("common --experimental_repository_cache_hardlinks");
      }
    }

    if (OS.getCurrent() == OS.DARWIN) {
      // Prefer ipv6 network on macOS
      lines.add("startup --host_jvm_args=-Djava.net.preferIPv6Addresses=true");
      lines.add("build --jvmopt=-Djava.net.preferIPv6Addresses");
    }

    context.write(".bazelrc", lines);
  }
}
