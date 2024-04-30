// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.repository;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.bazel.ResolvedEvent;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.XattrProvider;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyKey;
import java.util.Map;
import net.starlark.java.eval.Starlark;

/**
 * Access a repository on the local filesystem.
 */
public class LocalRepositoryFunction extends RepositoryFunction {

  @Override
  public boolean isLocal(Rule rule) {
    return true;
  }

  @Override
  public RepositoryDirectoryValue.Builder fetch(
      Rule rule,
      Path outputDirectory,
      BlazeDirectories directories,
      Environment env,
      Map<RepoRecordedInput, String> recordedInputValues,
      SkyKey key)
      throws InterruptedException, RepositoryFunctionException {
    ensureNativeRepoRuleEnabled(
        rule, env, "load(\"@bazel_tools//tools/build_defs/repo:local.bzl\", \"local_repository\")");
    // DO NOT MODIFY THIS! It's being deprecated in favor of Starlark counterparts.
    // See https://github.com/bazelbuild/bazel/issues/18285
    String userDefinedPath = RepositoryFunction.getPathAttr(rule);
    Path targetPath = directories.getWorkspace().getRelative(userDefinedPath);
    RepositoryDirectoryValue.Builder result =
        RepositoryDelegatorFunction.symlinkRepoRoot(
            directories, outputDirectory, targetPath, userDefinedPath, env);
    if (result != null) {
      env.getListener().post(resolve(rule, directories));
    }
    return result;
  }

  @Override
  public Class<? extends RuleDefinition> getRuleDefinition() {
    return LocalRepositoryRule.class;
  }

  private static ResolvedEvent resolve(Rule rule, BlazeDirectories directories) {
    String name = rule.getName();
    Object pathObj = rule.getAttr("path");
    String path;
    if (pathObj instanceof String) {
      path = (String) pathObj;
    } else {
      path = "";
    }
    // Find a descrption of the path; there is a case, where we do not(!) want to hard-code
    // the argument we obtained: if the path is under the embedded binaries root.
    String pathArg;
    PathFragment pathFragment = PathFragment.create(path);
    PathFragment embeddedDir = directories.getEmbeddedBinariesRoot().asFragment();
    if (pathFragment.isAbsolute() && pathFragment.startsWith(embeddedDir)) {
      pathArg =
          "__embedded_dir__ + \"/\" + "
              + Starlark.repr(pathFragment.relativeTo(embeddedDir).toString());
    } else {
      pathArg = Starlark.repr(path);
    }
    String repr =
        String.format("local_repository(name = %s, path = %s)", Starlark.repr(name), pathArg);
    return new ResolvedEvent() {
      @Override
      public String getName() {
        return name;
      }

      @Override
      public Object getResolvedInformation(XattrProvider xattrProvider) {
        return ImmutableMap.<String, Object>builder()
            .put(ResolvedFileValue.ORIGINAL_RULE_CLASS, "local_repository")
            .put(
                ResolvedFileValue.ORIGINAL_ATTRIBUTES,
                ImmutableMap.<String, Object>builder().put("name", name).put("path", path).build())
            .put(ResolvedFileValue.NATIVE, repr)
            .build();
      }
    };
  }
}
