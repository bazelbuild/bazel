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

package com.google.devtools.build.lib.bazel.repository.skylark;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.bazel.repository.downloader.HttpDownloader;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.rules.repository.RepositoryDirectoryValue;
import com.google.devtools.build.lib.rules.repository.RepositoryFunction;
import com.google.devtools.build.lib.syntax.BaseFunction;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Mutability;
import com.google.devtools.build.lib.syntax.Runtime;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import java.io.IOException;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * A repository function to delegate work done by skylark remote repositories.
 */
public class SkylarkRepositoryFunction extends RepositoryFunction {

  private final HttpDownloader httpDownloader;

  public SkylarkRepositoryFunction(HttpDownloader httpDownloader) {
    this.httpDownloader = httpDownloader;
  }

  @Nullable
  @Override
  public RepositoryDirectoryValue.Builder fetch(Rule rule, Path outputDirectory,
      BlazeDirectories directories, Environment env, Map<String, String> markerData)
      throws RepositoryFunctionException, InterruptedException {
    BaseFunction function = rule.getRuleClassObject().getConfiguredTargetFunction();
    if (declareEnvironmentDependencies(markerData, env, getEnviron(rule)) == null) {
      return null;
    }
    try (Mutability mutability = Mutability.create("skylark repository")) {
      // This Skylark environment ignores command line flags.
      com.google.devtools.build.lib.syntax.Environment buildEnv =
          com.google.devtools.build.lib.syntax.Environment.builder(mutability)
              .useDefaultSemantics()
              .setEventHandler(env.getListener())
              .build();
      SkylarkRepositoryContext skylarkRepositoryContext =
          new SkylarkRepositoryContext(
              rule, outputDirectory, env, clientEnvironment, httpDownloader, markerData);

      // This has side-effect, we don't care about the output.
      // Also we do a lot of stuff in there, maybe blocking operations and we should certainly make
      // it possible to return null and not block but it doesn't seem to be easy with Skylark
      // structure as it is.
      Object retValue =
          function.call(
              /*args=*/ ImmutableList.of(skylarkRepositoryContext),
              /*kwargs=*/ ImmutableMap.of(),
              null,
              buildEnv);
      if (retValue != Runtime.NONE) {
        throw new RepositoryFunctionException(
            new EvalException(
                rule.getLocation(),
                "Call to repository rule "
                    + rule.getName()
                    + " returned a non-None value, None expected."),
            Transience.PERSISTENT);
      }
    } catch (EvalException e) {
      if (e.getCause() instanceof RepositoryMissingDependencyException) {
        // A dependency is missing, cleanup and returns null
        try {
          if (outputDirectory.exists()) {
            FileSystemUtils.deleteTree(outputDirectory);
          }
        } catch (IOException e1) {
          throw new RepositoryFunctionException(e1, Transience.TRANSIENT);
        }
        return null;
      }
      throw new RepositoryFunctionException(e, Transience.TRANSIENT);
    }

    if (!outputDirectory.isDirectory()) {
      throw new RepositoryFunctionException(
          new IOException(rule + " must create a directory"), Transience.TRANSIENT);
    }

    if (!outputDirectory.getRelative("WORKSPACE").exists()) {
      createWorkspaceFile(outputDirectory, rule.getTargetKind(), rule.getName());
    }

    return RepositoryDirectoryValue.builder().setPath(outputDirectory);
  }

  @SuppressWarnings("unchecked")
  private static Iterable<String> getEnviron(Rule rule) {
    return (Iterable<String>) rule.getAttributeContainer().getAttr("$environ");
  }

  @Override
  protected boolean isLocal(Rule rule) {
    return (Boolean) rule.getAttributeContainer().getAttr("$local");
  }

  @Override
  public Class<? extends RuleDefinition> getRuleDefinition() {
    return null; // unused so safe to return null
  }
}
