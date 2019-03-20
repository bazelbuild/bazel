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
import com.google.common.collect.Maps;
import com.google.common.io.BaseEncoding;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.skylark.BazelStarlarkContext;
import com.google.devtools.build.lib.analysis.skylark.SymbolGenerator;
import com.google.devtools.build.lib.bazel.repository.RepositoryResolvedEvent;
import com.google.devtools.build.lib.bazel.repository.downloader.HttpDownloader;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.repository.ExternalPackageException;
import com.google.devtools.build.lib.repository.ExternalPackageUtil;
import com.google.devtools.build.lib.rules.repository.RepositoryDelegatorFunction;
import com.google.devtools.build.lib.rules.repository.RepositoryDirectoryValue;
import com.google.devtools.build.lib.rules.repository.RepositoryFunction;
import com.google.devtools.build.lib.rules.repository.ResolvedHashesValue;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.build.lib.syntax.BaseFunction;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Mutability;
import com.google.devtools.build.lib.syntax.StarlarkSemantics;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * A repository function to delegate work done by skylark remote repositories.
 */
public class SkylarkRepositoryFunction extends RepositoryFunction {

  private static final String REFRESH_ROOT = "REFRESH_ROOT:";
  private static final String DO_NOT_EXISTS = "DO_NOT_EXISTS";
  private final HttpDownloader httpDownloader;
  private double timeoutScaling = 1.0;

  public SkylarkRepositoryFunction(HttpDownloader httpDownloader) {
    this.httpDownloader = httpDownloader;
  }

  public void setTimeoutScaling(double timeoutScaling) {
    this.timeoutScaling = timeoutScaling;
  }

  @Nullable
  @Override
  public RepositoryDirectoryValue.Builder fetch(
      Rule rule,
      Path outputDirectory,
      BlazeDirectories directories,
      Environment env,
      Map<String, String> markerData,
      SkyKey key)
      throws RepositoryFunctionException, InterruptedException, ExternalPackageException {
    BaseFunction function = rule.getRuleClassObject().getConfiguredTargetFunction();
    if (declareEnvironmentDependencies(markerData, env, getEnviron(rule)) == null) {
      return null;
    }
    StarlarkSemantics starlarkSemantics = PrecomputedValue.STARLARK_SEMANTICS.get(env);
    if (starlarkSemantics == null) {
      return null;
    }

    Set<String> verificationRules =
        RepositoryDelegatorFunction.OUTPUT_VERIFICATION_REPOSITORY_RULES.get(env);
    if (verificationRules == null) {
      return null;
    }
    ResolvedHashesValue resolvedHashesValue =
        (ResolvedHashesValue) env.getValue(ResolvedHashesValue.key());
    if (resolvedHashesValue == null) {
      return null;
    }
    RepositoryName repositoryName = (RepositoryName) key.argument();
    List<RootedPath> refreshRoots = ExternalPackageUtil.getRefreshRootsPaths(repositoryName, env);
    if (env.valuesMissing()) {
      return null;
    }
    Map<String, String> resolvedHashes = resolvedHashesValue.getHashes();

    try (Mutability mutability = Mutability.create("Starlark repository")) {
      com.google.devtools.build.lib.syntax.Environment buildEnv =
          com.google.devtools.build.lib.syntax.Environment.builder(mutability)
              .setSemantics(starlarkSemantics)
              .setEventHandler(env.getListener())
              // The fetch phase does not need the tools repository or the fragment map because
              // it happens before analysis.
              .setStarlarkContext(
                  new BazelStarlarkContext(
                      /* toolsRepository = */ null,
                      /* fragmentNameToClass = */ null,
                      rule.getPackage().getRepositoryMapping(),
                      new SymbolGenerator<>(key)))
              .build();
      SkylarkRepositoryContext skylarkRepositoryContext =
          new SkylarkRepositoryContext(
              rule,
              outputDirectory,
              env,
              clientEnvironment,
              httpDownloader,
              timeoutScaling,
              markerData);

      if (!enforceLabelAttributes(skylarkRepositoryContext)) {
        return null;
      }

      // This rule is mainly executed for its side effect. Nevertheless, the return value is
      // of importance, as it provides information on how the call has to be modified to be a
      // reproducible rule.
      //
      // Also we do a lot of stuff in there, maybe blocking operations and we should certainly make
      // it possible to return null and not block but it doesn't seem to be easy with Skylark
      // structure as it is.
      Object retValue =
          function.call(
              /*args=*/ ImmutableList.of(skylarkRepositoryContext),
              /*kwargs=*/ ImmutableMap.of(),
              null,
              buildEnv);
      RepositoryResolvedEvent resolved =
          new RepositoryResolvedEvent(
              rule, skylarkRepositoryContext.getAttr(), outputDirectory, retValue);
      if (resolved.isNewInformationReturned()) {
        env.getListener().handle(Event.debug(resolved.getMessage()));
      }

      String ruleClass =
          rule.getRuleClassObject().getRuleDefinitionEnvironmentLabel() + "%" + rule.getRuleClass();
      if (verificationRules.contains(ruleClass)) {
        String expectedHash = resolvedHashes.get(rule.getName());
        if (expectedHash != null) {
          String actualHash = resolved.getDirectoryDigest();
          if (!expectedHash.equals(actualHash)) {
            throw new RepositoryFunctionException(
                new IOException(
                    rule + " failed to create a directory with expected hash " + expectedHash),
                Transience.PERSISTENT);
          }
        }
      }
      env.getListener().post(resolved);
    } catch (EvalException e) {
      if (e.getCause() instanceof RepositoryMissingDependencyException) {
        // A dependency is missing, cleanup and returns null
        try {
          if (outputDirectory.exists()) {
            outputDirectory.deleteTree();
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

    if (!outputDirectory.getRelative(LabelConstants.WORKSPACE_FILE_NAME).exists()) {
      createWorkspaceFile(outputDirectory, rule.getTargetKind(), rule.getName());
    }

    fillMarkerData(markerData, refreshRoots);
    return RepositoryDirectoryValue.builder().setPath(outputDirectory);
  }

  @Override
  public boolean needsMarkerDirtinessCheck(RepositoryName repositoryName, Environment env)
      throws InterruptedException {
    boolean hasRefreshRoots = ExternalPackageUtil.hasRefreshRoots(repositoryName, env);
    if (env.valuesMissing()) {
      return false;
    }
    return hasRefreshRoots;
  }

  private boolean fillRefreshRootsMarkerData(RepositoryName repositoryName, Environment env,
      Map<String, String> markerData)
      throws InterruptedException, ExternalPackageException, RepositoryFunctionException {
    List<RootedPath> refreshRoots = ExternalPackageUtil.getRefreshRootsPaths(repositoryName, env);
    if (env.valuesMissing()) {
      return false;
    }
    fillMarkerData(markerData, refreshRoots);
    return true;
  }

  private void fillMarkerData(Map<String, String> markerData, List<RootedPath> refreshRoots)
      throws RepositoryFunctionException {
    try {
      // we can not touch SkyFrame with these directories, or it will be the circular dependency
      for (RootedPath r : Objects.requireNonNull(refreshRoots)) {
        String key = REFRESH_ROOT + r.getRootRelativePath().getPathString();
        Path path = r.asPath();
        if (!path.exists()) {
          markerData.put(key, DO_NOT_EXISTS);
        } else {
          // the structural changes to the directory will be detected
          long lastModifiedTime = path.getLastModifiedTime(Symlinks.NOFOLLOW);
          markerData.put(key, String.valueOf(lastModifiedTime));
        }
      }
    } catch (IOException e) {
      throw new RepositoryFunctionException(e, Transience.TRANSIENT);
    }
  }

  @Override
  public boolean verifyMarkerData(RepositoryName repositoryName, Rule rule,
      Map<String, String> markerData, Environment env)
      throws InterruptedException, RepositoryFunctionException, ExternalPackageException, IOException {

    Map<String, String> newMarkerData = Maps.newHashMap();
    if (!fillRefreshRootsMarkerData(repositoryName, env, newMarkerData)) {
      return false;
    }
    Map<String, String> filteredMap = Maps.newHashMap();
    // we can not touch SkyFrame with these directories, or it will be the circular dependency
    for (String key : markerData.keySet()) {
      if (key.startsWith(REFRESH_ROOT)) {
        filteredMap.put(key, markerData.get(key));
      }
    }
    if (newMarkerData.containsValue(DO_NOT_EXISTS) || !newMarkerData.equals(filteredMap)) {
      return false;
    }
    return super.verifyMarkerData(repositoryName, rule, markerData, env);
  }

  private boolean enforceLabelAttributes(SkylarkRepositoryContext skylarkRepositoryContext)
      throws InterruptedException {
    // Since restarting a repository function can be really expensive, we first ensure that
    // all label-arguments can be resolved to paths.
    try {
      skylarkRepositoryContext.enforceLabelAttributes();
    } catch (EvalException e) {
      if (e instanceof RepositoryMissingDependencyException) {
        // Missing values are expected; just restart before we actually start the rule
        return false;
      }
      // Other EvalExceptions indicate labels not referring to existing files. This is fine,
      // as long as they are never resolved to files in the execution of the rule; we allow
      // non-strict rules. So now we have to start evaluating the actual rule, even if that
      // means the rule might get restarted for legitimate reasons.
    }
    return true;
  }

  @SuppressWarnings("unchecked")
  private static Iterable<String> getEnviron(Rule rule) {
    return (Iterable<String>) rule.getAttributeContainer().getAttr("$environ");
  }

  @Override
  protected boolean isLocal(Rule rule) {
    Object isLocal = rule.getAttributeContainer().getAttr("$local");
    return Boolean.TRUE.equals(isLocal);
  }

  @Override
  public Class<? extends RuleDefinition> getRuleDefinition() {
    return null; // unused so safe to return null
  }
}
