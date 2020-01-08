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

import com.google.common.base.Preconditions;
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.bazel.repository.RepositoryResolvedEvent;
import com.google.devtools.build.lib.bazel.repository.downloader.HttpDownloader;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.BazelStarlarkContext;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.SymbolGenerator;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.rules.repository.RepositoryDelegatorFunction;
import com.google.devtools.build.lib.rules.repository.RepositoryDirectoryValue;
import com.google.devtools.build.lib.rules.repository.RepositoryFunction;
import com.google.devtools.build.lib.rules.repository.ResolvedHashesValue;
import com.google.devtools.build.lib.rules.repository.WorkspaceFileHelper;
import com.google.devtools.build.lib.runtime.RepositoryRemoteExecutor;
import com.google.devtools.build.lib.skyframe.BlacklistedPackagePrefixesValue;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Mutability;
import com.google.devtools.build.lib.syntax.Starlark;
import com.google.devtools.build.lib.syntax.StarlarkFunction;
import com.google.devtools.build.lib.syntax.StarlarkSemantics;
import com.google.devtools.build.lib.syntax.StarlarkThread;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * A repository function to delegate work done by skylark remote repositories.
 */
public class SkylarkRepositoryFunction extends RepositoryFunction {

  private final HttpDownloader httpDownloader;
  private double timeoutScaling = 1.0;
  @Nullable private RepositoryRemoteExecutor repositoryRemoteExecutor;

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
      throws RepositoryFunctionException, InterruptedException {
    if (rule.getDefinitionInformation() != null) {
      env.getListener()
          .post(
              new SkylarkRepositoryDefinitionLocationEvent(
                  rule.getName(), rule.getDefinitionInformation()));
    }
    StarlarkFunction function = rule.getRuleClassObject().getConfiguredTargetFunction();
    if (declareEnvironmentDependencies(markerData, env, getEnviron(rule)) == null) {
      return null;
    }
    StarlarkSemantics starlarkSemantics = PrecomputedValue.STARLARK_SEMANTICS.get(env);
    if (env.valuesMissing()) {
      return null;
    }

    Set<String> verificationRules =
        RepositoryDelegatorFunction.OUTPUT_VERIFICATION_REPOSITORY_RULES.get(env);
    if (env.valuesMissing()) {
      return null;
    }
    ResolvedHashesValue resolvedHashesValue =
        (ResolvedHashesValue) env.getValue(ResolvedHashesValue.key());
    if (env.valuesMissing()) {
      return null;
    }
    Map<String, String> resolvedHashes =
        Preconditions.checkNotNull(resolvedHashesValue).getHashes();

    PathPackageLocator packageLocator = PrecomputedValue.PATH_PACKAGE_LOCATOR.get(env);
    if (env.valuesMissing()) {
      return null;
    }

    BlacklistedPackagePrefixesValue blacklistedPackagesValue =
        (BlacklistedPackagePrefixesValue) env.getValue(BlacklistedPackagePrefixesValue.key());
    if (env.valuesMissing()) {
      return null;
    }
    ImmutableSet<PathFragment> blacklistedPatterns =
        Preconditions.checkNotNull(blacklistedPackagesValue).getPatterns();

    try (Mutability mutability = Mutability.create("Starlark repository")) {
      StarlarkThread thread =
          StarlarkThread.builder(mutability)
              .setSemantics(starlarkSemantics)
              .build();
      thread.setPrintHandler(StarlarkThread.makeDebugPrintHandler(env.getListener()));

      // The fetch phase does not need the tools repository
      // or the fragment map because it happens before analysis.
      new BazelStarlarkContext(
              BazelStarlarkContext.Phase.LOADING, // ("fetch")
              /*toolsRepository=*/ null,
              /*fragmentNameToClass=*/ null,
              rule.getPackage().getRepositoryMapping(),
              new SymbolGenerator<>(key),
              /*analysisRuleLabel=*/ null)
          .storeInThread(thread);

      SkylarkRepositoryContext skylarkRepositoryContext =
          new SkylarkRepositoryContext(
              rule,
              packageLocator,
              outputDirectory,
              blacklistedPatterns,
              env,
              clientEnvironment,
              httpDownloader,
              directories.getEmbeddedBinariesRoot(),
              timeoutScaling,
              markerData,
              starlarkSemantics,
              repositoryRemoteExecutor);

      if (skylarkRepositoryContext.isRemotable()) {
        // If a rule is declared remotable then invalidate it if remote execution gets
        // enabled or disabled.
        PrecomputedValue.REMOTE_EXECUTION_ENABLED.get(env);
      }

      // Since restarting a repository function can be really expensive, we first ensure that
      // all label-arguments can be resolved to paths.
      try {
        skylarkRepositoryContext.enforceLabelAttributes();
      } catch (RepositoryMissingDependencyException e) {
        // Missing values are expected; just restart before we actually start the rule
        return null;
      } catch (EvalException e) {
        // EvalExceptions indicate labels not referring to existing files. This is fine,
        // as long as they are never resolved to files in the execution of the rule; we allow
        // non-strict rules. So now we have to start evaluating the actual rule, even if that
        // means the rule might get restarted for legitimate reasons.
      }

      // This rule is mainly executed for its side effect. Nevertheless, the return value is
      // of importance, as it provides information on how the call has to be modified to be a
      // reproducible rule.
      //
      // Also we do a lot of stuff in there, maybe blocking operations and we should certainly make
      // it possible to return null and not block but it doesn't seem to be easy with Skylark
      // structure as it is.
      Object result =
          Starlark.call(
              thread,
              function,
              /*args=*/ ImmutableList.of(skylarkRepositoryContext),
              /*kwargs=*/ ImmutableMap.of());
      RepositoryResolvedEvent resolved =
          new RepositoryResolvedEvent(
              rule, skylarkRepositoryContext.getAttr(), outputDirectory, result);
      if (resolved.isNewInformationReturned()) {
        env.getListener().handle(Event.debug(resolved.getMessage()));
        env.getListener().handle(Event.debug(rule.getDefinitionInformation()));
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
    } catch (RepositoryMissingDependencyException e) {
      // A dependency is missing, cleanup and returns null
      try {
        if (outputDirectory.exists()) {
          outputDirectory.deleteTree();
        }
      } catch (IOException e1) {
        throw new RepositoryFunctionException(e1, Transience.TRANSIENT);
      }
      return null;
    } catch (EvalException e) {
      env.getListener()
          .handle(
              Event.error(
                  "An error occurred during the fetch of repository '"
                      + rule.getName()
                      + "':\n   "
                      + e.getMessage()));
      if (!Strings.isNullOrEmpty(rule.getDefinitionInformation())) {
        env.getListener().handle(Event.info(rule.getDefinitionInformation()));
      }
      throw new RepositoryFunctionException(e, Transience.TRANSIENT);
    }

    if (!outputDirectory.isDirectory()) {
      throw new RepositoryFunctionException(
          new IOException(rule + " must create a directory"), Transience.TRANSIENT);
    }

    if (!WorkspaceFileHelper.doesWorkspaceFileExistUnder(outputDirectory)) {
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
  protected boolean isConfigure(Rule rule) {
    return (Boolean) rule.getAttributeContainer().getAttr("$configure");
  }

  /**
   * Static method to determine if for a starlark repository rule {@code isConfigure} holds true. It
   * also checks that the rule is indeed a Starlark rule so that this class is the appropriate
   * handler for the given rule. As, however, only Starklark rules can be configure rules, this
   * method can also be used as a universal check.
   */
  public static boolean isConfigureRule(Rule rule) {
    return rule.getRuleClassObject().isSkylark()
        && ((Boolean) rule.getAttributeContainer().getAttr("$configure"));
  }

  @Override
  public Class<? extends RuleDefinition> getRuleDefinition() {
    return null; // unused so safe to return null
  }

  public void setRepositoryRemoteExecutor(RepositoryRemoteExecutor repositoryRemoteExecutor) {
    this.repositoryRemoteExecutor = repositoryRemoteExecutor;
  }
}
