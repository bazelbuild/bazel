// Copyright 2021 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.bazel.repository.downloader.DownloadManager;
import com.google.devtools.build.lib.cmdline.BazelModuleContext;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.rules.repository.NeedsSkyframeRestartException;
import com.google.devtools.build.lib.runtime.ProcessWrapper;
import com.google.devtools.build.lib.runtime.RepositoryRemoteExecutor;
import com.google.devtools.build.lib.server.FailureDetails.ExternalDeps.Code;
import com.google.devtools.build.lib.skyframe.BzlLoadFunction;
import com.google.devtools.build.lib.skyframe.BzlLoadFunction.BzlLoadFailedException;
import com.google.devtools.build.lib.skyframe.BzlLoadValue;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Map;
import java.util.function.Supplier;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Mutability;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkList;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.syntax.Location;

/**
 * Evaluates a single module extension. This function loads the .bzl file containing the extension,
 * runs its implementation function with a module_ctx object containing all relevant information,
 * and returns the generated repos.
 */
public class SingleExtensionEvalFunction implements SkyFunction {
  private final BlazeDirectories directories;
  private final Supplier<Map<String, String>> clientEnvironmentSupplier;
  private final DownloadManager downloadManager;

  private double timeoutScaling = 1.0;
  @Nullable private ProcessWrapper processWrapper = null;
  @Nullable private RepositoryRemoteExecutor repositoryRemoteExecutor = null;

  public SingleExtensionEvalFunction(
      BlazeDirectories directories,
      Supplier<Map<String, String>> clientEnvironmentSupplier,
      DownloadManager downloadManager) {
    this.directories = directories;
    this.clientEnvironmentSupplier = clientEnvironmentSupplier;
    this.downloadManager = downloadManager;
  }

  public void setTimeoutScaling(double timeoutScaling) {
    this.timeoutScaling = timeoutScaling;
  }

  public void setProcessWrapper(ProcessWrapper processWrapper) {
    this.processWrapper = processWrapper;
  }

  public void setRepositoryRemoteExecutor(RepositoryRemoteExecutor repositoryRemoteExecutor) {
    this.repositoryRemoteExecutor = repositoryRemoteExecutor;
  }

  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws SkyFunctionException, InterruptedException {
    StarlarkSemantics starlarkSemantics = PrecomputedValue.STARLARK_SEMANTICS.get(env);
    if (starlarkSemantics == null) {
      return null;
    }

    ModuleExtensionId extensionId = (ModuleExtensionId) skyKey.argument();
    SingleExtensionUsagesValue usagesValue =
        (SingleExtensionUsagesValue) env.getValue(SingleExtensionUsagesValue.key(extensionId));
    if (usagesValue == null) {
      return null;
    }
    Location sampleUsageLocation =
        usagesValue.getExtensionUsages().values().iterator().next().getLocation();

    // Check that the .bzl label isn't crazy.
    try {
      BzlLoadFunction.checkValidLoadLabel(
          extensionId.getBzlFileLabel(), /*fromBuiltinsRepo=*/ false);
    } catch (LabelSyntaxException e) {
      throw new SingleExtensionEvalFunctionException(
          ExternalDepsException.withCauseAndMessage(
              Code.BAD_MODULE, e, "invalid module extension label"),
          Transience.PERSISTENT);
    }

    // Load the .bzl file pointed to by the label.
    BzlLoadValue bzlLoadValue;
    try {
      bzlLoadValue =
          (BzlLoadValue)
              env.getValueOrThrow(
                  BzlLoadValue.keyForBzlmod(extensionId.getBzlFileLabel()),
                  BzlLoadFailedException.class);
    } catch (BzlLoadFailedException e) {
      throw new SingleExtensionEvalFunctionException(
          ExternalDepsException.withCauseAndMessage(
              Code.BAD_MODULE,
              e,
              "Error loading '%s' for module extensions, requested by %s: %s",
              extensionId.getBzlFileLabel(),
              sampleUsageLocation,
              e.getMessage()),
          Transience.PERSISTENT);
    }
    if (bzlLoadValue == null) {
      return null;
    }

    // Check that the .bzl file actually exports a module extension by our name.
    Object exported = bzlLoadValue.getModule().getGlobal(extensionId.getExtensionName());
    if (!(exported instanceof ModuleExtension.InStarlark)) {
      throw new SingleExtensionEvalFunctionException(
          ExternalDepsException.withMessage(
              Code.BAD_MODULE,
              "%s does not export a module extension called %s, yet its use is requested at %s",
              extensionId.getBzlFileLabel(),
              extensionId.getExtensionName(),
              sampleUsageLocation),
          Transience.PERSISTENT);
    }

    // Run that extension!
    ModuleExtension extension = ((ModuleExtension.InStarlark) exported).get();
    ModuleExtensionEvalStarlarkThreadContext threadContext =
        new ModuleExtensionEvalStarlarkThreadContext(
            usagesValue.getExtensionUniqueName() + ".",
            extensionId.getBzlFileLabel().getPackageIdentifier(),
            BazelModuleContext.of(bzlLoadValue.getModule()).repoMapping(),
            directories,
            env.getListener());
    try (Mutability mu =
        Mutability.create("module extension", usagesValue.getExtensionUniqueName())) {
      StarlarkThread thread = new StarlarkThread(mu, starlarkSemantics);
      ModuleExtensionContext moduleContext =
          createContext(env, usagesValue, starlarkSemantics, extension);
      threadContext.storeInThread(thread);
      // TODO(wyv): support native.register_toolchains
      // TODO(wyv): make sure the Label constructor works
      try {
        Starlark.fastcall(
            thread, extension.getImplementation(), new Object[] {moduleContext}, new Object[0]);
      } catch (NeedsSkyframeRestartException e) {
        // Clean up and restart by returning null.
        try {
          if (moduleContext.getWorkingDirectory().exists()) {
            moduleContext.getWorkingDirectory().deleteTree();
          }
        } catch (IOException e1) {
          throw new SingleExtensionEvalFunctionException(e1, Transience.TRANSIENT);
        }
        return null;
      } catch (EvalException e) {
        env.getListener().handle(Event.error(e.getMessageWithStack()));
        throw new SingleExtensionEvalFunctionException(
            ExternalDepsException.withMessage(
                Code.BAD_MODULE,
                "error evaluating module extension %s in %s",
                extensionId.getExtensionName(),
                extensionId.getBzlFileLabel()),
            Transience.TRANSIENT);
      }
    }

    // TODO(wyv): check that all used repos have been generated (for better error reporting). Or...
    //   somehow check it at use time?

    return SingleExtensionEvalValue.create(threadContext.getGeneratedRepos());
  }

  private ModuleExtensionContext createContext(
      Environment env,
      SingleExtensionUsagesValue usagesValue,
      StarlarkSemantics starlarkSemantics,
      ModuleExtension extension)
      throws SingleExtensionEvalFunctionException {
    Path workingDirectory =
        directories
            .getOutputBase()
            .getRelative(LabelConstants.MODULE_EXTENSION_WORKING_DIRECTORY_LOCATION)
            .getRelative(usagesValue.getExtensionUniqueName());
    ArrayList<StarlarkBazelModule> modules = new ArrayList<>();
    for (AbridgedModule abridgedModule : usagesValue.getAbridgedModules()) {
      ModuleKey moduleKey = abridgedModule.getKey();
      try {
        modules.add(
            StarlarkBazelModule.create(
                abridgedModule,
                extension,
                usagesValue.getRepoMappings().get(moduleKey),
                usagesValue.getExtensionUsages().get(moduleKey)));
      } catch (ExternalDepsException e) {
        throw new SingleExtensionEvalFunctionException(e, Transience.PERSISTENT);
      }
    }
    return new ModuleExtensionContext(
        workingDirectory,
        env,
        clientEnvironmentSupplier.get(),
        downloadManager,
        timeoutScaling,
        processWrapper,
        starlarkSemantics,
        repositoryRemoteExecutor,
        StarlarkList.immutableCopyOf(modules));
  }

  static final class SingleExtensionEvalFunctionException extends SkyFunctionException {

    SingleExtensionEvalFunctionException(Exception cause, Transience transience) {
      super(cause, transience);
    }
  }
}
