// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.java;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.actions.AbstractAction;
import com.google.devtools.build.lib.actions.ActionEnvironment;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.ActionResult;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.BaseSpawn;
import com.google.devtools.build.lib.actions.CommandAction;
import com.google.devtools.build.lib.actions.CommandLineExpansionException;
import com.google.devtools.build.lib.actions.CommandLines;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.ExecutionInfoSpecifier;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.actions.RunfilesSupplier;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnActionContext;
import com.google.devtools.build.lib.actions.extra.ExtraActionInfo;
import com.google.devtools.build.lib.analysis.actions.SpawnAction.ExtraActionInfoSupplier;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.collect.IterablesChain;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadCompatible;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.rules.java.JavaPluginInfoProvider.JavaPluginInfo;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.util.Fingerprint;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

/** Action that represents a Java compilation. */
@ThreadCompatible
@Immutable
public class JavaCompileAction extends AbstractAction
    implements ExecutionInfoSpecifier, CommandAction {
  private static final String MNEMONIC = "Javac";
  private static final ResourceSet LOCAL_RESOURCES =
      ResourceSet.createWithRamCpu(/* memoryMb= */ 750, /* cpuUsage= */ 1);
  private static final UUID GUID = UUID.fromString("e423747c-2827-49e6-b961-f6c08c10bb51");

  private final ImmutableMap<String, String> executionInfo;
  private final CommandLines commandLines;
  private final BuildConfiguration configuration;

  private final ImmutableSet<Artifact> sourceFiles;
  private final ImmutableList<Artifact> sourceJars;
  private final JavaPluginInfo plugins;

  private final ExtraActionInfoSupplier extraActionInfoSupplier;

  public JavaCompileAction(
      ActionOwner owner,
      ActionEnvironment env,
      NestedSet<Artifact> tools,
      RunfilesSupplier runfilesSupplier,
      ImmutableSet<Artifact> sourceFiles,
      ImmutableList<Artifact> sourceJars,
      JavaPluginInfo plugins,
      NestedSet<Artifact> mandatoryInputs,
      NestedSet<Artifact> transitiveInputs,
      NestedSet<Artifact> outputs,
      ImmutableMap<String, String> executionInfo,
      ExtraActionInfoSupplier extraActionInfoSupplier,
      CommandLines commandLines,
      BuildConfiguration configuration) {
    super(
        owner,
        tools,
        IterablesChain.concat(mandatoryInputs, transitiveInputs),
        runfilesSupplier,
        outputs,
        env);
    // TODO(djasper): The only thing that is conveyed through the executionInfo is whether worker
    // mode is enabled or not. Investigate whether we can store just that.
    this.executionInfo = configuration.modifiedExecutionInfo(executionInfo, MNEMONIC);
    this.commandLines = commandLines;
    this.configuration = configuration;
    this.sourceFiles = sourceFiles;
    this.sourceJars = sourceJars;
    this.plugins = plugins;
    this.extraActionInfoSupplier = extraActionInfoSupplier;
  }

  @Override
  public String getMnemonic() {
    return MNEMONIC;
  }

  @Override
  protected void computeKey(ActionKeyContext actionKeyContext, Fingerprint fp)
      throws CommandLineExpansionException {
    fp.addUUID(GUID);
    commandLines.addToFingerprint(actionKeyContext, fp);
    // We don't need the toolManifests here, because they are a subset of the inputManifests by
    // definition and the output of an action shouldn't change whether something is considered a
    // tool or not.
    fp.addPaths(getRunfilesSupplier().getRunfilesDirs());
    ImmutableList<Artifact> runfilesManifests = getRunfilesSupplier().getManifests();
    fp.addInt(runfilesManifests.size());
    for (Artifact runfilesManifest : runfilesManifests) {
      fp.addPath(runfilesManifest.getExecPath());
    }
    env.addTo(fp);
    fp.addStringMap(executionInfo);
  }

  @Override
  public ActionResult execute(ActionExecutionContext actionExecutionContext)
      throws ActionExecutionException, InterruptedException {
    SpawnActionContext spawnActionContext =
        actionExecutionContext.getContext(SpawnActionContext.class);
    try {
      CommandLines.ExpandedCommandLines expandedCommandLines =
          commandLines.expand(
              actionExecutionContext.getArtifactExpander(),
              getPrimaryOutput().getExecPath(),
              configuration.getCommandLineLimits());
      LinkedHashMap<String, String> effectiveEnvironment =
          Maps.newLinkedHashMapWithExpectedSize(env.size());
      env.resolve(effectiveEnvironment, actionExecutionContext.getClientEnv());
      Spawn spawn =
          new JavaSpawn(
              expandedCommandLines, ImmutableMap.copyOf(effectiveEnvironment), executionInfo);
      return ActionResult.create(spawnActionContext.exec(spawn, actionExecutionContext));
    } catch (ExecException e) {
      throw e.toActionExecutionException(
          getRawProgressMessage(), actionExecutionContext.getVerboseFailures(), this);
    } catch (CommandLineExpansionException e) {
      throw new ActionExecutionException(e, this, false);
    }
  }

  @Override
  protected String getRawProgressMessage() {
    StringBuilder sb = new StringBuilder("Building ");
    sb.append(getPrimaryOutput().prettyPrint());
    sb.append(" (");
    boolean first = true;
    first = appendCount(sb, first, sourceFiles.size(), "source file");
    first = appendCount(sb, first, sourceJars.size(), "source jar");
    sb.append(")");
    sb.append(getProcessorNames(plugins.processorClasses()));
    return sb.toString();
  }

  @Override
  public ExtraActionInfo.Builder getExtraActionInfo(ActionKeyContext actionKeyContext)
      throws CommandLineExpansionException {
    ExtraActionInfo.Builder builder = super.getExtraActionInfo(actionKeyContext);
    extraActionInfoSupplier.extend(builder);
    return builder;
  }

  private static String getProcessorNames(NestedSet<String> processorClasses) {
    if (processorClasses.isEmpty()) {
      return "";
    }
    StringBuilder sb = new StringBuilder();
    List<String> shortNames = new ArrayList<>();
    for (String name : processorClasses) {
      // Annotation processor names are qualified class names. Omit the package part for the
      // progress message, e.g. `com.google.Foo` -> `Foo`.
      int idx = name.lastIndexOf('.');
      String shortName = idx != -1 ? name.substring(idx + 1) : name;
      shortNames.add(shortName);
    }
    sb.append(" and running annotation processors (");
    Joiner.on(", ").appendTo(sb, shortNames);
    sb.append(")");
    return sb.toString();
  }

  /**
   * Append an input count to the progress message, e.g. "2 source jars". If an input count has
   * already been appended, prefix with ", ".
   */
  private static boolean appendCount(StringBuilder sb, boolean first, int count, String name) {
    if (count > 0) {
      if (!first) {
        sb.append(", ");
      } else {
        first = false;
      }
      sb.append(count).append(' ').append(name);
      if (count > 1) {
        sb.append('s');
      }
    }
    return first;
  }

  private final class JavaSpawn extends BaseSpawn {
    final Iterable<ActionInput> inputs;

    public JavaSpawn(
        CommandLines.ExpandedCommandLines expandedCommandLines,
        Map<String, String> environment,
        Map<String, String> executionInfo) {
      super(
          ImmutableList.copyOf(expandedCommandLines.arguments()),
          environment,
          executionInfo,
          JavaCompileAction.this,
          LOCAL_RESOURCES);
      inputs = Iterables.concat(getInputs(), expandedCommandLines.getParamFiles());
    }

    @Override
    @SuppressWarnings("unchecked")
    public Iterable<? extends ActionInput> getInputFiles() {
      return inputs;
    }
  }

  @VisibleForTesting
  CommandLines getCommandLines() {
    return commandLines;
  }

  @Override
  public SkylarkList<String> getSkylarkArgv() throws EvalException {
    try {
      return SkylarkList.createImmutable(getArguments());
    } catch (CommandLineExpansionException exception) {
      throw new EvalException(Location.BUILTIN, exception);
    }
  }

  /** Returns the out-of-band execution data for this action. */
  @Override
  public Map<String, String> getExecutionInfo() {
    return executionInfo;
  }

  @Override
  public List<String> getArguments() throws CommandLineExpansionException {
    return ImmutableList.copyOf(commandLines.allArguments());
  }

  @Override
  @VisibleForTesting
  public final ImmutableMap<String, String> getIncompleteEnvironmentForTesting() {
    // TODO(ulfjack): AbstractAction should declare getEnvironment with a return value of type
    // ActionEnvironment to avoid developers misunderstanding the purpose of this method. That
    // requires first updating all subclasses and callers to actually handle environments correctly,
    // so it's not a small change.
    return env.getFixedEnv().toMap();
  }

  @Override
  public Iterable<Artifact> getPossibleInputsForTesting() {
    return null;
  }
}
