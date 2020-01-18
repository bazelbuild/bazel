// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules.android;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ExecutionRequirements;
import com.google.devtools.build.lib.actions.ParamFileInfo;
import com.google.devtools.build.lib.actions.ParameterFile.ParameterFileType;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine.VectorArg;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.util.OS;
import com.google.errorprone.annotations.CompileTimeConstant;
import java.util.Collection;
import java.util.List;
import javax.annotation.Nullable;

/** Builder for actions that invoke the Android BusyBox. */
public final class BusyBoxActionBuilder {

  // Some flags (e.g. --mainData) may specify lists (or lists of lists) separated by special
  // characters (colon, semicolon, hashmark, ampersand) that don't work on Windows, and quoting
  // semantics are very complicated (more so than in Bash), so let's just always use a parameter
  // file.
  // TODO(laszlocsomor), TODO(corysmith): restructure the Android BusyBux's flags by deprecating
  // list-type and list-of-list-type flags that use such problematic separators in favor of
  // multi-value flags (to remove one level of listing) and by changing all list separators to a
  // platform-safe character (= comma).
  private static final ParamFileInfo FORCED_PARAM_FILE_INFO =
      ParamFileInfo.builder(ParameterFileType.SHELL_QUOTED)
          .setUseAlways(OS.getCurrent() == OS.WINDOWS)
          .build();

  private static final ParamFileInfo WORKERS_FORCED_PARAM_FILE_INFO =
      ParamFileInfo.builder(ParameterFileType.UNQUOTED)
          .setUseAlways(true)
          .build();

  private final AndroidDataContext dataContext;
  private final NestedSetBuilder<Artifact> inputs = NestedSetBuilder.naiveLinkOrder();
  private final ImmutableList.Builder<Artifact> outputs = ImmutableList.builder();
  private final SpawnAction.Builder spawnActionBuilder = new SpawnAction.Builder();
  private final CustomCommandLine.Builder commandLine = CustomCommandLine.builder();

  public static BusyBoxActionBuilder create(
      AndroidDataContext dataContext, @CompileTimeConstant String toolName) {
    BusyBoxActionBuilder builder = new BusyBoxActionBuilder(dataContext);
    builder.commandLine.add("--tool").add(toolName).add("--");
    return builder;
  }

  private BusyBoxActionBuilder(AndroidDataContext dataContext) {
    this.dataContext = dataContext;
  }

  /** Adds a direct input artifact. */
  public BusyBoxActionBuilder addInput(Artifact value) {
    Preconditions.checkNotNull(value);
    commandLine.addExecPath(value);
    inputs.add(value);
    return this;
  }

  /** Adds a direct input artifact. */
  public BusyBoxActionBuilder addInput(@CompileTimeConstant String arg, Artifact value) {
    Preconditions.checkNotNull(value);
    commandLine.addExecPath(arg, value);
    inputs.add(value);
    return this;
  }

  /**
   * Adds a series of direct input artifacts.
   *
   * <p>For efficiency, when adding a NestedSet of artifacts, use one of the transitive methods,
   * such as {@link #addTransitiveFlag(String, NestedSet, AndroidDataConverter)} and {@link
   * #addTransitiveInputValues(NestedSet)}, instead.
   *
   * @param value a string representation of the value artifacts
   */
  public BusyBoxActionBuilder addInput(
      @CompileTimeConstant String arg, String value, Iterable<Artifact> valueArtifacts) {
    commandLine.add(arg, value);
    inputs.addAll(valueArtifacts);
    return this;
  }

  /** Adds the given input artifacts without any command line options. */
  public BusyBoxActionBuilder addInputs(Iterable<Artifact> inputs) {
    this.inputs.addAll(inputs);
    return this;
  }

  /** Adds an input artifact if it is non-null */
  public BusyBoxActionBuilder maybeAddInput(
      @CompileTimeConstant String arg, @Nullable Artifact value) {
    if (value != null) {
      addInput(arg, value);
    }
    return this;
  }

  /** Adds an input artifact if it is non-null */
  public BusyBoxActionBuilder maybeAddInput(@Nullable Artifact value) {
    if (value != null) {
      this.inputs.add(value);
    }
    return this;
  }

  /**
   * Adds a series of direct input artifacts if the list containing them is not null or empty.
   *
   * <p>For efficiency, when adding a NestedSet of artifacts, use one of the transitive methods,
   * such as {@link #addTransitiveFlag(String, NestedSet, AndroidDataConverter)} and {@link
   * #addTransitiveInputValues(NestedSet)}, instead.
   */
  public BusyBoxActionBuilder maybeAddInput(
      @CompileTimeConstant String arg, @Nullable Collection<Artifact> values) {
    if (values != null && !values.isEmpty()) {
      commandLine.addExecPaths(arg, values);
      inputs.addAll(values);
    }
    return this;
  }

  /**
   * Adds a series of direct input artifacts if the list containing them is not null or empty.
   *
   * <p>For efficiency, when adding a NestedSet of artifacts, use one of the transitive methods,
   * such as {@link #addTransitiveFlag(String, NestedSet, AndroidDataConverter)} and {@link
   * #addTransitiveInputValues(NestedSet)}, instead.
   *
   * @param value a string representation of the value artifacts
   */
  public BusyBoxActionBuilder maybeAddInput(
      @CompileTimeConstant String arg,
      String value,
      @Nullable Collection<Artifact> valueArtifacts) {
    if (valueArtifacts != null && !valueArtifacts.isEmpty()) {
      addInput(arg, value, valueArtifacts);
    }
    return this;
  }

  /** Adds an output artifact */
  public BusyBoxActionBuilder addOutput(@CompileTimeConstant String arg, Artifact value) {
    Preconditions.checkNotNull(value);
    commandLine.addExecPath(arg, value);
    outputs.add(value);
    return this;
  }

  /** Adds the given output artifacts without adding any command line options. */
  public BusyBoxActionBuilder addOutputs(Iterable<Artifact> outputs) {
    this.outputs.addAll(outputs);
    return this;
  }

  /** Adds an output artifact if it is non-null */
  public BusyBoxActionBuilder maybeAddOutput(
      @CompileTimeConstant String arg, @Nullable Artifact value) {
    if (value != null) {
      return addOutput(arg, value);
    }
    return this;
  }

  /**
   * Adds a series of transitive input artifacts.
   *
   * <p>These artifacts will not be mentioned on the command line - use {@link
   * #addTransitiveFlag(String, NestedSet, AndroidDataConverter)} for that.
   */
  public BusyBoxActionBuilder addTransitiveInputValues(NestedSet<Artifact> values) {
    inputs.addTransitive(values);
    return this;
  }

  /**
   * Adds an efficient flag based on transitive values.
   *
   * <p>The flag will only be specified once, followed by the joined values specified by the
   * converter, for example: --flag value1,value2
   *
   * <p>The values will only be collapsed and turned into a flag at execution time.
   *
   * <p>The values will not be added as inputs - use {@link #addTransitiveInputValues(NestedSet)}
   * for that.
   */
  public <T> BusyBoxActionBuilder addTransitiveFlag(
      @CompileTimeConstant String arg,
      NestedSet<? extends T> transitiveValues,
      AndroidDataConverter<T> converter) {
    commandLine.addAll(arg, converter.getVectorArg(transitiveValues));
    return this;
  }

  /**
   * Adds an efficient flag based on transitive values.
   *
   * <p>Each transitive value, as created using the converter, will be proceeded by the flag, for
   * example: --flag value1 --flag value2
   *
   * <p>The values will only be collapsed and turned into a flag at execution time.
   *
   * <p>The values will not be added as inputs - use {@link #addTransitiveInputValues(NestedSet)}
   * for that.
   */
  public <T> BusyBoxActionBuilder addTransitiveFlagForEach(
      @CompileTimeConstant String arg,
      NestedSet<? extends T> transitiveValues,
      AndroidDataConverter<T> converter) {
    commandLine.addAll(converter.getVectorArgForEach(arg, transitiveValues));
    return this;
  }

  /**
   * Adds an efficient flag and inputs based on transitive values.
   *
   * <p>Each value will be separated on the command line by the ':' character, the option parser's
   * PathListConverter delimiter.
   *
   * <p>Unlike other transitive input methods in this class, this method adds the values to both the
   * command line and the list of inputs.
   */
  public BusyBoxActionBuilder addTransitiveVectoredInput(
      @CompileTimeConstant String arg, NestedSet<Artifact> values) {
    commandLine.addExecPaths(arg, VectorArg.join(":").each(values));
    inputs.addTransitive(values);
    return this;
  }

  /** Adds a flag with a value set to the current target's label */
  public BusyBoxActionBuilder addLabelFlag(@CompileTimeConstant String arg) {
    commandLine.addLabel(arg, dataContext.getLabel());
    return this;
  }

  /** Adds a flag with no arguments to the command line. */
  public BusyBoxActionBuilder addFlag(@CompileTimeConstant String value) {
    commandLine.add(value);
    return this;
  }

  /** Adds a flag with a String value to the command line. */
  public BusyBoxActionBuilder addFlag(@CompileTimeConstant String arg, String value) {
    Preconditions.checkNotNull(value);
    commandLine.add(arg, value);
    return this;
  }

  /** If the condition is true, adds a flag with no arguments to the command line. */
  public BusyBoxActionBuilder maybeAddFlag(@CompileTimeConstant String arg, boolean condition) {
    if (condition) {
      commandLine.add(arg);
    }
    return this;
  }

  /** If the flag is a non-null, non-empty String, adds the flag and value to the command line. */
  public BusyBoxActionBuilder maybeAddFlag(
      @CompileTimeConstant String arg, @Nullable String value) {
    if (value != null && !value.isEmpty()) {
      addFlag(arg, value);
    }
    return this;
  }

  /**
   * Efficiently adds a flag and a list of values to the command line.
   *
   * <p>The values will be joined in execution and separated by commas.
   */
  public BusyBoxActionBuilder addVectoredFlag(
      @CompileTimeConstant String arg, List<String> values) {
    Preconditions.checkNotNull(values);
    commandLine.addAll(arg, VectorArg.join(",").each(values));

    return this;
  }

  /**
   * If the values are not null or empty, efficiently adds a flag with them to the command line.
   *
   * <p>The values will be joined in execution and separated by commas.
   */
  public BusyBoxActionBuilder maybeAddVectoredFlag(
      @CompileTimeConstant String arg, @Nullable List<String> values) {
    if (values != null && !values.isEmpty()) {
      addVectoredFlag(arg, values);
    }
    return this;
  }

  /** Adds aapt to the command line and inputs. */
  public BusyBoxActionBuilder addAapt() {
    FilesToRunProvider aapt2 = dataContext.getSdk().getAapt2();
    commandLine.addExecPath("--aapt2", aapt2.getExecutable());
    spawnActionBuilder.addTool(aapt2);
    return this;
  }

  /** Adds the Android JAR from the SDK to the command line and inputs */
  public BusyBoxActionBuilder addAndroidJar() {
    return addInput("--androidJar", dataContext.getSdk().getAndroidJar());
  }

  /**
   * Builds and registers this action.
   *
   * @param message a progress message (visible in Bazel output), for example "Running tool". The
   *     current label will be appended to this message.
   * @param mnemonic a mnemonic used to indicate the tool being run, for example, "BusyBoxTool".
   */
  public void buildAndRegister(String message, String mnemonic) {
    spawnActionBuilder
        .useDefaultShellEnvironment()
        .addTransitiveInputs(inputs.build())
        .addOutputs(outputs.build())
        .setExecutable(dataContext.getBusybox())
        .setProgressMessage("%s for %s", message, dataContext.getLabel())
        .setMnemonic(mnemonic);

    if (dataContext.isPersistentBusyboxToolsEnabled()) {
      spawnActionBuilder
          .setExecutionInfo(ExecutionRequirements.WORKER_MODE_ENABLED)
          .addCommandLine(commandLine.build(), WORKERS_FORCED_PARAM_FILE_INFO);
    } else {
      spawnActionBuilder.addCommandLine(commandLine.build(), FORCED_PARAM_FILE_INFO);
    }

    dataContext.registerAction(spawnActionBuilder);
  }
}
