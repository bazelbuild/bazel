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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ParamFileInfo;
import com.google.devtools.build.lib.actions.ParameterFile.ParameterFileType;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.util.OS;
import com.google.errorprone.annotations.CompileTimeConstant;

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

  private final RuleContext ruleContext;
  private final NestedSetBuilder<Artifact> inputs = NestedSetBuilder.naiveLinkOrder();
  private final ImmutableList.Builder<Artifact> outputs = ImmutableList.builder();
  private final CustomCommandLine.Builder commandLine = CustomCommandLine.builder();

  public static BusyBoxActionBuilder create(
      RuleContext ruleContext, @CompileTimeConstant String toolName) {
    BusyBoxActionBuilder builder = new BusyBoxActionBuilder(ruleContext);
    builder.commandLine.add("--tool").add(toolName).add("--");
    return builder;
  }

  private BusyBoxActionBuilder(RuleContext ruleContext) {
    this.ruleContext = ruleContext;
  }

  public BusyBoxActionBuilder addInput(@CompileTimeConstant String arg, Artifact value) {
    commandLine.addExecPath(arg, value);
    inputs.add(value);
    return this;
  }

  /**
   * Adds a series of input artifacts. For efficiency, when adding a NestedSet of artifacts, use
   * {@link #addTransitiveFlag(String, NestedSet, AndroidDataConverter)} and {@link
   * #addTransitiveInputValues(NestedSet)} instead.
   */
  public BusyBoxActionBuilder addInput(
      @CompileTimeConstant String arg, String value, Iterable<Artifact> valueArtifacts) {
    commandLine.add(arg, value);
    inputs.addAll(valueArtifacts);
    return this;
  }

  public BusyBoxActionBuilder addOutput(@CompileTimeConstant String arg, Artifact value) {
    commandLine.addExecPath(arg, value);
    outputs.add(value);
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

  public BusyBoxActionBuilder addFlag(@CompileTimeConstant String value) {
    commandLine.add(value);
    return this;
  }

  /**
   * Builds and registers this action.
   *
   * @param message a progress message (visible in Bazel output), for example "Running tool". The
   *     current label will be appended to this message.
   * @param mnemonic a mnemonic used to indicate the tool being run, for example, "BusyBoxTool".
   */
  public void buildAndRegister(String message, String mnemonic) {
    ruleContext.registerAction(
        new SpawnAction.Builder()
            .useDefaultShellEnvironment()
            .addTransitiveInputs(inputs.build())
            .addOutputs(outputs.build())
            .addCommandLine(commandLine.build(), FORCED_PARAM_FILE_INFO)
            .setExecutable(
                ruleContext.getExecutablePrerequisite("$android_resources_busybox", Mode.HOST))
            .setProgressMessage("%s for %s", message, ruleContext.getLabel())
            .setMnemonic(mnemonic)
            .build(ruleContext));
  }
}
