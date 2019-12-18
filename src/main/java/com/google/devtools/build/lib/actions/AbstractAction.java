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

package com.google.devtools.build.lib.actions;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.extra.ExtraActionInfo;
import com.google.devtools.build.lib.analysis.platform.PlatformInfo;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.CollectionUtils;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.AspectDescriptor;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.VisibleForSerialization;
import com.google.devtools.build.lib.skylarkbuildapi.ActionApi;
import com.google.devtools.build.lib.skylarkbuildapi.CommandLineArgsApi;
import com.google.devtools.build.lib.syntax.Depset;
import com.google.devtools.build.lib.syntax.Dict;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Printer;
import com.google.devtools.build.lib.syntax.Sequence;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.Symlinks;
import java.io.IOException;
import java.util.Collection;
import java.util.Map;
import javax.annotation.Nullable;
import javax.annotation.concurrent.GuardedBy;

/**
 * Abstract implementation of Action which implements basic functionality: the inputs, outputs, and
 * toString method. Both input and output sets are immutable. Subclasses must be generally immutable
 * - see the documentation on {@link Action}.
 */
@Immutable
@ThreadSafe
public abstract class AbstractAction extends ActionKeyCacher implements Action, ActionApi {
  /**
   * An arbitrary default resource set. We assume that a typical subprocess is single-threaded
   * (i.e., uses one CPU core) and CPU-bound, and uses a small-ish amount of memory. In the past,
   * we've seen that assuming less than one core can lead to local overload. Unless you have data
   * indicating otherwise (for example, we've observed in the past that C++ linking can use large
   * amounts of memory), we suggest to use this default set.
   */
  // TODO(ulfjack): Collect actual data to confirm that this is an acceptable approximation.
  public static final ResourceSet DEFAULT_RESOURCE_SET = ResourceSet.createWithRamCpu(250, 1);

  /**
   * The owner/inputs/outputs attributes below should never be directly accessed even within
   * AbstractAction itself. The appropriate getter methods should be used instead. This has to be
   * done due to the fact that the getter methods can be overridden in subclasses.
   */
  @VisibleForSerialization protected final ActionOwner owner;

  /**
   * Tools are a subset of inputs and used by the WorkerSpawnStrategy to determine whether a
   * compiler has changed since the last time it was used. This should include all artifacts that
   * the tool does not dynamically reload / check on each unit of work - e.g. its own binary, the
   * JDK for Java binaries, shared libraries, ... but not a configuration file, if it reloads that
   * when it has changed.
   *
   * <p>If the "tools" set does not contain exactly the right set of artifacts, the following can
   * happen: If an artifact that should be included is missing, the tool might not be restarted when
   * it should, and builds can become incorrect (example: The compiler binary is not part of this
   * set, then the compiler gets upgraded, but the worker strategy still reuses the old version).
   * If an artifact that should *not* be included is accidentally part of this set, the worker
   * process will be restarted more often that is necessary - e.g. if a file that is unique to each
   * unit of work, e.g. the source code that a compiler should compile for a compile action, is
   * part of this set, then the worker will never be reused and will be restarted for each unit of
   * work.
   */
  private final Iterable<Artifact> tools;

  @GuardedBy("this")
  private boolean inputsDiscovered = false;  // Only used when discoversInputs() returns true

  // The variable inputs is non-final only so that actions that discover their inputs can modify it.
  @GuardedBy("this")
  @VisibleForSerialization
  protected Iterable<Artifact> inputs;

  protected final ActionEnvironment env;
  private final RunfilesSupplier runfilesSupplier;
  @VisibleForSerialization protected final ImmutableSet<Artifact> outputs;

  /**
   * Construct an abstract action with the specified inputs and outputs;
   */
  protected AbstractAction(
      ActionOwner owner,
      Iterable<Artifact> inputs,
      Iterable<Artifact> outputs) {
    this(
        owner,
        /*tools = */ImmutableList.of(),
        inputs,
        EmptyRunfilesSupplier.INSTANCE,
        outputs,
        ActionEnvironment.EMPTY);
  }

  protected AbstractAction(
      ActionOwner owner,
      Iterable<Artifact> inputs,
      Iterable<Artifact> outputs,
      ActionEnvironment env) {
    this(
        owner,
        /*tools = */ImmutableList.of(),
        inputs,
        EmptyRunfilesSupplier.INSTANCE,
        outputs,
        env);
  }

  protected AbstractAction(
      ActionOwner owner,
      Iterable<Artifact> tools,
      Iterable<Artifact> inputs,
      RunfilesSupplier runfilesSupplier,
      Iterable<? extends Artifact> outputs,
      ActionEnvironment env) {
    Preconditions.checkNotNull(owner);
    this.owner = owner;
    this.tools = CollectionUtils.makeImmutable(tools);
    this.inputs = CollectionUtils.makeImmutable(inputs);
    this.env = Preconditions.checkNotNull(env);
    this.outputs = ImmutableSet.copyOf(outputs);
    this.runfilesSupplier = Preconditions.checkNotNull(runfilesSupplier);
    Preconditions.checkArgument(!this.outputs.isEmpty(), "action outputs may not be empty");
  }

  @Override
  public final ActionOwner getOwner() {
    return owner;
  }

  @Override
  public final synchronized boolean inputsDiscovered() {
    return discoversInputs() ? inputsDiscovered : true;
  }

  /**
   * Should be overridden by actions that do input discovery.
   *
   * <p>The value returned by each instance should be constant over the lifetime of that instance.
   *
   * <p>If this returns true, {@link #discoverInputs(ActionExecutionContext)} must also be
   * implemented.
   */
  @Override
  public boolean discoversInputs() {
    return false;
  }

  /**
   * Run input discovery on the action.
   *
   * <p>Called by Blaze if {@link #discoversInputs()} returns true. It must return the set of
   * input artifacts that were not known at analysis time. May also call
   * {@link #updateInputs(Iterable<Artifact>)}; if it doesn't, the action itself must arrange for
   * the newly discovered artifacts to be available during action execution, probably by keeping
   * state in the action instance and using a custom action execution context and for
   * {@code #updateInputs()} to be called during the execution of the action.
   *
   * <p>Since keeping state within an action bad, don't do that unless there is a very good reason
   * to do so.
   */
  @Override
  public Iterable<Artifact> discoverInputs(ActionExecutionContext actionExecutionContext)
      throws ActionExecutionException, InterruptedException {
    throw new IllegalStateException("discoverInputs cannot be called for " + this.prettyPrint()
        + " since it does not discover inputs");
  }

  @Override
  public Iterable<Artifact> getAllowedDerivedInputs() {
    throw new IllegalStateException(
        "Method must be overridden for actions that may have unknown inputs.");
  }

  /**
   * Should be called when the inputs of the action become known, that is, either during {@link
   * #discoverInputs(ActionExecutionContext)} or during {@link #execute(ActionExecutionContext)}.
   *
   * <p>When an action discovers inputs, it must have been called by the time {@code #execute()}
   * returns. It can be called both during {@code discoverInputs} and during {@code execute()}.
   *
   * <p>In addition to being called from action implementations, it will also be called by Bazel
   * itself when an action is loaded from the on-disk action cache.
   */
  @Override
  public synchronized void updateInputs(Iterable<Artifact> inputs) {
    Preconditions.checkState(
        discoversInputs(), "Can't update inputs unless discovering: %s %s", this, inputs);
    this.inputs = CollectionUtils.makeImmutable(inputs);
    inputsDiscovered = true;
  }

  @Override
  public Iterable<Artifact> getTools() {
    return tools;
  }

  /**
   * Should not be overridden (it's non-final only for tests)
   */
  @Override
  public synchronized Iterable<Artifact> getInputs() {
    return inputs;
  }

  public final ActionEnvironment getEnvironment() {
    return env;
  }

  @Override
  public Iterable<String> getClientEnvironmentVariables() {
    return env.getInheritedEnv();
  }

  @Override
  public RunfilesSupplier getRunfilesSupplier() {
    return runfilesSupplier;
  }

  @Override
  public ImmutableSet<Artifact> getOutputs() {
    return outputs;
  }

  @Override
  public Artifact getPrimaryInput() {
    // The default behavior is to return the first input artifact.
    // Call through the method, not the field, because it may be overridden.
    return Iterables.getFirst(getInputs(), null);
  }

  @Override
  public Artifact getPrimaryOutput() {
    // Default behavior is to return the first output artifact.
    // Use the method rather than field in case of overriding in subclasses.
    return Iterables.getFirst(getOutputs(), null);
  }

  @Override
  public Iterable<Artifact> getMandatoryInputs() {
    return getInputs();
  }

  @Override
  public String toString() {
    return prettyPrint() + " (" + getMnemonic() + "[" + ImmutableList.copyOf(getInputs())
        + (inputsDiscovered() ? " -> " : ", unknown inputs -> ")
        + getOutputs() + "]" + ")";
  }

  @Override
  public abstract String getMnemonic();

  @Override
  public String describeKey() {
    return null;
  }

  @Override
  public boolean executeUnconditionally() {
    return false;
  }

  @Override
  public boolean isVolatile() {
    return false;
  }

  @Override
  public boolean isShareable() {
    return true;
  }

  @Override
  public boolean showsOutputUnconditionally() {
    return false;
  }

  @Override
  public final String getProgressMessage() {
    String message = getRawProgressMessage();
    if (message == null) {
      return null;
    }
    String additionalInfo = getOwner().getAdditionalProgressInfo();
    return additionalInfo == null ? message : message + " [" + additionalInfo + "]";
  }

  /**
   * Returns a progress message string that is specific for this action. This is
   * then annotated with additional information, currently the string '[for host]'
   * for actions in the host configurations.
   *
   * <p>A return value of null indicates no message should be reported.
   */
  protected String getRawProgressMessage() {
    // A cheesy default implementation.  Subclasses are invited to do something
    // more meaningful.
    return defaultProgressMessage();
  }

  private String defaultProgressMessage() {
    return getMnemonic() + " " + getPrimaryOutput().prettyPrint();
  }

  @Override
  public String prettyPrint() {
    return "action '" + describe() + "'";
  }

  @Override
  public void repr(Printer printer) {
    printer.append(prettyPrint()); // TODO(bazel-team): implement a readable representation
  }

  /**
   * Deletes all of the action's output files, if they exist. If any of the Artifacts refers to a
   * directory recursively removes the contents of the directory.
   *
   * @param execRoot the exec root in which this action is executed
   */
  protected void deleteOutputs(Path execRoot) throws IOException {
    for (Artifact output : getOutputs()) {
      deleteOutput(output.getPath(), output.getRoot());
    }
  }

  /**
   * Helper method to remove an output file.
   *
   * <p>If the path refers to a directory, recursively removes the contents of the directory.
   *
   * @param path the output to remove
   * @param root the root containing the output. This is used to sanity-check that we don't delete
   *     arbitrary files in the file system.
   */
  public static void deleteOutput(Path path, @Nullable ArtifactRoot root) throws IOException {
    try {
      // Optimize for the common case: output artifacts are files.
      path.delete();
    } catch (IOException e) {
      // Handle a couple of scenarios where the output can still be deleted, but make sure we're not
      // deleting random files on the filesystem.
      if (root == null) {
        throw new IOException(e);
      }
      Root outputRoot = root.getRoot();
      if (!outputRoot.contains(path)) {
        throw new IOException(e);
      }

      Path parentDir = path.getParentDirectory();
      if (!parentDir.isWritable() && outputRoot.contains(parentDir)) {
        // Retry deleting after making the parent writable.
        parentDir.setWritable(true);
        deleteOutput(path, root);
      } else if (path.isDirectory(Symlinks.NOFOLLOW)) {
        path.deleteTree();
      } else {
        throw new IOException(e);
      }
    }
  }

  /**
   * If the action might read directories as inputs in a way that is unsound wrt dependency
   * checking, this method must be called.
   */
  protected void checkInputsForDirectories(
      EventHandler eventHandler, MetadataProvider metadataProvider) throws IOException {
    // Report "directory dependency checking" warning only for non-generated directories (generated
    // ones will be reported earlier).
    for (Artifact input : getMandatoryInputs()) {
      // Assume that if the file did not exist, we would not have gotten here.
      if (input.isSourceArtifact() && metadataProvider.getMetadata(input).getType().isDirectory()) {
        // TODO(ulfjack): What about dependency checking of special files?
        eventHandler.handle(
            Event.warn(
                getOwner().getLocation(),
                String.format(
                    "input '%s' to %s is a directory; "
                        + "dependency checking of directories is unsound",
                    input.prettyPrint(), getOwner().getLabel())));
      }
    }
  }

  @Override
  public MiddlemanType getActionType() {
    return MiddlemanType.NORMAL;
  }

  /** If the action might create directories as outputs this method must be called. */
  protected void checkOutputsForDirectories(ActionExecutionContext actionExecutionContext) {
    for (Artifact output : getOutputs()) {
      Path path = actionExecutionContext.getInputPath(output);
      String ownerString = Label.print(getOwner().getLabel());
      if (path.isDirectory()) {
        actionExecutionContext
            .getEventHandler()
            .handle(
                Event.warn(
                        getOwner().getLocation(),
                        "output '"
                            + output.prettyPrint()
                            + "' of "
                            + ownerString
                            + " is a directory; dependency checking of directories is unsound")
                    .withTag(ownerString));
      }
    }
  }

  @Override
  public void prepare(Path execRoot) throws IOException {
    deleteOutputs(execRoot);
  }

  @Override
  public String describe() {
    String progressMessage = getProgressMessage();
    return progressMessage != null ? progressMessage : defaultProgressMessage();
  }

  @Override
  public boolean shouldReportPathPrefixConflict(ActionAnalysisMetadata action) {
    return this != action;
  }

  @Override
  public ExtraActionInfo.Builder getExtraActionInfo(ActionKeyContext actionKeyContext)
      throws CommandLineExpansionException {
    ActionOwner owner = getOwner();
    ExtraActionInfo.Builder result =
        ExtraActionInfo.newBuilder()
            .setOwner(owner.getLabel().toString())
            .setId(getKey(actionKeyContext))
            .setMnemonic(getMnemonic());
    Iterable<AspectDescriptor> aspectDescriptors = owner.getAspectDescriptors();
    AspectDescriptor lastAspect = null;

    for (AspectDescriptor aspectDescriptor : aspectDescriptors) {
      ExtraActionInfo.AspectDescriptor.Builder builder =
          ExtraActionInfo.AspectDescriptor.newBuilder()
            .setAspectName(aspectDescriptor.getAspectClass().getName());
      for (Map.Entry<String, Collection<String>> entry :
          aspectDescriptor.getParameters().getAttributes().asMap().entrySet()) {
          builder.putAspectParameters(
            entry.getKey(),
            ExtraActionInfo.AspectDescriptor.StringList.newBuilder()
                .addAllValue(entry.getValue())
                .build()
          );
      }
      lastAspect = aspectDescriptor;
    }
    if (lastAspect != null) {
      result.setAspectName(lastAspect.getAspectClass().getName());

      for (Map.Entry<String, Collection<String>> entry :
          lastAspect.getParameters().getAttributes().asMap().entrySet()) {
        result.putAspectParameters(
            entry.getKey(),
            ExtraActionInfo.StringList.newBuilder().addAllValue(entry.getValue()).build());
      }
    }
    return result;
  }

  @Override
  public ImmutableSet<Artifact> getMandatoryOutputs() {
    return ImmutableSet.of();
  }

  /**
   * Returns input files that need to be present to allow extra_action rules to shadow this action
   * correctly when run remotely. This is at least the normal inputs of the action, but may include
   * other files as well. For example C(++) compilation may perform include file header scanning.
   * This needs to be mirrored by the extra_action rule. Called by
   * {@link com.google.devtools.build.lib.analysis.extra.ExtraAction} at execution time for actions
   * that return true for {link #discoversInputs()}.
   *
   * @param actionExecutionContext Services in the scope of the action, like the Out/Err streams.
   * @throws ActionExecutionException only when code called from this method
   *     throws that exception.
   * @throws InterruptedException if interrupted
   */
  @Override
  public Iterable<Artifact> getInputFilesForExtraAction(
      ActionExecutionContext actionExecutionContext)
      throws ActionExecutionException, InterruptedException {
    return ImmutableList.of();
  }

  @Override
  public Depset getSkylarkInputs() {
    return Depset.of(Artifact.TYPE, NestedSetBuilder.wrap(Order.STABLE_ORDER, getInputs()));
  }

  @Override
  public Depset getSkylarkOutputs() {
    return Depset.of(Artifact.TYPE, NestedSetBuilder.wrap(Order.STABLE_ORDER, getOutputs()));
  }

  @Override
  public Sequence<String> getSkylarkArgv() throws EvalException {
    return null;
  }

  @Override
  public Sequence<CommandLineArgsApi> getStarlarkArgs() throws EvalException {
    // Not all action types support returning Args.
    return null;
  }

  @Override
  public String getSkylarkContent() throws IOException {
    return null;
  }

  @Override
  public Dict<String, String> getSkylarkSubstitutions() {
    return null;
  }

  @Override
  public Dict<String, String> getExecutionInfoDict() {
    Map<String, String> executionInfo = getExecutionInfo();
    if (executionInfo == null) {
      return null;
    }
    return Dict.copyOf(null, executionInfo);
  }

  @Override
  public Dict<String, String> getEnv() {
    return Dict.copyOf(null, env.getFixedEnv().toMap());
  }

  @Override
  public ImmutableMap<String, String> getExecProperties() {
    return getOwner().getExecProperties();
  }

  @Override
  @Nullable
  public PlatformInfo getExecutionPlatform() {
    return getOwner().getExecutionPlatform();
  }
}
