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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.cache.MetadataHandler;
import com.google.devtools.build.lib.actions.extra.ExtraActionInfo;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.CollectionUtils;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.skylarkinterface.SkylarkValue;
import com.google.devtools.build.lib.syntax.Printer;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Symlinks;

import java.io.IOException;
import java.util.Collection;

import javax.annotation.Nullable;

/**
 * Abstract implementation of Action which implements basic functionality: the
 * inputs, outputs, and toString method.  Both input and output sets are
 * immutable.
 */
@Immutable @ThreadSafe
public abstract class AbstractAction implements Action, SkylarkValue {

  /**
   * An arbitrary default resource set. Currently 250MB of memory, 50% CPU and 0% of total I/O.
   */
  public static final ResourceSet DEFAULT_RESOURCE_SET =
      ResourceSet.createWithRamCpuIo(250, 0.5, 0);

  /**
   * The owner/inputs/outputs attributes below should never be directly accessed even within
   * AbstractAction itself. The appropriate getter methods should be used instead. This has to be
   * done due to the fact that the getter methods can be overridden in subclasses.
   */
  private final ActionOwner owner;

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

  // The variable inputs is non-final only so that actions that discover their inputs can modify it.
  private Iterable<Artifact> inputs;
  private final RunfilesSupplier runfilesSupplier;
  private final ImmutableSet<Artifact> outputs;

  private String cachedKey;

  /**
   * Construct an abstract action with the specified inputs and outputs;
   */
  protected AbstractAction(ActionOwner owner,
                           Iterable<Artifact> inputs,
                           Iterable<Artifact> outputs) {
    this(owner, ImmutableList.<Artifact>of(), inputs, EmptyRunfilesSupplier.INSTANCE, outputs);
  }

  /**
   * Construct an abstract action with the specified tools, inputs and outputs;
   */
  protected AbstractAction(
      ActionOwner owner,
      Iterable<Artifact> tools,
      Iterable<Artifact> inputs,
      Iterable<Artifact> outputs) {
    this(owner, tools, inputs, EmptyRunfilesSupplier.INSTANCE, outputs);
  }

  protected AbstractAction(
      ActionOwner owner,
      Iterable<Artifact> inputs,
      RunfilesSupplier runfilesSupplier,
      Iterable<Artifact> outputs) {
    this(owner, ImmutableList.<Artifact>of(), inputs, runfilesSupplier, outputs);
  }

  protected AbstractAction(
      ActionOwner owner,
      Iterable<Artifact> tools,
      Iterable<Artifact> inputs,
      RunfilesSupplier runfilesSupplier,
      Iterable<Artifact> outputs) {
    Preconditions.checkNotNull(owner);
    // TODO(bazel-team): Use RuleContext.actionOwner here instead
    this.owner = new ActionOwnerDescription(owner);
    this.tools = CollectionUtils.makeImmutable(tools);
    this.inputs = CollectionUtils.makeImmutable(inputs);
    this.outputs = ImmutableSet.copyOf(outputs);
    this.runfilesSupplier = Preconditions.checkNotNull(runfilesSupplier,
        "runfilesSupplier may not be null");
    Preconditions.checkArgument(!this.outputs.isEmpty(), "action outputs may not be empty");
  }

  @Override
  public final ActionOwner getOwner() {
    return owner;
  }

  @Override
  public boolean inputsKnown() {
    return true;
  }

  @Override
  public boolean discoversInputs() {
    return false;
  }

  @Override
  public Collection<Artifact> discoverInputs(ActionExecutionContext actionExecutionContext)
      throws ActionExecutionException, InterruptedException {
    throw new IllegalStateException("discoverInputs cannot be called for " + this.prettyPrint()
        + " since it does not discover inputs");
  }

  @Nullable
  @Override
  public Iterable<Artifact> resolveInputsFromCache(ArtifactResolver artifactResolver,
      PackageRootResolver resolver, Collection<PathFragment> inputPaths)
          throws PackageRootResolutionException {
    throw new IllegalStateException(
        "Method must be overridden for actions that may have unknown inputs.");
  }

  @Override
  public void updateInputs(Iterable<Artifact> inputs) {
    throw new IllegalStateException(
        "Method must be overridden for actions that may have unknown inputs.");
  }

  @Override
  public Iterable<Artifact> getTools() {
    return tools;
  }

  /**
   * Should only be overridden by actions that need to optionally insert inputs. Actions that
   * discover their inputs should use {@link #setInputs} to set the new iterable of inputs when they
   * know it.
   */
  @Override
  public Iterable<Artifact> getInputs() {
    return inputs;
  }

  @Override
  public RunfilesSupplier getRunfilesSupplier() {
    return runfilesSupplier;
  }

  /**
   * Set the inputs of the action. May only be used by an action that {@link #discoversInputs()}.
   * The iterable passed in is automatically made immutable.
   */
  protected void setInputs(Iterable<Artifact> inputs) {
    Preconditions.checkState(discoversInputs(), this);
    this.inputs = CollectionUtils.makeImmutable(inputs);
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
        + (inputsKnown() ? " -> " : ", unknown inputs -> ")
        + getOutputs() + "]" + ")";
  }

  @Override
  public abstract String getMnemonic();
  protected abstract String computeKey();

  @Override
  public final synchronized String getKey() {
    if (cachedKey == null) {
      cachedKey = computeKey();
    }
    return cachedKey;
  }

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
  public boolean isImmutable() {
    return false;
  }

  @Override
  public void write(Appendable buffer, char quotationMark) {
    Printer.append(buffer, prettyPrint()); // TODO(bazel-team): implement a readable representation
  }

  /**
   * Deletes all of the action's output files, if they exist. If any of the
   * Artifacts refers to a directory recursively removes the contents of the
   * directory.
   *
   * @param execRoot the exec root in which this action is executed
   */
  protected void deleteOutputs(Path execRoot) throws IOException {
    for (Artifact output : getOutputs()) {
      deleteOutput(output);
    }
  }

  /**
   * Helper method to remove an Artifact. If the Artifact refers to a directory
   * recursively removes the contents of the directory.
   */
  protected void deleteOutput(Artifact output) throws IOException {
    Path path = output.getPath();
    try {
      // Optimize for the common case: output artifacts are files.
      path.delete();
    } catch (IOException e) {
      // Only try to recursively delete a directory if the output root is known. This is just a
      // sanity check so that we do not start deleting random files on disk.
      // TODO(bazel-team): Strengthen this test by making sure that the output is part of the
      // output tree.
      if (path.isDirectory(Symlinks.NOFOLLOW) && output.getRoot() != null) {
        FileSystemUtils.deleteTree(path);
      } else {
        throw e;
      }
    }
  }

  /**
   * If the action might read directories as inputs in a way that is unsound wrt dependency
   * checking, this method must be called.
   */
  protected void checkInputsForDirectories(EventHandler eventHandler,
                                           MetadataHandler metadataHandler) {
    // Report "directory dependency checking" warning only for non-generated directories (generated
    // ones will be reported earlier).
    for (Artifact input : getMandatoryInputs()) {
      // Assume that if the file did not exist, we would not have gotten here.
      if (input.isSourceArtifact() && !metadataHandler.isRegularFile(input)) {
        eventHandler.handle(Event.warn(getOwner().getLocation(), "input '"
            + input.prettyPrint() + "' to " + getOwner().getLabel()
            + " is a directory; dependency checking of directories is unsound"));
      }
    }
  }

  @Override
  public MiddlemanType getActionType() {
    return MiddlemanType.NORMAL;
  }

  /**
   * If the action might create directories as outputs this method must be called.
   */
  protected void checkOutputsForDirectories(EventHandler eventHandler) {
    for (Artifact output : getOutputs()) {
      Path path = output.getPath();
      String ownerString = Label.print(getOwner().getLabel());
      if (path.isDirectory()) {
        eventHandler.handle(new Event(EventKind.WARNING, getOwner().getLocation(),
            "output '" + output.prettyPrint() + "' of " + ownerString
                  + " is a directory; dependency checking of directories is unsound",
                  ownerString));
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
  public boolean shouldReportPathPrefixConflict(Action action) {
    return this != action;
  }

  @Override
  public ExtraActionInfo.Builder getExtraActionInfo() {
    return ExtraActionInfo.newBuilder()
        .setOwner(getOwner().getLabel().toString())
        .setId(getKey())
        .setMnemonic(getMnemonic());
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
   * {@link com.google.devtools.build.lib.rules.extra.ExtraAction} at execution time.
   *
   * <p>As this method is called from the ExtraAction, make sure it is ok to call
   * this method from a different thread than the one this action is executed on.
   *
   * @param actionExecutionContext Services in the scope of the action, like the Out/Err streams.
   * @throws ActionExecutionException only when code called from this method
   *     throws that exception.
   * @throws InterruptedException if interrupted
   */
  public Iterable<Artifact> getInputFilesForExtraAction(
      ActionExecutionContext actionExecutionContext)
      throws ActionExecutionException, InterruptedException {
    return getInputs();
  }

  /**
   * A copying implementation of {@link ActionOwner}.
   *
   * <p>ConfiguredTargets implement ActionOwner themselves, but we do not want actions
   * to keep direct references to configured targets just for a label and a few strings.
   */
  @Immutable
  private static class ActionOwnerDescription implements ActionOwner {

    private final Location location;
    private final Label label;
    private final String configurationMnemonic;
    private final String configurationChecksum;
    private final String targetKind;
    private final String additionalProgressInfo;

    private ActionOwnerDescription(ActionOwner originalOwner) {
      this.location = originalOwner.getLocation();
      this.label = originalOwner.getLabel();
      this.configurationMnemonic = originalOwner.getConfigurationMnemonic();
      this.configurationChecksum = originalOwner.getConfigurationChecksum();
      this.targetKind = originalOwner.getTargetKind();
      this.additionalProgressInfo = originalOwner.getAdditionalProgressInfo();
    }

    @Override
    public Location getLocation() {
      return location;
    }

    @Override
    public Label getLabel() {
      return label;
    }

    @Override
    public String getConfigurationMnemonic() {
      return configurationMnemonic;
    }

    @Override
    public String getConfigurationChecksum() {
      return configurationChecksum;
    }

    @Override
    public String getTargetKind() {
      return targetKind;
    }

    @Override
    public String getAdditionalProgressInfo() {
      return additionalProgressInfo;
    }
  }
}
