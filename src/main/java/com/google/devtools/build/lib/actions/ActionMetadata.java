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

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;

import javax.annotation.Nullable;

/**
 * Side-effect free query methods for information about an {@link Action}.
 *
 * <p>This method is intended for use in situations when the intention is to pass around information
 * about an action without allowing actual execution of the action.
 *
 * <p>The split between {@link Action} and {@link ActionMetadata} is somewhat arbitrary, other than
 * that all methods with side effects must belong to the former.
 */
public interface ActionMetadata {
  /**
   * If this executable can supply verbose information, returns a string that can be used as a
   * progress message while this executable is running. A return value of {@code null} indicates no
   * message should be reported.
   */
  @Nullable
  String getProgressMessage();

  /**
   * Returns the owner of this executable if this executable can supply verbose information. This is
   * typically the rule that constructed it; see ActionOwner class comment for details. Returns
   * {@code null} if no owner can be determined.
   *
   * <p>If this executable does not supply verbose information, this function may throw an
   * IllegalStateException.
   */
  ActionOwner getOwner();

  /**
   * Returns a mnemonic (string constant) for this kind of action; written into
   * the master log so that the appropriate parser can be invoked for the output
   * of the action. Effectively a public method as the value is used by the
   * extra_action feature to match actions.
   */
  String getMnemonic();

  /**
   * Returns a pretty string representation of this action, suitable for use in
   * progress messages or error messages.
   */
  String prettyPrint();

  /**
   * Returns true iff the getInputs set is known to be complete.
   *
   * <p>For most Actions, this always returns true, but in some cases (e.g. C++ compilation), inputs
   * are dynamically discovered from the previous execution of the Action, and so before the initial
   * execution, this method will return false in those cases.
   *
   * <p>Any builder <em>must</em> unconditionally execute an Action for which inputsKnown() returns
   * false, regardless of all other inferences made by its dependency analysis. In addition, all
   * prerequisites mentioned in the (possibly incomplete) value returned by getInputs must also be
   * built first, as usual.
   */
  @ThreadSafe
  boolean inputsKnown();

  /**
   * Returns true iff inputsKnown() may ever return false.
   */
  @ThreadSafe
  boolean discoversInputs();

  /**
   * Returns the tool Artifacts that this Action depends upon. May be empty. This is a subset of
   * getInputs().
   *
   * <p>This may be used by spawn strategies to determine whether an external tool has not changed
   * since the last time it was used and could thus be reused, or whether it has to be restarted.
   *
   * <p>See {@link AbstractAction#getTools()} for an explanation of why it's important that this
   * set contains exactly the right set of artifacts in order for the build to stay correct and the
   * worker strategy to work.
   */
  Iterable<Artifact> getTools();

  /**
   * Returns the input Artifacts that this Action depends upon. May be empty.
   *
   * <p>During execution, the {@link Iterable} returned by {@code getInputs} <em>must not</em> be
   * concurrently modified before the value is fully read in {@code JavaDistributorDriver#exec} (via
   * the {@code Iterable<ActionInput>} argument there). Violating this would require somewhat
   * pathological behavior by the {@link Action}, since it would have to modify its inputs, as a
   * list, say, without reassigning them. This should never happen with any Action subclassing
   * AbstractAction, since AbstractAction's implementation of getInputs() returns an immutable
   * iterable.
   */
  Iterable<Artifact> getInputs();

  /**
   * Get the {@link RunfilesSupplier} providing runfiles needed by this action.
   */
  RunfilesSupplier getRunfilesSupplier();

  /**
   * Returns the (unordered, immutable) set of output Artifacts that
   * this action generates.  (It would not make sense for this to be empty.)
   */
  ImmutableSet<Artifact> getOutputs();

  /**
   * Returns the set of output Artifacts that are required to be saved. This is
   * used to identify items that would otherwise be potentially identified as
   * orphaned (not consumed by any downstream {@link Action}s and potentially
   * discarded during the build process.
   */
  ImmutableSet<Artifact> getMandatoryOutputs();

  /**
   * Returns the "primary" input of this action, if applicable.
   *
   * <p>For example, a C++ compile action would return the .cc file which is being compiled,
   * irrespective of the other inputs.
   *
   * <p>May return null.
   */
  Artifact getPrimaryInput();

  /**
   * Returns the "primary" output of this action.
   *
   * <p>For example, the linked library would be the primary output of a LinkAction.
   *
   * <p>Never returns null.
   */
  Artifact getPrimaryOutput();

  /**
   * Returns an iterable of input Artifacts that MUST exist prior to executing an action. In other
   * words, in case when action is scheduled for execution, builder will ensure that all artifacts
   * returned by this method are present in the filesystem (artifact.getPath().exists() is true) or
   * action execution will be aborted with an error that input file does not exist. While in
   * majority of cases this method will return all action inputs, for some actions (e.g.
   * CppCompileAction) it can return a subset of inputs because that not all action inputs might be
   * mandatory for action execution to succeed (e.g. header files retrieved from *.d file from the
   * previous build).
   */
  Iterable<Artifact> getMandatoryInputs();

  /**
   * <p>Returns a string encoding all of the significant behaviour of this
   * Action that might affect the output.  The general contract of
   * <code>getKey</code> is this: if the work to be performed by the
   * execution of this action changes, the key must change. </p>
   *
   * <p>As a corollary, the build system is free to omit the execution of an
   * Action <code>a1</code> if (a) at some time in the past, it has already
   * executed an Action <code>a0</code> with the same key as
   * <code>a1</code>, and (b) the names and contents of the input files listed
   * by <code>a1.getInputs()</code> are identical to the names and contents of
   * the files listed by <code>a0.getInputs()</code>. </p>
   *
   * <p>Examples of changes that should affect the key are:
   * <ul>
   *  <li>Changes to the BUILD file that materially affect the rule which gave
   *  rise to this Action.</li>
   *
   *  <li>Changes to the command-line options, environment, or other global
   *  configuration resources which affect the behaviour of this kind of Action
   *  (other than changes to the names of the input/output files, which are
   *  handled externally).</li>
   *
   *  <li>An upgrade to the build tools which changes the program logic of this
   *  kind of Action (typically this is achieved by incorporating a UUID into
   *  the key, which is changed each time the program logic of this action
   *  changes).</li>
   *
   * </ul></p>
   */
  String getKey();

  /**
   * Returns a human-readable description of the inputs to {@link #getKey()}.
   * Used in the output from '--explain', and in error messages for
   * '--check_up_to_date' and '--check_tests_up_to_date'.
   * May return null, meaning no extra information is available.
   *
   * <p>If the return value is non-null, for consistency it should be a multiline message of the
   * form:
   * <pre>
   *   <var>Summary</var>
   *     <var>Fieldname</var>: <var>value</var>
   *     <var>Fieldname</var>: <var>value</var>
   *     ...
   * </pre>
   * where each line after the first one is intended two spaces, and where any fields that might
   * contain newlines or other funny characters are escaped using {@link
   * com.google.devtools.build.lib.shell.ShellUtils#shellEscape}.
   * For example:
   * <pre>
   *   Compiling foo.cc
   *     Command: /usr/bin/gcc
   *     Argument: '-c'
   *     Argument: foo.cc
   *     Argument: '-o'
   *     Argument: foo.o
   * </pre>
   */
  @Nullable String describeKey();
}
