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

package com.google.devtools.build.lib.analysis;

import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.AbstractAction;
import com.google.devtools.build.lib.actions.ActionContext;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.util.OptionsUtils;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsProvider;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * An action writing the workspace status files.
 *
 * <p>These files represent information about the environment the build was run in. They are used by
 * language-specific build info factories to make the data in them available for individual
 * languages (e.g. by turning them into .h files for C++)
 *
 * <p>The format of these files a list of key-value pairs, one for each line. The key and the value
 * are separated by a space.
 *
 * <p>There are two of these files: volatile and stable. Changes in the volatile file do not cause
 * rebuilds if no other file is changed. This is useful for frequently-changing information that
 * does not significantly affect the build, e.g. the current time.
 */
public abstract class WorkspaceStatusAction extends AbstractAction {

  /** Options controlling the workspace status command. */
  public static class Options extends OptionsBase {
    @Option(
        name = "embed_label",
        defaultValue = "",
        valueHelp = "<string>",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        help = "Embed source control revision or release label in binary")
    public String embedLabel;

    @Option(
        name = "workspace_status_command",
        defaultValue = "",
        converter = OptionsUtils.PathFragmentConverter.class,
        valueHelp = "<path>",
        documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
        effectTags = {OptionEffectTag.UNKNOWN},
        help =
            "A command invoked at the beginning of the build to provide status "
                + "information about the workspace in the form of key/value pairs.  "
                + "See the User's Manual for the full specification. Also see"
                + "tools/buildstamp/get_workspace_status for an example.")
    public PathFragment workspaceStatusCommand;
  }

  /** The type of a workspace status action key. */
  public enum KeyType {
    INTEGER,
    STRING,
  }

  /**
   * Action context required by the workspace status action as well as language-specific actions
   * that write workspace status artifacts.
   */
  public interface Context extends ActionContext {
    ImmutableMap<String, Key> getStableKeys();

    ImmutableMap<String, Key> getVolatileKeys();

    // TODO(ulfjack): Maybe move these to a separate ActionContext interface?
    WorkspaceStatusAction.Options getOptions();

    ImmutableMap<String, String> getClientEnv();

    com.google.devtools.build.lib.shell.Command getCommand();
  }

  /** A key in the workspace status info file. */
  public static class Key {
    private final KeyType type;

    private final String defaultValue;
    private final String redactedValue;

    private Key(KeyType type, String defaultValue, String redactedValue) {
      this.type = type;
      this.defaultValue = defaultValue;
      this.redactedValue = redactedValue;
    }

    public KeyType getType() {
      return type;
    }

    public String getDefaultValue() {
      return defaultValue;
    }

    public String getRedactedValue() {
      return redactedValue;
    }

    public static Key of(KeyType type, String defaultValue, String redactedValue) {
      return new Key(type, defaultValue, redactedValue);
    }
  }

  /**
   * Parses the output of the workspace status action.
   *
   * <p>The output is a text file with each line representing a workspace status info key. The key
   * is the part of the line before the first space and should consist of the characters [A-Z_]
   * (although this is not checked). Everything after the first space is the value.
   */
  public static Map<String, String> parseValues(Path file) throws IOException {
    HashMap<String, String> result = new HashMap<>();
    Splitter lineSplitter = Splitter.on(' ').limit(2);
    for (String line :
        Splitter.on('\n').split(new String(FileSystemUtils.readContentAsLatin1(file)))) {
      List<String> items = lineSplitter.splitToList(line);
      if (items.size() != 2) {
        continue;
      }

      result.put(items.get(0), items.get(1));
    }

    return ImmutableMap.copyOf(result);
  }

  /** Environment for the {@link Factory} to create the workspace status action. */
  public interface Environment {
    Artifact createStableArtifact(String name);

    Artifact createVolatileArtifact(String name);
  }

  /**
   * Environment for the {@link Factory} to create the dummy workspace status information. This is a
   * subset of the information provided by CommandEnvironment. However, we cannot reference the
   * CommandEnvironment from here due to layering.
   */
  public interface DummyEnvironment {
    Path getWorkspace();

    String getBuildRequestId();

    OptionsProvider getOptions();
  }

  /** Factory for {@link WorkspaceStatusAction}. */
  public interface Factory {
    /**
     * Creates the workspace status action.
     *
     * <p>The action is never re-created, but the same action object is executed on every build. Use
     * {@link Context} to access any non-hermetic data.
     */
    WorkspaceStatusAction createWorkspaceStatusAction(Environment env);

    /**
     * Creates a dummy workspace status map. Used in cases where the build failed, so that part of
     * the workspace status is nevertheless available.
     */
    Map<String, String> createDummyWorkspaceStatus(DummyEnvironment env);
  }

  protected WorkspaceStatusAction(
      ActionOwner owner, NestedSet<Artifact> inputs, ImmutableSet<Artifact> outputs) {
    super(owner, inputs, outputs);
  }

  /**
   * The volatile status artifact containing items that may change even if nothing changed between
   * the two builds, e.g. current time.
   */
  public abstract Artifact getVolatileStatus();

  /**
   * The stable status artifact containing items that change only if information relevant to the
   * build changes, e.g. the name of the user running the build or the hostname.
   */
  public abstract Artifact getStableStatus();
}
