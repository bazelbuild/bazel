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

package com.google.devtools.build.lib.rules.cpp;

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.BuildInfoHelper;
import com.google.devtools.build.lib.analysis.WorkspaceStatusAction;
import com.google.devtools.build.lib.analysis.actions.AbstractFileWriteAction;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.util.Preconditions;

import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * An action that creates a C++ header containing the build information in the
 * form of #define directives.
 */
public final class WriteBuildInfoHeaderAction extends AbstractFileWriteAction {
  private static final String GUID = "b0798174-1352-4a54-854a-9785aaea491b";

  private final ImmutableList<Artifact> valueArtifacts;

  private final boolean writeVolatileInfo;
  private final boolean writeStableInfo;

  /**
   * Creates an action that writes a C++ header with the build information.
   *
   * <p>It reads the set of build info keys from an action context that is usually contributed
   * to Bazel by the workspace status module, and the value associated with said keys from the
   * workspace status files (stable and volatile) written by the workspace status action.
   *
   * <p>Without input artifacts this action uses redacted build information.
   * @param inputs Artifacts that contain build information, or an empty
   *        collection to use redacted build information
   * @param output the C++ header Artifact created by this action
   * @param writeVolatileInfo whether to write the volatile part of the build
   *        information to the generated header
   * @param writeStableInfo whether to write the non-volatile part of the
   *        build information to the generated header
   */
  public WriteBuildInfoHeaderAction(Collection<Artifact> inputs,
      Artifact output, boolean writeVolatileInfo, boolean writeStableInfo) {
    super(BuildInfoHelper.BUILD_INFO_ACTION_OWNER,
        inputs, output, /*makeExecutable=*/false);
    valueArtifacts = ImmutableList.copyOf(inputs);
    if (!inputs.isEmpty()) {
      // With non-empty inputs we should not generate both volatile and non-volatile data
      // in the same header file.
      Preconditions.checkState(writeVolatileInfo ^ writeStableInfo);
    }
    Preconditions.checkState(
        output.isConstantMetadata() == (writeVolatileInfo && !inputs.isEmpty()));

    this.writeVolatileInfo = writeVolatileInfo;
    this.writeStableInfo = writeStableInfo;
  }

  @Override
  public DeterministicWriter newDeterministicWriter(ActionExecutionContext ctx)
      throws IOException {
    WorkspaceStatusAction.Context context =
        ctx.getExecutor().getContext(WorkspaceStatusAction.Context.class);

    final Map<String, WorkspaceStatusAction.Key> keys = new LinkedHashMap<>();
    if (writeVolatileInfo) {
      keys.putAll(context.getVolatileKeys());
    }

    if (writeStableInfo) {
      keys.putAll(context.getStableKeys());
    }

    final Map<String, String> values = new LinkedHashMap<>();
    for (Artifact valueFile : valueArtifacts) {
      values.putAll(WorkspaceStatusAction.parseValues(valueFile.getPath()));
    }

    final boolean redacted = valueArtifacts.isEmpty();

    return new DeterministicWriter() {
      @Override
      public void writeOutputFile(OutputStream out) throws IOException {
        Writer writer = new OutputStreamWriter(out, UTF_8);

       for (Map.Entry<String, WorkspaceStatusAction.Key> key : keys.entrySet()) {
          if (!key.getValue().isInLanguage("C++")) {
            continue;
          }

          String value = redacted ? key.getValue().getRedactedValue()
              : values.containsKey(key.getKey()) ? values.get(key.getKey())
              : key.getValue().getDefaultValue();

          switch (key.getValue().getType()) {
            case VERBATIM:
            case INTEGER:
              break;

            case STRING:
              value = quote(value);
              break;

            default:
              throw new IllegalStateException();
          }
          define(writer, key.getKey(), value);

        }
        writer.flush();
      }
    };
  }

  @Override
  protected String computeKey() {
    Fingerprint f = new Fingerprint();
    f.addString(GUID);
    f.addBoolean(writeStableInfo);
    f.addBoolean(writeVolatileInfo);
    return f.hexDigestAndReset();
  }

  @Override
  public boolean executeUnconditionally() {
    // Note: isVolatile must return true if executeUnconditionally can ever return true
    // for this instance.
    return isUnconditional();
  }

  @Override
  public boolean isVolatile() {
    return isUnconditional();
  }

  private boolean isUnconditional() {
    // Because of special handling in the MetadataHandler, changed volatile build
    // information does not trigger relinking of all libraries that have
    // linkstamps. But we do want to regenerate the header in case libraries are
    // relinked because of other reasons.
    // Without inputs the contents of the header do not change, so there is no
    // point in executing the action again in that case.
    return writeVolatileInfo && !Iterables.isEmpty(getInputs());
  }

  /**
   * Quote a string with double quotes.
   */
  private String quote(String string) {
    // TODO(bazel-team): This is doesn't really work if the string contains quotes. Or a newline.
    // Or a backslash. Or anything unusual, really.
    return "\"" + string + "\"";
  }

  /**
   * Write a preprocessor define directive to a Writer.
   */
  private void define(Writer writer, String name, String value) throws IOException {
    writer.write("#define ");
    writer.write(name);
    writer.write(' ');
    writer.write(value);
    writer.write('\n');
  }

  @Override
  protected String getRawProgressMessage() {
    return null;
  }
}
