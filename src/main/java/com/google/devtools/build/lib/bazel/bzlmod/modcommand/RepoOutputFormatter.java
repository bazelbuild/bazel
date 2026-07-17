// Copyright 2025 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.bzlmod.modcommand;

import static com.google.devtools.build.lib.util.StringEncoding.internalToUnicode;

import com.google.devtools.build.lib.bazel.bzlmod.modcommand.ModOptions.OutputFormat;
import com.google.devtools.build.lib.bazel.repository.RepoDefinition;
import com.google.devtools.build.lib.bazel.repository.RepoDefinitionValue;
import com.google.devtools.build.lib.bazel.repository.RepoRule;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.AttributeFormatter;
import com.google.devtools.build.lib.packages.LabelPrinter;
import com.google.devtools.build.lib.query2.proto.proto2api.Build;
import com.google.protobuf.InvalidProtocolBufferException;
import com.google.protobuf.util.JsonFormat;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.util.Map;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkSemantics;

/** Outputs repository definitions for {@code mod show_repo}. */
public class RepoOutputFormatter {
  private static final JsonFormat.Printer jsonPrinter =
      JsonFormat.printer().omittingInsignificantWhitespace();

  private final PrintWriter printer;
  private final OutputStream outputStream;
  private final OutputFormat outputFormat;

  public RepoOutputFormatter(
      PrintWriter printer, OutputStream outputStream, OutputFormat outputFormat) {
    this.printer = printer;
    this.outputStream = outputStream;
    this.outputFormat = outputFormat;
  }

  public void print(String key, RepoDefinitionValue repoDefinition) {
    switch (outputFormat) {
      case TEXT -> printStarlark(key, repoDefinition);
      case STREAMED_JSONPROTO, STREAMED_PROTO -> {
        // In proto output formats, we only print repo definitions, not overrides.
        if (repoDefinition instanceof RepoDefinitionValue.Found repoDefValue) {
          if (outputFormat == OutputFormat.STREAMED_JSONPROTO) {
            printProtoJson(key, repoDefValue.repoDefinition());
          } else {
            printStreamedProto(key, repoDefValue.repoDefinition());
          }
        }
      }
      default -> throw new IllegalArgumentException("Unknown output format: " + outputFormat);
    }
  }

  private void printStarlark(String key, RepoDefinitionValue repoDefinition) {
    if (repoDefinition instanceof RepoDefinitionValue.Found repoDefValue) {
      printer.printf("## %s:\n", key);
      printStarlark(repoDefValue.repoDefinition());
    }
    if (repoDefinition instanceof RepoDefinitionValue.RepoOverride repoOverrideValue) {
      printer.printf(
          "## %s:\nBuiltin or overridden repo located at: %s\n\n",
          key, repoOverrideValue.repoPath());
    }
  }

  private void printStarlark(RepoDefinition repoDefinition) {
    RepoRule repoRule = repoDefinition.repoRule();
    printer
        .append("load(\"")
        .append(repoRule.id().bzlFileLabel().getUnambiguousCanonicalForm())
        .append("\", \"")
        .append(repoRule.id().ruleName())
        .append("\")\n");
    printer.append(repoRule.id().ruleName()).append("(\n");
    printer.append("  name = \"").append(repoDefinition.name()).append("\",\n");
    if (repoDefinition.originalName() != null) {
      printer.append("  _original_name = \"").append(repoDefinition.originalName()).append("\",\n");
    }
    for (Map.Entry<String, Object> attr : repoDefinition.attrValues().attributes().entrySet()) {
      printer
          .append("  ")
          .append(attr.getKey())
          .append(" = ")
          .append(Starlark.repr(attr.getValue(), StarlarkSemantics.DEFAULT))
          .append(",\n");
    }
    printer.append(")\n");
    // TODO: record and print the call stack for the repo definition itself?
    printer.append("\n");
  }

  private void printStreamedProto(String key, RepoDefinition repoDefinition) {
    Build.Repository serialized = serializeRepoDefinitionAsProto(key, repoDefinition);
    try {
      serialized.writeDelimitedTo(outputStream);
    } catch (IOException e) {
      // Ignore IOException like PrintWriter.
    }
  }

  private void printProtoJson(String key, RepoDefinition repoDefinition) {
    Build.Repository serialized = serializeRepoDefinitionAsProto(key, repoDefinition);
    try {
      printer.println(jsonPrinter.print(serialized));
    } catch (InvalidProtocolBufferException e) {
      throw new IllegalArgumentException(e);
    }
  }

  private Build.Repository serializeRepoDefinitionAsProto(
      String key, RepoDefinition repoDefinition) {
    RepoRule repoRule = repoDefinition.repoRule();

    Build.Repository.Builder pbBuilder = Build.Repository.newBuilder();
    pbBuilder.setCanonicalName(internalToUnicode(repoDefinition.name()));
    pbBuilder.setRepoRuleName(internalToUnicode(repoRule.id().ruleName()));
    pbBuilder.setRepoRuleBzlLabel(
        internalToUnicode(repoRule.id().bzlFileLabel().getUnambiguousCanonicalForm()));

    // TODO: record and print the call stack for the repo definition itself?

    if (key.startsWith("@")) {
      if (!key.startsWith("@@")) {
        pbBuilder.setApparentName(internalToUnicode(key));
      }
    } else {
      pbBuilder.setModuleKey(internalToUnicode(key));
    }
    if (repoDefinition.originalName() != null) {
      pbBuilder.setOriginalName(internalToUnicode(repoDefinition.originalName()));
    }

    for (Map.Entry<String, Integer> attr : repoRule.attributeIndices().entrySet()) {
      String attrName = attr.getKey();
      Attribute attrDefinition = repoRule.attributes().get(attr.getValue());

      boolean explicitlySpecified = repoDefinition.attrValues().attributes().containsKey(attrName);
      Object attrValue = repoDefinition.attrValues().attributes().get(attrName);
      if (attrValue == null) {
        attrValue = attrDefinition.getDefaultValueUnchecked();
      }
      Build.Attribute serializedAttribute =
          AttributeFormatter.getAttributeProto(
              attrDefinition,
              attrValue,
              explicitlySpecified,
              /* encodeBooleanAndTriStateAsIntegerAndString= */ true,
              /* sourceAspect= */ null,
              /* includeAttributeSourceAspects= */ false,
              LabelPrinter.legacy());
      pbBuilder.addAttribute(serializedAttribute);
    }

    return pbBuilder.build();
  }
}
