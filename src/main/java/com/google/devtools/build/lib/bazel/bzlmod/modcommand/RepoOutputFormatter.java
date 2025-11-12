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
          .append(Starlark.repr(attr.getValue()))
          .append(",\n");
    }
    printer.append(")\n");
    // TODO: record and print the call stack for the repo definition itself?
    printer.append("\n");
  }

  private void printStreamedProto(String key, RepoDefinition repoDefinition) {
    Build.Target serialized = serializeRepoDefinitionAsProto(key, repoDefinition);
    try {
      serialized.writeDelimitedTo(outputStream);
    } catch (InvalidProtocolBufferException e) {
      throw new IllegalArgumentException(e);
    } catch (IOException e) {
      // Ignore IOException like PrintWriter.
    }
  }

  private void printProtoJson(String key, RepoDefinition repoDefinition) {
    Build.Target serialized = serializeRepoDefinitionAsProto(key, repoDefinition);
    try {
      printer.println(jsonPrinter.print(serialized));
    } catch (InvalidProtocolBufferException e) {
      throw new IllegalArgumentException(e);
    }
  }

  private Build.Target serializeRepoDefinitionAsProto(String key, RepoDefinition repoDefinition) {
    RepoRule repoRule = repoDefinition.repoRule();

    Build.Target.Builder pbBuilder = Build.Target.newBuilder();
    pbBuilder.setType(Build.Target.Discriminator.RULE);

    Build.Rule.Builder ruleBuilder = pbBuilder.getRuleBuilder();
    ruleBuilder.setName(internalToUnicode(repoDefinition.name()));
    ruleBuilder.setRuleClass(internalToUnicode(repoRule.id().ruleName()));
    ruleBuilder.setRuleClassKey(internalToUnicode(repoRule.id().toString()));

    // TODO: record and print the call stack for the repo definition itself?
    // ruleBuilder.setLocation(repoDefinition.location());
    // ruleBuilder.addAllInstantiationStack();

    if (key.startsWith("@")) {
      if (!key.startsWith("@@")) {
        ruleBuilder
            .addAttributeBuilder()
            .setName("$apparent_repo_name")
            .setType(Build.Attribute.Discriminator.STRING)
            .setStringValue(internalToUnicode(key))
            .setExplicitlySpecified(true);
      }
    } else {
      ruleBuilder
          .addAttributeBuilder()
          .setName("$module_key")
          .setType(Build.Attribute.Discriminator.STRING)
          .setStringValue(internalToUnicode(key))
          .setExplicitlySpecified(true);
    }

    if (repoDefinition.originalName() != null) {
      ruleBuilder
          .addAttributeBuilder()
          .setName("$original_name")
          .setType(Build.Attribute.Discriminator.STRING)
          .setStringValue(internalToUnicode(repoDefinition.originalName()))
          .setExplicitlySpecified(true);
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
      ruleBuilder.addAttribute(serializedAttribute);
    }

    return pbBuilder.build();
  }
}
