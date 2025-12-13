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

import com.google.devtools.build.lib.bazel.bzlmod.BzlmodRepoRuleValue;
import com.google.devtools.build.lib.bazel.bzlmod.modcommand.ModOptions.OutputFormat;
import com.google.devtools.build.lib.packages.LabelPrinter;
import com.google.devtools.build.lib.packages.RawAttributeMapper;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.query2.proto.proto2api.Build;
import com.google.devtools.build.lib.query2.query.output.BuildOutputFormatter.AttributeReader;
import com.google.devtools.build.lib.query2.query.output.BuildOutputFormatter.TargetOutputter;
import com.google.devtools.build.lib.query2.query.output.PossibleAttributeValues;
import com.google.devtools.build.lib.query2.query.output.ProtoOutputFormatter;
import com.google.protobuf.InvalidProtocolBufferException;
import com.google.protobuf.util.JsonFormat;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintWriter;

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

  public void print(String key, BzlmodRepoRuleValue repoRuleValue) {
    switch (outputFormat) {
      case TEXT -> printStarlark(key, repoRuleValue.getRule());
      case STREAMED_JSONPROTO, STREAMED_PROTO -> {
        if (outputFormat == OutputFormat.STREAMED_JSONPROTO) {
          printProtoJson(key, repoRuleValue.getRule());
        } else {
          printStreamedProto(key, repoRuleValue.getRule());
        }
      }
      default -> throw new IllegalArgumentException("Unknown output format: " + outputFormat);
    }
  }

  private void printStarlark(String key, Rule repoRule) {
    RuleDisplayOutputter outputter = new RuleDisplayOutputter(printer);
    printer.printf("## %s:\n", key);
    outputter.outputRule(repoRule);
  }

  private void printStreamedProto(String key, Rule repoRule) {
    Build.Repository serialized = serializeRepoDefinitionAsProto(key, repoRule);
    try {
      serialized.writeDelimitedTo(outputStream);
    } catch (IOException e) {
      // Ignore IOException like PrintWriter.
    }
  }

  private void printProtoJson(String key, Rule repoRule) {
    Build.Repository serialized = serializeRepoDefinitionAsProto(key, repoRule);
    try {
      printer.println(jsonPrinter.print(serialized));
    } catch (InvalidProtocolBufferException e) {
      throw new IllegalArgumentException(e);
    }
  }

  private Build.Repository serializeRepoDefinitionAsProto(String key, Rule repoRule) {
    // First use ProtoOutputFormatter to convert the Rule into a Target proto.
    // Then convert the Target proto to a Repository proto.

    var formatter = new ProtoOutputFormatter();
    try {
      Build.Target targetPb = formatter.toTargetProtoBuffer(repoRule, LabelPrinter.legacy());
      Build.Rule rulePb = targetPb.getRule();

      Build.Repository.Builder pbBuilder = Build.Repository.newBuilder();
      pbBuilder.setCanonicalName(rulePb.getName());
      pbBuilder.setRepoRuleName(rulePb.getRuleClass());
      pbBuilder.setRepoRuleBzlLabel(internalToUnicode(repoRule.getRuleClassObject().getKey()));

      // TODO: record and print the call stack for the repo definition itself?

      if (key.startsWith("@")) {
        if (!key.startsWith("@@")) {
          pbBuilder.setApparentName(internalToUnicode(key));
        }
      } else {
        pbBuilder.setModuleKey(internalToUnicode(key));
      }

      for (Build.Attribute attr : rulePb.getAttributeList()) {
        if (attr.getName().equals("_original_name")
            && attr.getType() == Build.Attribute.Discriminator.STRING) {
          pbBuilder.setOriginalName(attr.getStringValue());
          continue;
        }
        pbBuilder.addAttribute(attr);
      }

      return pbBuilder.build();
    } catch (InterruptedException ex) {
      // should never happen
      throw new RuntimeException(ex);
    }
  }

  /**
   * Uses Query's {@link TargetOutputter} to display the generating repo rule and other information.
   */
  static class RuleDisplayOutputter {
    private static final AttributeReader attrReader =
        (rule, attr) ->
            // Query's implementation copied
            PossibleAttributeValues.forRuleAndAttribute(
                rule, attr, /* mayTreatMultipleAsNone= */ true);
    private final TargetOutputter targetOutputter;
    private final PrintWriter printer;

    RuleDisplayOutputter(PrintWriter printer) {
      this.printer = printer;
      this.targetOutputter =
          new TargetOutputter(
              this.printer,
              (rule, attr) -> RawAttributeMapper.of(rule).isConfigurable(attr.getName()),
              "\n",
              LabelPrinter.legacy());
    }

    private void outputRule(Rule rule) {
      try {
        targetOutputter.outputRule(rule, attrReader, this.printer);
      } catch (IOException e) {
        throw new IllegalStateException(e);
      }
    }
  }
}
