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
package com.google.devtools.build.lib.query2.cquery;

import static com.google.common.collect.ImmutableList.toImmutableList;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Ordering;
import com.google.devtools.build.lib.analysis.AnalysisProtosV2;
import com.google.devtools.build.lib.analysis.AnalysisProtosV2.Configuration;
import com.google.devtools.build.lib.analysis.AnalysisProtosV2.CqueryResult;
import com.google.devtools.build.lib.analysis.AnalysisProtosV2.CqueryResultOrBuilder;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.ConfigMatchingProvider;
import com.google.devtools.build.lib.analysis.config.output.ConfigurationForOutput;
import com.google.devtools.build.lib.analysis.config.output.FragmentForOutput;
import com.google.devtools.build.lib.analysis.config.output.FragmentOptionsForOutput;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.AttributeFormatter;
import com.google.devtools.build.lib.packages.ConfiguredAttributeMapper;
import com.google.devtools.build.lib.packages.LabelPrinter;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.query2.common.CqueryNode;
import com.google.devtools.build.lib.query2.cquery.CqueryOptions.Transitions;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.TargetAccessor;
import com.google.devtools.build.lib.query2.proto.proto2api.Build;
import com.google.devtools.build.lib.query2.proto.proto2api.Build.ConfiguredRuleInput;
import com.google.devtools.build.lib.query2.proto.proto2api.Build.QueryResult;
import com.google.devtools.build.lib.query2.query.aspectresolvers.AspectResolver;
import com.google.devtools.build.lib.query2.query.output.ProtoOutputFormatter;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import com.google.devtools.build.lib.skyframe.config.BuildConfigurationKey;
import com.google.protobuf.Message;
import com.google.protobuf.TextFormat;
import com.google.protobuf.util.JsonFormat;
import java.io.IOException;
import java.io.OutputStream;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

/** Proto output formatter for cquery results. */
class ProtoOutputFormatterCallback extends CqueryThreadsafeCallback {

  /** Defines the types of proto output this class can handle. */
  public enum OutputType {
    BINARY("proto"),
    DELIMITED_BINARY("streamed_proto"),
    TEXT("textproto"),
    JSON("jsonproto");

    private final String formatName;

    OutputType(String formatName) {
      this.formatName = formatName;
    }

    public String formatName() {
      return formatName;
    }
  }

  private static class ConfigurationCache {
    private final Map<BuildConfigurationValue, Integer> cache = new LinkedHashMap<>();
    private final Function<BuildConfigurationKey, BuildConfigurationValue> configurationGetter;

    private ConfigurationCache(
        Function<BuildConfigurationKey, BuildConfigurationValue> configurationGetter) {
      this.configurationGetter = configurationGetter;
    }

    public int getId(BuildConfigurationValue buildConfigurationValue) {
      return cache.computeIfAbsent(buildConfigurationValue, event -> cache.size() + 1);
    }

    public int getId(BuildOptions options) {
      BuildConfigurationValue configurationValue =
          configurationGetter.apply(BuildConfigurationKey.create(options));
      return getId(configurationValue);
    }

    public ImmutableList<Configuration> getConfigurations() {
      return cache.entrySet().stream()
          .map(v -> createConfigurationProto(v.getKey(), v.getValue()))
          .collect(toImmutableList());
    }

    private Configuration createConfigurationProto(
        BuildConfigurationValue configurationValue, int id) {
      ConfigurationForOutput configurationForOutput =
          ConfigurationForOutput.getConfigurationForOutput(configurationValue);
      Configuration.Builder configBuilder = Configuration.newBuilder();

      for (FragmentForOutput fragmentForOutput : configurationForOutput.getFragments()) {
        AnalysisProtosV2.Fragment.Builder fragment = AnalysisProtosV2.Fragment.newBuilder();
        fragment.setName(fragmentForOutput.getName());
        fragmentForOutput.getFragmentOptions().forEach(fragment::addFragmentOptionNames);
        configBuilder.addFragments(fragment);
      }

      for (FragmentOptionsForOutput fragmentOptionsForOutput :
          configurationForOutput.getFragmentOptions()) {
        AnalysisProtosV2.FragmentOptions.Builder fragmentOptions =
            AnalysisProtosV2.FragmentOptions.newBuilder()
                .setName(fragmentOptionsForOutput.getName());
        for (Map.Entry<String, String> option : fragmentOptionsForOutput.getOptions().entrySet()) {
          AnalysisProtosV2.Option optionProto =
              AnalysisProtosV2.Option.newBuilder()
                  .setName(option.getKey())
                  .setValue(option.getValue())
                  .build();
          fragmentOptions.addOptions(optionProto);
        }
        configBuilder.addFragmentOptions(fragmentOptions.build());
      }

      String checksum = configurationValue.getEventId().getConfiguration().getId();
      BuildEventStreamProtos.Configuration configProto =
          configurationValue
              .toBuildEvent()
              .asStreamProto(/* unusedConverters= */ null)
              .getConfiguration();

      return configBuilder
          .setChecksum(checksum)
          .setMnemonic(configProto.getMnemonic())
          .setPlatformName(configProto.getPlatformName())
          .setId(id)
          .setIsTool(configProto.getIsTool())
          .build();
    }
  }

  private CqueryResult.Builder cqueryResultBuilder;
  private final OutputType outputType;
  private final AspectResolver resolver;
  private final SkyframeExecutor skyframeExecutor;
  private final ConfigurationCache configurationCache =
      new ConfigurationCache(this::getConfiguration);
  private final JsonFormat.Printer jsonPrinter = JsonFormat.printer();

  private final LabelPrinter labelPrinter;
  private CqueryNode currentTarget;

  ProtoOutputFormatterCallback(
      ExtendedEventHandler eventHandler,
      CqueryOptions options,
      OutputStream out,
      SkyframeExecutor skyframeExecutor,
      TargetAccessor<CqueryNode> accessor,
      AspectResolver resolver,
      OutputType outputType,
      LabelPrinter labelPrinter) {
    super(eventHandler, options, out, skyframeExecutor, accessor, /* uniquifyResults= */ false);
    this.outputType = outputType;
    this.skyframeExecutor = skyframeExecutor;
    this.resolver = resolver;
    this.labelPrinter = labelPrinter;
  }

  @Override
  public void start() {
    cqueryResultBuilder = AnalysisProtosV2.CqueryResult.newBuilder();
  }

  private static QueryResult queryResultFromCqueryResult(CqueryResultOrBuilder cqueryResult) {
    Build.QueryResult.Builder queryResult = Build.QueryResult.newBuilder();
    cqueryResult.getResultsList().forEach(ct -> queryResult.addTarget(ct.getTarget()));
    return queryResult.build();
  }

  @Override
  public void close(boolean failFast) throws IOException {
    if (failFast || printStream == null) {
      return;
    }

    // There are a few cases that affect the shape of the output:
    //   1. --output=proto|textproto|jsonproto --proto:include_configurations =>
    //        Writes a single CqueryResult containing all the ConfiguredTarget(s) and
    //        Configuration(s) in the specified output format.
    //   2. --output=streamed_proto --proto:include_configurations =>
    //        Writes multiple length delimited CqueryResult protos, each containing a single
    //        ConfiguredTarget or Configuration.
    //   3. --output=proto|textproto|jsonproto --noproto:include_configurations =>
    //        Writes a single QueryResult containing all the corresponding Target(s) in the
    //        specified output format.
    //   4.--output=streamed_proto --noproto:include_configurations =>
    //        Writes multiple length delimited QueryResult protos, each containing a single Target.
    switch (outputType) {
      case BINARY, TEXT, JSON -> {
        // Only at the end, we write the entire CqueryResult / QueryResult is written all together.
        if (options.protoIncludeConfigurations) {
          cqueryResultBuilder.addAllConfigurations(configurationCache.getConfigurations());
        }
        writeData(
            options.protoIncludeConfigurations
                ? cqueryResultBuilder.build()
                : queryResultFromCqueryResult(cqueryResultBuilder));
      }
      case DELIMITED_BINARY -> {
        if (options.protoIncludeConfigurations) {
          // The wrapped CqueryResult + ConfiguredTarget are already written in
          // {@link #processOutput}, so we just need to write the Configuration(s) each wrapped in
          // a CqueryResult.
          for (Configuration configuration : configurationCache.getConfigurations()) {
            writeData(
                AnalysisProtosV2.CqueryResult.newBuilder()
                    .addConfigurations(configuration)
                    .build());
          }
        }
      }
    }

    outputStream.flush();
    printStream.flush();
  }

  private void writeData(Message message) throws IOException {
    switch (outputType) {
      case BINARY -> {
        // Avoid a crash due to a failed precondition check in protobuf.
        if (message.getSerializedSize() < 0) {
          throw new IOException(
              "--output=proto does not support results larger than 2GB, use --output=streamed_proto"
                  + " instead.");
        }
        message.writeTo(outputStream);
      }
      case DELIMITED_BINARY -> message.writeDelimitedTo(outputStream);
      case TEXT -> TextFormat.printer().print(message, printStream);
      case JSON -> {
        jsonPrinter.appendTo(message, printStream);
        printStream.append('\n');
      }
    }
  }

  @Override
  public String getName() {
    return outputType.formatName();
  }

  @Override
  public void processOutput(Iterable<CqueryNode> partialResult)
      throws InterruptedException, IOException {
    ConfiguredProtoOutputFormatter formatter = new ConfiguredProtoOutputFormatter();
    formatter.setOptions(options, resolver, skyframeExecutor.getDigestFunction().getHashFunction());
    for (CqueryNode keyedConfiguredTarget : partialResult) {
      AnalysisProtosV2.ConfiguredTarget.Builder builder =
          AnalysisProtosV2.ConfiguredTarget.newBuilder();
      // Re: testing. Since this formatter relies on the heavily tested ProtoOutputFormatter class
      // for all its work with targets, ProtoOutputFormatterCallbackTest doesn't test any of the
      // logic in this next line. If this were to change (i.e. we manipulate targets any further),
      // we will want to add relevant tests.
      currentTarget = keyedConfiguredTarget;
      Target target = accessor.getTarget(keyedConfiguredTarget);
      Build.Target.Builder targetBuilder =
          formatter.toTargetProtoBuffer(target, labelPrinter).toBuilder();
      if (target instanceof Rule && !Transitions.NONE.equals(options.transitions)) {
        // To set configured_rule_input dependencies, use ConfiguredTargetAccessor.getPrerequisites.
        // Note that both that and CqueryTransitionResolver can get a target's direct deps. We use
        // the former because it implements cquery's "canonical" view of the dependency graph, which
        // might not match the underlying Skyframe graph. For example, without
        // QueryEnvironment.Setting#EXPLICIT_ASPECTS, if CT //foo depends on aspect A which has
        // implicit dep //dep, cquery outputs //dep as a direct dep of //foo. Even though this isn't
        // technically true according to the Skyframe graph. If we used CqueryTransitionResolver,
        // which directly queries Skyframe, it wouldn't return //dep.
        //
        // cquery users should always view the graph according to cquery's canonical interpretation.
        for (CqueryNode dep : accessor.getPrerequisites(keyedConfiguredTarget)) {
          ConfiguredRuleInput.Builder configuredRuleInput =
              Build.ConfiguredRuleInput.newBuilder()
                  .setLabel(labelPrinter.toString(dep.getOriginalLabel()));
          if (dep.getConfigurationChecksum() != null) {
            configuredRuleInput
                .setConfigurationChecksum(dep.getConfigurationChecksum())
                .setConfigurationId(
                    configurationCache.getId(dep.getConfigurationKey().getOptions()));
          }
          targetBuilder.getRuleBuilder().addConfiguredRuleInput(configuredRuleInput);
        }
      }

      builder.setTarget(targetBuilder);

      if (options.protoIncludeConfigurations) {
        String checksum = keyedConfiguredTarget.getConfigurationChecksum();
        builder.setConfiguration(
            AnalysisProtosV2.Configuration.newBuilder().setChecksum(String.valueOf(checksum)));

        var configuredTargetKey = ConfiguredTargetKey.fromConfiguredTarget(keyedConfiguredTarget);
        // Some targets don't have a configuration, e.g. InputFileConfiguredTarget
        if (configuredTargetKey != null) {
          BuildConfigurationKey configurationKey = configuredTargetKey.getConfigurationKey();
          if (configurationKey != null) {
            BuildConfigurationValue configuration = getConfiguration(configurationKey);
            int id = configurationCache.getId(configuration);
            builder.setConfigurationId(id);
          }
        }
      }

      if (outputType == OutputType.DELIMITED_BINARY) {
        // If --proto:include_configurations, we wrap the single ConfiguredTarget in a CqueryResult.
        // If --noproto:include_configurations, we wrap the single Target in a QueryResult.
        // Then we write either result delimited to the stream.
        writeData(
            options.protoIncludeConfigurations
                ? CqueryResult.newBuilder().addResults(builder).build()
                : QueryResult.newBuilder().addTarget(builder.getTarget()).build());
      } else {
        // Except --output=streamed_proto, all other output types require they be wrapped in a
        // CqueryResult or QueryResult. So we instead of writing straight to the stream, we
        // aggregate the results in a CqueryResult.Builder before writing in {@link #close}.
        cqueryResultBuilder.addResults(builder.build());
      }
    }
  }

  private class ConfiguredProtoOutputFormatter extends ProtoOutputFormatter {
    @Override
    protected void addAttributes(
        Build.Rule.Builder rulePb,
        Rule rule,
        Object extraDataForAttrHash,
        LabelPrinter labelPrinter) {
      // We know <code>currentTarget</code> will be either an AliasConfiguredTarget or
      // RuleConfiguredTarget,
      // because this method is only triggered in ProtoOutputFormatter.toTargetProtoBuffer when
      // the target in currentTarget is an instanceof Rule.
      ImmutableMap<Label, ConfigMatchingProvider> configConditions =
          currentTarget.getConfigConditions();
      ConfiguredAttributeMapper attributeMapper =
          ConfiguredAttributeMapper.of(
              rule,
              configConditions,
              currentTarget.getConfigurationKey().getOptionsChecksum(),
              /* alwaysSucceed= */ false);
      for (Attribute attr : sortAttributes(rule.getAttributes())) {
        if (!shouldIncludeAttribute(rule, attr)) {
          continue;
        }
        Object attributeValue = attributeMapper.get(attr.getName(), attr.getType());
        Build.Attribute serializedAttribute =
            AttributeFormatter.getAttributeProto(
                attr,
                attributeValue,
                rule.isAttributeValueExplicitlySpecified(attr),
                /* encodeBooleanAndTriStateAsIntegerAndString= */ true,
                /* sourceAspect= */ null,
                includeAttributeSourceAspects,
                labelPrinter);
        rulePb.addAttribute(serializedAttribute);
      }
    }
  }

  static List<Attribute> sortAttributes(Iterable<Attribute> attributes) {
    return Ordering.from(Comparator.comparing(Attribute::getName)).sortedCopy(attributes);
  }
}
