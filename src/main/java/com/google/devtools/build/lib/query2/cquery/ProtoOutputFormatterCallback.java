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
import com.google.common.collect.Maps;
import com.google.common.collect.Ordering;
import com.google.devtools.build.lib.actions.BuildConfigurationEvent;
import com.google.devtools.build.lib.analysis.AnalysisProtosV2;
import com.google.devtools.build.lib.analysis.AnalysisProtosV2.Configuration;
import com.google.devtools.build.lib.analysis.AnalysisProtosV2.CqueryResult;
import com.google.devtools.build.lib.analysis.AnalysisProtosV2.CqueryResultOrBuilder;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.ConfigMatchingProvider;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.AttributeFormatter;
import com.google.devtools.build.lib.packages.ConfiguredAttributeMapper;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleClassProvider;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.query2.cquery.CqueryOptions.Transitions;
import com.google.devtools.build.lib.query2.cquery.CqueryTransitionResolver.EvaluateException;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.TargetAccessor;
import com.google.devtools.build.lib.query2.proto.proto2api.Build;
import com.google.devtools.build.lib.query2.proto.proto2api.Build.QueryResult;
import com.google.devtools.build.lib.query2.query.aspectresolvers.AspectResolver;
import com.google.devtools.build.lib.query2.query.output.ProtoOutputFormatter;
import com.google.devtools.build.lib.skyframe.BuildConfigurationKey;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import com.google.protobuf.Message;
import com.google.protobuf.TextFormat;
import com.google.protobuf.util.JsonFormat;
import java.io.IOException;
import java.io.OutputStream;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

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
    private final Map<BuildConfigurationEvent, Integer> cache = new HashMap<>();

    public int getId(BuildConfigurationEvent buildConfigurationEvent) {
      return cache.computeIfAbsent(buildConfigurationEvent, event -> cache.size() + 1);
    }

    public ImmutableList<Configuration> getConfigurations() {
      return cache.entrySet().stream()
          .map(
              entry -> {
                BuildConfigurationEvent event = entry.getKey();
                String checksum = event.getEventId().getConfiguration().getId();
                BuildEventStreamProtos.Configuration configProto =
                    event.asStreamProto(/* unusedConverters= */ null).getConfiguration();

                return AnalysisProtosV2.Configuration.newBuilder()
                    .setChecksum(checksum)
                    .setMnemonic(configProto.getMnemonic())
                    .setPlatformName(configProto.getPlatformName())
                    .setId(entry.getValue())
                    .setIsTool(configProto.getIsTool())
                    .build();
              })
          .collect(toImmutableList());
    }
  }

  private CqueryResult.Builder cqueryResultBuilder;
  private final OutputType outputType;
  private final AspectResolver resolver;
  private final SkyframeExecutor skyframeExecutor;
  private final ConfigurationCache configurationCache = new ConfigurationCache();
  private final JsonFormat.Printer jsonPrinter = JsonFormat.printer();
  private final RuleClassProvider ruleClassProvider;

  private final Map<Label, Target> partialResultMap;
  private ConfiguredTarget currentTarget;

  ProtoOutputFormatterCallback(
      ExtendedEventHandler eventHandler,
      CqueryOptions options,
      OutputStream out,
      SkyframeExecutor skyframeExecutor,
      TargetAccessor<ConfiguredTarget> accessor,
      AspectResolver resolver,
      OutputType outputType,
      RuleClassProvider ruleClassProvider) {
    super(eventHandler, options, out, skyframeExecutor, accessor, /*uniquifyResults=*/ false);
    this.outputType = outputType;
    this.skyframeExecutor = skyframeExecutor;
    this.resolver = resolver;
    this.ruleClassProvider = ruleClassProvider;
    this.partialResultMap = Maps.newHashMap();
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
      case BINARY:
      case TEXT:
      case JSON:
        // Only at the end, we write the entire CqueryResult / QueryResult is written all together.
        if (options.protoIncludeConfigurations) {
          cqueryResultBuilder.addAllConfigurations(configurationCache.getConfigurations());
        }
        writeData(
            options.protoIncludeConfigurations
                ? cqueryResultBuilder.build()
                : queryResultFromCqueryResult(cqueryResultBuilder));
        break;
      case DELIMITED_BINARY:
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
        break;
    }

    outputStream.flush();
    printStream.flush();
  }

  private void writeData(Message message) throws IOException {
    switch (outputType) {
      case BINARY:
        message.writeTo(outputStream);
        break;
      case DELIMITED_BINARY:
        message.writeDelimitedTo(outputStream);
        break;
      case TEXT:
        TextFormat.printer().print(message, printStream);
        break;
      case JSON:
        jsonPrinter.appendTo(message, printStream);
        printStream.append('\n');
        break;
    }
  }

  @Override
  public String getName() {
    return outputType.formatName();
  }

  @Override
  public void processOutput(Iterable<ConfiguredTarget> partialResult)
      throws InterruptedException, IOException {
    partialResult.forEach(
        kct -> partialResultMap.put(kct.getOriginalLabel(), accessor.getTarget(kct)));

    CqueryTransitionResolver transitionResolver =
        new CqueryTransitionResolver(
            eventHandler,
            accessor,
            this,
            ruleClassProvider,
            skyframeExecutor.getSkyframeBuildView().getStarlarkTransitionCache());

    ConfiguredProtoOutputFormatter formatter = new ConfiguredProtoOutputFormatter();
    formatter.setOptions(options, resolver, skyframeExecutor.getDigestFunction().getHashFunction());
    for (ConfiguredTarget keyedConfiguredTarget : partialResult) {
      AnalysisProtosV2.ConfiguredTarget.Builder builder =
          AnalysisProtosV2.ConfiguredTarget.newBuilder();

      // Re: testing. Since this formatter relies on the heavily tested ProtoOutputFormatter class
      // for all its work with targets, ProtoOuputFormatterCallbackTest doesn't test any of the
      // logic in this next line. If this were to change (i.e. we manipulate targets any further),
      // we will want to add relevant tests.
      currentTarget = keyedConfiguredTarget;
      Target target = accessor.getTarget(keyedConfiguredTarget);
      Build.Target.Builder targetBuilder = formatter.toTargetProtoBuffer(target).toBuilder();
      if (target instanceof Rule && !Transitions.NONE.equals(options.transitions)) {
        try {
          for (CqueryTransitionResolver.ResolvedTransition resolvedTransition :
              transitionResolver.dependencies(keyedConfiguredTarget)) {
            if (resolvedTransition.options().isEmpty()) {
              targetBuilder
                  .getRuleBuilder()
                  .addConfiguredRuleInput(
                      Build.ConfiguredRuleInput.newBuilder()
                          .setLabel(resolvedTransition.label().toString()));
            } else {
              for (BuildOptions options : resolvedTransition.options()) {
                BuildConfigurationEvent buildConfigurationEvent =
                    getConfiguration(BuildConfigurationKey.withoutPlatformMapping(options))
                        .toBuildEvent();
                int configurationId = configurationCache.getId(buildConfigurationEvent);

                targetBuilder
                    .getRuleBuilder()
                    .addConfiguredRuleInput(
                        Build.ConfiguredRuleInput.newBuilder()
                            .setLabel(resolvedTransition.label().toString())
                            .setConfigurationChecksum(options.checksum())
                            .setConfigurationId(configurationId));
              }
            }
          }
        } catch (EvaluateException e) {
          // This is an abuse of InterruptedException.
          throw new InterruptedException(e.getMessage());
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
            BuildConfigurationEvent buildConfigurationEvent =
                getConfiguration(configurationKey).toBuildEvent();
            int id = configurationCache.getId(buildConfigurationEvent);
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
        Build.Rule.Builder rulePb, Rule rule, Object extraDataForAttrHash) {
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
                includeAttributeSourceAspects);
        rulePb.addAttribute(serializedAttribute);
      }
    }
  }

  static List<Attribute> sortAttributes(Iterable<Attribute> attributes) {
    return Ordering.from(Comparator.comparing(Attribute::getName)).sortedCopy(attributes);
  }
}
