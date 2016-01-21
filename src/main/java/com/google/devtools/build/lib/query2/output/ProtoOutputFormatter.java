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
package com.google.devtools.build.lib.query2.output;

import static com.google.devtools.build.lib.query2.proto.proto2api.Build.Target.Discriminator.ENVIRONMENT_GROUP;
import static com.google.devtools.build.lib.query2.proto.proto2api.Build.Target.Discriminator.GENERATED_FILE;
import static com.google.devtools.build.lib.query2.proto.proto2api.Build.Target.Discriminator.PACKAGE_GROUP;
import static com.google.devtools.build.lib.query2.proto.proto2api.Build.Target.Discriminator.RULE;
import static com.google.devtools.build.lib.query2.proto.proto2api.Build.Target.Discriminator.SOURCE_FILE;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMultimap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.graph.Digraph;
import com.google.devtools.build.lib.packages.AggregatingAttributeMapper;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.AttributeSerializer;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.DependencyFilter;
import com.google.devtools.build.lib.packages.EnvironmentGroup;
import com.google.devtools.build.lib.packages.InputFile;
import com.google.devtools.build.lib.packages.OutputFile;
import com.google.devtools.build.lib.packages.PackageGroup;
import com.google.devtools.build.lib.packages.ProtoUtils;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.query2.FakeSubincludeTarget;
import com.google.devtools.build.lib.query2.engine.OutputFormatterCallback;
import com.google.devtools.build.lib.query2.output.AspectResolver.BuildFileDependencyMode;
import com.google.devtools.build.lib.query2.output.OutputFormatter.AbstractUnorderedFormatter;
import com.google.devtools.build.lib.query2.output.QueryOptions.OrderOutput;
import com.google.devtools.build.lib.query2.proto.proto2api.Build;
import com.google.devtools.build.lib.query2.proto.proto2api.Build.GeneratedFile;
import com.google.devtools.build.lib.query2.proto.proto2api.Build.QueryResult.Builder;
import com.google.devtools.build.lib.query2.proto.proto2api.Build.SourceFile;
import com.google.devtools.build.lib.syntax.Environment;

import java.io.IOException;
import java.io.PrintStream;
import java.util.Collection;
import java.util.LinkedHashSet;
import java.util.Map.Entry;
import java.util.Set;

/**
 * An output formatter that outputs a protocol buffer representation
 * of a query result and outputs the proto bytes to the output print stream.
 * By taking the bytes and calling {@code mergeFrom()} on a
 * {@code Build.QueryResult} object the full result can be reconstructed.
 */
public class ProtoOutputFormatter extends AbstractUnorderedFormatter {

  /**
   * A special attribute name for the rule implementation hash code.
   */
  public static final String RULE_IMPLEMENTATION_HASH_ATTR_NAME = "$rule_implementation_hash";

  private transient DependencyFilter dependencyFilter;
  protected transient AspectResolver aspectResolver;

  private boolean relativeLocations = false;
  protected boolean includeDefaultValues = true;

  protected void setDependencyFilter(QueryOptions options) {
    this.dependencyFilter = OutputFormatter.getDependencyFilter(options);
  }

  @Override
  public String getName() {
    return "proto";
  }

  @Override
  public OutputFormatterCallback<Target> createStreamCallback(QueryOptions options,
      final PrintStream out, AspectResolver aspectResolver) {
    relativeLocations = options.relativeLocations;
    this.aspectResolver = aspectResolver;
    this.includeDefaultValues = options.protoIncludeDefaultValues;
    setDependencyFilter(options);

    return new OutputFormatterCallback<Target>() {

      private Builder queryResult;

      @Override
      public void start() {
        queryResult = Build.QueryResult.newBuilder();
      }

      @Override
      protected void processOutput(Iterable<Target> partialResult)
          throws IOException, InterruptedException {

        for (Target target : partialResult) {
          queryResult.addTarget(toTargetProtoBuffer(target));
        }
      }

      @Override
      public void close() throws IOException {
        queryResult.build().writeTo(out);
      }
    };
  }

  private static Iterable<Target> getSortedLabels(Digraph<Target> result) {
    return Iterables.transform(
        result.getTopologicalOrder(new TargetOrdering()), EXTRACT_NODE_LABEL);
  }

  @Override
  protected Iterable<Target> getOrderedTargets(Digraph<Target> result, QueryOptions options) {
    return options.orderOutput == OrderOutput.FULL ? getSortedLabels(result) : result.getLabels();
  }

  /**
   * Converts a logical Target object into a Target protobuffer.
   */
  protected Build.Target toTargetProtoBuffer(Target target)
      throws InterruptedException {
    Build.Target.Builder targetPb = Build.Target.newBuilder();

    String location = getLocation(target, relativeLocations);
    if (target instanceof Rule) {
      Rule rule = (Rule) target;
      Build.Rule.Builder rulePb = Build.Rule.newBuilder()
          .setName(rule.getLabel().toString())
          .setRuleClass(rule.getRuleClass());
      if (includeLocation()) {
        rulePb.setLocation(location);
      }

      for (Attribute attr : rule.getAttributes()) {
        if (!includeDefaultValues && !rule.isAttributeValueExplicitlySpecified(attr)
            || !includeAttribute(rule, attr)) {
          continue;
        }
        Iterable<Object> possibleAttributeValues =
            AggregatingAttributeMapper.of(rule).getPossibleAttributeValues(rule, attr);
        Object flattenedAttributeValue =
            AggregatingAttributeMapper.flattenAttributeValues(
                attr.getType(), possibleAttributeValues);
        Build.Attribute serializedAttribute =
            AttributeSerializer.getAttributeProto(
                attr,
                flattenedAttributeValue,
                rule.isAttributeValueExplicitlySpecified(attr),
                /*includeGlobs=*/ false,
                /*encodeBooleanAndTriStateAsIntegerAndString=*/ true);
        rulePb.addAttribute(serializedAttribute);
      }

      postProcess(rule, rulePb);

      Environment env = rule.getRuleClassObject().getRuleDefinitionEnvironment();
      if (env != null && includeRuleDefinitionEnvironment()) {
        // The RuleDefinitionEnvironment is always defined for Skylark rules and
        // always null for non Skylark rules.
        rulePb.addAttribute(
            Build.Attribute.newBuilder()
                .setName(RULE_IMPLEMENTATION_HASH_ATTR_NAME)
                .setType(ProtoUtils.getDiscriminatorFromType(
                    com.google.devtools.build.lib.syntax.Type.STRING))
                .setStringValue(env.getTransitiveContentHashCode()));
      }

      ImmutableMultimap<Attribute, Label> aspectsDependencies =
          aspectResolver.computeAspectDependencies(target);
      // Add information about additional attributes from aspects.
      for (Entry<Attribute, Collection<Label>> entry : aspectsDependencies.asMap().entrySet()) {
        Attribute attribute = entry.getKey();
        if (!includeAttribute(rule, attribute)) {
          continue;
        }
        Collection<Label> labels = entry.getValue();
        Object attributeValue = getAspectAttributeValue(attribute, labels);
        Build.Attribute serializedAttribute =
            AttributeSerializer.getAttributeProto(
                attribute,
                attributeValue,
                /*explicitlySpecified=*/ false,
                /*includeGlobs=*/ false,
                /*encodeBooleanAndTriStateAsIntegerAndString=*/ true);
        rulePb.addAttribute(serializedAttribute);
      }
      if (includeRuleInputsAndOutputs()) {
        // Add all deps from aspects as rule inputs of current target.
        for (Label label : aspectsDependencies.values()) {
          rulePb.addRuleInput(label.toString());
        }

        // Include explicit elements for all direct inputs and outputs of a rule;
        // this goes beyond what is available from the attributes above, since it
        // may also (depending on options) include implicit outputs,
        // host-configuration outputs, and default values.
        for (Label label : rule.getLabels(dependencyFilter)) {
          rulePb.addRuleInput(label.toString());
        }
        for (OutputFile outputFile : rule.getOutputFiles()) {
          Label fileLabel = outputFile.getLabel();
          rulePb.addRuleOutput(fileLabel.toString());
        }
      }
      for (String feature : rule.getFeatures()) {
        rulePb.addDefaultSetting(feature);
      }

      targetPb.setType(RULE);
      targetPb.setRule(rulePb);
    } else if (target instanceof OutputFile) {
      OutputFile outputFile = (OutputFile) target;
      Label label = outputFile.getLabel();

      Rule generatingRule = outputFile.getGeneratingRule();
      GeneratedFile.Builder output =
          GeneratedFile.newBuilder()
                       .setGeneratingRule(generatingRule.getLabel().toString())
                       .setName(label.toString());

      if (includeLocation()) {
        output.setLocation(location);
      }
      targetPb.setType(GENERATED_FILE);
      targetPb.setGeneratedFile(output.build());
    } else if (target instanceof InputFile) {
      InputFile inputFile = (InputFile) target;
      Label label = inputFile.getLabel();

      Build.SourceFile.Builder input = Build.SourceFile.newBuilder()
          .setName(label.toString());

      if (includeLocation()) {
        input.setLocation(location);
      }

      if (inputFile.getName().equals("BUILD")) {
        Set<Label> subincludeLabels = new LinkedHashSet<>();
        subincludeLabels.addAll(aspectResolver == null
            ? inputFile.getPackage().getSubincludeLabels()
            : aspectResolver.computeBuildFileDependencies(
                inputFile.getPackage(), BuildFileDependencyMode.SUBINCLUDE));
        subincludeLabels.addAll(aspectResolver == null
            ? inputFile.getPackage().getSkylarkFileDependencies()
            : aspectResolver.computeBuildFileDependencies(
                inputFile.getPackage(), BuildFileDependencyMode.SKYLARK));

        for (Label skylarkFileDep : subincludeLabels) {
          input.addSubinclude(skylarkFileDep.toString());
        }

        for (String feature : inputFile.getPackage().getFeatures()) {
          input.addFeature(feature);
        }

        input.setPackageContainsErrors(inputFile.getPackage().containsErrors());
      }

      for (Label visibilityDependency : target.getVisibility().getDependencyLabels()) {
        input.addPackageGroup(visibilityDependency.toString());
      }

      for (Label visibilityDeclaration : target.getVisibility().getDeclaredLabels()) {
        input.addVisibilityLabel(visibilityDeclaration.toString());
      }

      targetPb.setType(SOURCE_FILE);
      targetPb.setSourceFile(input);
    } else if (target instanceof FakeSubincludeTarget) {
      Label label = target.getLabel();
      SourceFile.Builder input = SourceFile.newBuilder()
                                           .setName(label.toString());

      if (includeLocation()) {
        input.setLocation(location);
      }
      targetPb.setType(SOURCE_FILE);
      targetPb.setSourceFile(input.build());
    } else if (target instanceof PackageGroup) {
      PackageGroup packageGroup = (PackageGroup) target;
      Build.PackageGroup.Builder packageGroupPb = Build.PackageGroup.newBuilder()
          .setName(packageGroup.getLabel().toString());
      for (String containedPackage : packageGroup.getContainedPackages()) {
        packageGroupPb.addContainedPackage(containedPackage);
      }
      for (Label include : packageGroup.getIncludes()) {
        packageGroupPb.addIncludedPackageGroup(include.toString());
      }

      targetPb.setType(PACKAGE_GROUP);
      targetPb.setPackageGroup(packageGroupPb);
    } else if (target instanceof EnvironmentGroup) {
      EnvironmentGroup envGroup = (EnvironmentGroup) target;
      Build.EnvironmentGroup.Builder envGroupPb =
          Build.EnvironmentGroup
              .newBuilder()
              .setName(envGroup.getLabel().toString());
      for (Label env : envGroup.getEnvironments()) {
        envGroupPb.addEnvironment(env.toString());
      }
      for (Label defaultEnv : envGroup.getDefaults()) {
        envGroupPb.addDefault(defaultEnv.toString());
      }
      targetPb.setType(ENVIRONMENT_GROUP);
      targetPb.setEnvironmentGroup(envGroupPb);
    } else {
      throw new IllegalArgumentException(target.toString());
    }

    return targetPb.build();
  }

  private static Object getAspectAttributeValue(Attribute attribute, Collection<Label> labels) {
    com.google.devtools.build.lib.syntax.Type<?> attributeType = attribute.getType();
    if (attributeType.equals(BuildType.LABEL)) {
      Preconditions.checkState(labels.size() == 1, "attribute=%s, labels=%s", attribute, labels);
      return Iterables.getOnlyElement(labels);
    } else {
      Preconditions.checkState(
          attributeType.equals(BuildType.LABEL_LIST),
          "attribute=%s, type=%s, labels=%s",
          attribute,
          attributeType,
          labels);
      return labels;
    }
  }

  /** Further customize the proto output */
  protected void postProcess(Rule rule, Build.Rule.Builder rulePb) { }

  /** Filter out some attributes */
  protected boolean includeAttribute(Rule rule, Attribute attr) {
    return true;
  }

  protected boolean includeRuleDefinitionEnvironment() {
    return true;
  }

  protected boolean includeRuleInputsAndOutputs() {
    return true;
  }

  protected boolean includeLocation() {
    return true;
  }
}
