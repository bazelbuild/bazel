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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMultimap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.graph.Digraph;
import com.google.devtools.build.lib.packages.AggregatingAttributeMapper;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.AttributeFormatter;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.EnvironmentGroup;
import com.google.devtools.build.lib.packages.InputFile;
import com.google.devtools.build.lib.packages.OutputFile;
import com.google.devtools.build.lib.packages.PackageGroup;
import com.google.devtools.build.lib.packages.ProtoUtils;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.query2.FakeLoadTarget;
import com.google.devtools.build.lib.query2.engine.OutputFormatterCallback;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment;
import com.google.devtools.build.lib.query2.engine.SynchronizedDelegatingOutputFormatterCallback;
import com.google.devtools.build.lib.query2.engine.ThreadSafeOutputFormatterCallback;
import com.google.devtools.build.lib.query2.output.AspectResolver.BuildFileDependencyMode;
import com.google.devtools.build.lib.query2.output.OutputFormatter.AbstractUnorderedFormatter;
import com.google.devtools.build.lib.query2.output.QueryOptions.OrderOutput;
import com.google.devtools.build.lib.query2.proto.proto2api.Build;
import com.google.devtools.build.lib.query2.proto.proto2api.Build.GeneratedFile;
import com.google.devtools.build.lib.query2.proto.proto2api.Build.QueryResult.Builder;
import com.google.devtools.build.lib.query2.proto.proto2api.Build.SourceFile;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.Type;
import java.io.IOException;
import java.io.OutputStream;
import java.util.Collection;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import javax.annotation.Nullable;

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

  @SuppressWarnings("unchecked")
  private static final ImmutableSet<Type<?>> SCALAR_TYPES =
      ImmutableSet.<Type<?>>of(
          Type.INTEGER, Type.STRING, BuildType.LABEL, BuildType.NODEP_LABEL, BuildType.OUTPUT,
          Type.BOOLEAN, BuildType.TRISTATE, BuildType.LICENSE);

  private boolean relativeLocations = false;
  protected boolean includeDefaultValues = true;
  private boolean flattenSelects = true;

  protected void setDependencyFilter(QueryOptions options) {
    this.dependencyFilter = OutputFormatter.getDependencyFilter(options);
  }

  @Override
  public String getName() {
    return "proto";
  }

  @Override
  public void setOptions(QueryOptions options, AspectResolver aspectResolver) {
    super.setOptions(options, aspectResolver);
    this.relativeLocations = options.relativeLocations;
    this.includeDefaultValues = options.protoIncludeDefaultValues;
    this.flattenSelects = options.protoFlattenSelects;
  }

  @Override
  public OutputFormatterCallback<Target> createPostFactoStreamCallback(
      final OutputStream out, final QueryOptions options) {
    return new OutputFormatterCallback<Target>() {

      private Builder queryResult;

      @Override
      public void start() {
        queryResult = Build.QueryResult.newBuilder();
      }

      @Override
      public void processOutput(Iterable<Target> partialResult)
          throws IOException, InterruptedException {

        for (Target target : partialResult) {
          queryResult.addTarget(toTargetProtoBuffer(target));
        }
      }

      @Override
      public void close(boolean failFast) throws IOException {
        if (!failFast) {
          queryResult.build().writeTo(out);
        }
      }
    };
  }

  @Override
  public ThreadSafeOutputFormatterCallback<Target> createStreamCallback(
      OutputStream out, QueryOptions options, QueryEnvironment<?> env) {
    return createStreamCallback(out, options);
  }

  @VisibleForTesting
  public ThreadSafeOutputFormatterCallback<Target> createStreamCallback(
      OutputStream out, QueryOptions options) {
    return new SynchronizedDelegatingOutputFormatterCallback<>(
        createPostFactoStreamCallback(out, options));
  }

  private static Iterable<Target> getSortedLabels(Digraph<Target> result) {
    return Iterables.transform(
        result.getTopologicalOrder(new TargetOrdering()), EXTRACT_NODE_LABEL);
  }

  @Override
  protected Iterable<Target> getOrderedTargets(Digraph<Target> result, QueryOptions options) {
    return options.orderOutput == OrderOutput.FULL ? getSortedLabels(result) : result.getLabels();
  }

  /** Converts a logical {@link Target} object into a {@link Build.Target} protobuffer. */
  @VisibleForTesting
  public Build.Target toTargetProtoBuffer(Target target) throws InterruptedException {
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
      Map<Attribute, Build.Attribute> serializedAttributes = Maps.newHashMap();
      AggregatingAttributeMapper attributeMapper = AggregatingAttributeMapper.of(rule);
      for (Attribute attr : rule.getAttributes()) {
        if ((!includeDefaultValues && !rule.isAttributeValueExplicitlySpecified(attr))
            || !includeAttribute(rule, attr)) {
          continue;
        }
        Object attributeValue;
        if (flattenSelects || !attributeMapper.isConfigurable(attr.getName())) {
          attributeValue =
              flattenAttributeValues(attr.getType(), getPossibleAttributeValues(rule, attr));
        } else {
          attributeValue = attributeMapper.getSelectorList(attr.getName(), attr.getType());
        }
        Build.Attribute serializedAttribute =
            AttributeFormatter.getAttributeProto(
                attr,
                attributeValue,
                rule.isAttributeValueExplicitlySpecified(attr),
                /*encodeBooleanAndTriStateAsIntegerAndString=*/ true);
        rulePb.addAttribute(serializedAttribute);
        serializedAttributes.put(attr, serializedAttribute);
      }

      postProcess(rule, rulePb, serializedAttributes);

      Environment env = rule.getRuleClassObject().getRuleDefinitionEnvironment();
      if (env != null && includeRuleDefinitionEnvironment()) {
        // The RuleDefinitionEnvironment is always defined for Skylark rules and
        // always null for non Skylark rules.
        rulePb.addAttribute(
            Build.Attribute.newBuilder()
                .setName(RULE_IMPLEMENTATION_HASH_ATTR_NAME)
                .setType(ProtoUtils.getDiscriminatorFromType(Type.STRING))
                .setStringValue(env.getTransitiveContentHashCode()));
      }

      ImmutableMultimap<Attribute, Label> aspectsDependencies =
          aspectResolver.computeAspectDependencies(target, dependencyFilter);
      // Add information about additional attributes from aspects.
      for (Entry<Attribute, Collection<Label>> entry : aspectsDependencies.asMap().entrySet()) {
        Attribute attribute = entry.getKey();
        Collection<Label> labels = entry.getValue();
        if (!includeAspectAttribute(attribute, labels)) {
          continue;
        }
        Object attributeValue = getAspectAttributeValue(attribute, labels);
        Build.Attribute serializedAttribute =
            AttributeFormatter.getAttributeProto(
                attribute,
                attributeValue,
                /*explicitlySpecified=*/ false,
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
    } else if (target instanceof FakeLoadTarget) {
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
    Type<?> attributeType = attribute.getType();
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
  protected void postProcess(Rule rule, Build.Rule.Builder rulePb, Map<Attribute,
      Build.Attribute> serializedAttributes) { }

  /** Filter out some attributes */
  protected boolean includeAttribute(Rule rule, Attribute attr) {
    return true;
  }

  /** Allow filtering of aspect attributes. */
  protected boolean includeAspectAttribute(Attribute attr, Collection<Label> value) {
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

  /**
   * Coerces the list {@param possibleValues} of values of type {@param attrType} to a single
   * value of that type, in the following way:
   *
   * <p>If the list contains a single value, return that value.
   *
   * <p>If the list contains zero or multiple values and the type is a scalar type, return {@code
   * null}.
   *
   * <p>If the list contains zero or multiple values and the type is a collection or map type,
   * merge the collections/maps in the list and return the merged collection/map.
   */
  @Nullable
  @SuppressWarnings("unchecked")
  private static Object flattenAttributeValues(Type<?> attrType, Iterable<Object> possibleValues) {

    // If there is only one possible value, return it.
    if (Iterables.size(possibleValues) == 1) {
      return Iterables.getOnlyElement(possibleValues);
    }

    // Otherwise, there are multiple possible values. To conform to the message shape expected by
    // query output's clients, we must transform the list of possible values. This transformation
    // will be lossy, but this is the best we can do.

    // If the attribute's type is not a collection type, return null. Query output's clients do
    // not support list values for scalar attributes.
    if (SCALAR_TYPES.contains(attrType)) {
      return null;
    }

    // If the attribute's type is a collection type, merge the list of collections into a single
    // collection. This is a sensible solution for query output's clients, which are happy to get
    // the union of possible values.
    // TODO(bazel-team): replace below with "is ListType" check (or some variant)
    if (attrType == Type.STRING_LIST
        || attrType == BuildType.LABEL_LIST
        || attrType == BuildType.NODEP_LABEL_LIST
        || attrType == BuildType.OUTPUT_LIST
        || attrType == BuildType.DISTRIBUTIONS
        || attrType == Type.INTEGER_LIST
        || attrType == BuildType.FILESET_ENTRY_LIST) {
      ImmutableList.Builder<Object> builder = ImmutableList.<Object>builder();
      for (Object possibleValue : possibleValues) {
        Collection<Object> collection = (Collection<Object>) possibleValue;
        for (Object o : collection) {
          builder.add(o);
        }
      }
      return builder.build();
    }

    // Same for maps as for collections.
    if (attrType == Type.STRING_DICT
        || attrType == Type.STRING_LIST_DICT
        || attrType == BuildType.LABEL_DICT_UNARY
        || attrType == BuildType.LABEL_KEYED_STRING_DICT) {
      Map<Object, Object> mergedDict = new HashMap<>();
      for (Object possibleValue : possibleValues) {
        Map<Object, Object> stringDict = (Map<Object, Object>) possibleValue;
        for (Entry<Object, Object> entry : stringDict.entrySet()) {
          mergedDict.put(entry.getKey(), entry.getValue());
        }
      }
      return mergedDict;
    }

    throw new AssertionError("Unknown type: " + attrType);
  }
}
