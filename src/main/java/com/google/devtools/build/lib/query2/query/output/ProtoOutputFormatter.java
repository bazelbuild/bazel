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
package com.google.devtools.build.lib.query2.query.output;

import static com.google.devtools.build.lib.query2.proto.proto2api.Build.Target.Discriminator.ENVIRONMENT_GROUP;
import static com.google.devtools.build.lib.query2.proto.proto2api.Build.Target.Discriminator.GENERATED_FILE;
import static com.google.devtools.build.lib.query2.proto.proto2api.Build.Target.Discriminator.PACKAGE_GROUP;
import static com.google.devtools.build.lib.query2.proto.proto2api.Build.Target.Discriminator.RULE;
import static com.google.devtools.build.lib.query2.proto.proto2api.Build.Target.Discriminator.SOURCE_FILE;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableMultimap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.graph.Digraph;
import com.google.devtools.build.lib.graph.Node;
import com.google.devtools.build.lib.packages.AggregatingAttributeMapper;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.AttributeFormatter;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.DependencyFilter;
import com.google.devtools.build.lib.packages.EnvironmentGroup;
import com.google.devtools.build.lib.packages.InputFile;
import com.google.devtools.build.lib.packages.OutputFile;
import com.google.devtools.build.lib.packages.PackageGroup;
import com.google.devtools.build.lib.packages.ProtoUtils;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.query2.common.CommonQueryOptions;
import com.google.devtools.build.lib.query2.compat.FakeLoadTarget;
import com.google.devtools.build.lib.query2.engine.OutputFormatterCallback;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment;
import com.google.devtools.build.lib.query2.engine.SynchronizedDelegatingOutputFormatterCallback;
import com.google.devtools.build.lib.query2.engine.ThreadSafeOutputFormatterCallback;
import com.google.devtools.build.lib.query2.proto.proto2api.Build;
import com.google.devtools.build.lib.query2.proto.proto2api.Build.Attribute.Discriminator;
import com.google.devtools.build.lib.query2.proto.proto2api.Build.GeneratedFile;
import com.google.devtools.build.lib.query2.proto.proto2api.Build.QueryResult;
import com.google.devtools.build.lib.query2.proto.proto2api.Build.SourceFile;
import com.google.devtools.build.lib.query2.query.aspectresolvers.AspectResolver;
import com.google.devtools.build.lib.query2.query.output.QueryOptions.OrderOutput;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
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

  private static final Comparator<Build.Attribute> ATTRIBUTE_NAME =
      Comparator.comparing(Build.Attribute::getName);

  @SuppressWarnings("unchecked")
  private static final ImmutableSet<Type<?>> SCALAR_TYPES =
      ImmutableSet.<Type<?>>of(
          Type.INTEGER, Type.STRING, BuildType.LABEL, BuildType.NODEP_LABEL, BuildType.OUTPUT,
          Type.BOOLEAN, BuildType.TRISTATE, BuildType.LICENSE);

  private AspectResolver aspectResolver;
  private DependencyFilter dependencyFilter;
  private boolean relativeLocations;
  private boolean includeDefaultValues = true;
  private Predicate<String> ruleAttributePredicate = Predicates.alwaysTrue();
  private boolean flattenSelects = true;
  private boolean includeLocations = true;
  private boolean includeRuleInputsAndOutputs = true;
  private boolean includeSyntheticAttributeHash = false;

  @Nullable private EventHandler eventHandler;

  @Override
  public String getName() {
    return "proto";
  }

  @Override
  public void setOptions(CommonQueryOptions options, AspectResolver aspectResolver) {
    super.setOptions(options, aspectResolver);
    this.aspectResolver = aspectResolver;
    this.dependencyFilter = FormatUtils.getDependencyFilter(options);
    this.relativeLocations = options.relativeLocations;
    this.includeDefaultValues = options.protoIncludeDefaultValues;
    this.ruleAttributePredicate = newAttributePredicate(options.protoOutputRuleAttributes);
    this.flattenSelects = options.protoFlattenSelects;
    this.includeLocations = options.protoIncludeLocations;
    this.includeRuleInputsAndOutputs = options.protoIncludeRuleInputsAndOutputs;
    this.includeSyntheticAttributeHash = options.protoIncludeSyntheticAttributeHash;
  }

  @Override
  public void setEventHandler(@Nullable EventHandler eventHandler) {
    this.eventHandler = eventHandler;
  }

  private static Predicate<String> newAttributePredicate(List<String> outputAttributes) {
    if (outputAttributes.equals(ImmutableList.of("all"))) {
      return Predicates.alwaysTrue();
    } else if (outputAttributes.isEmpty()) {
      return Predicates.alwaysFalse();
    } else {
      return Predicates.in(ImmutableSet.copyOf(outputAttributes));
    }
  }

  @Override
  public OutputFormatterCallback<Target> createPostFactoStreamCallback(
      OutputStream out, QueryOptions options) {
    return new StreamedQueryResultFormatter(out);
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
        result.getTopologicalOrder(new FormatUtils.TargetOrdering()), Node::getLabel);
  }

  @Override
  protected Iterable<Target> getOrderedTargets(Digraph<Target> result, QueryOptions options) {
    return options.orderOutput == OrderOutput.FULL ? getSortedLabels(result) : result.getLabels();
  }

  /** Converts a logical {@link Target} object into a {@link Build.Target} protobuffer. */
  public Build.Target toTargetProtoBuffer(Target target) throws InterruptedException {
    return toTargetProtoBuffer(target, /*extraDataForAttrHash=*/ "");
  }

  /** Converts a logical {@link Target} object into a {@link Build.Target} protobuffer. */
  @VisibleForTesting
  public Build.Target toTargetProtoBuffer(Target target, Object extraDataForAttrHash)
      throws InterruptedException {
    Build.Target.Builder targetPb = Build.Target.newBuilder();

    if (target instanceof Rule) {
      Rule rule = (Rule) target;
      Build.Rule.Builder rulePb = Build.Rule.newBuilder()
          .setName(rule.getLabel().toString())
          .setRuleClass(rule.getRuleClass());
      if (includeLocations) {
        rulePb.setLocation(FormatUtils.getLocation(target, relativeLocations));
      }
      addAttributes(rulePb, rule, extraDataForAttrHash);
      String transitiveHashCode = rule.getRuleClassObject().getRuleDefinitionEnvironmentHashCode();
      if (transitiveHashCode != null && includeRuleDefinitionEnvironment()) {
        // The RuleDefinitionEnvironment is always defined for Skylark rules and
        // always null for non Skylark rules.
        rulePb.addAttribute(
            Build.Attribute.newBuilder()
                .setName(RULE_IMPLEMENTATION_HASH_ATTR_NAME)
                .setType(ProtoUtils.getDiscriminatorFromType(Type.STRING))
                .setStringValue(transitiveHashCode));
      }

      ImmutableMultimap<Attribute, Label> aspectsDependencies =
          aspectResolver.computeAspectDependencies(target, dependencyFilter);
      if (!aspectsDependencies.isEmpty()) {
        // Add information about additional attributes from aspects.
        List<Build.Attribute> attributes = new ArrayList<>(aspectsDependencies.asMap().size());
        for (Map.Entry<Attribute, Collection<Label>> entry :
            aspectsDependencies.asMap().entrySet()) {
          Attribute attribute = entry.getKey();
          Collection<Label> labels = entry.getValue();
          if (!includeAspectAttribute(attribute, labels)) {
            continue;
          }
          Object attributeValue = getAspectAttributeValue(target, attribute, labels);
          Build.Attribute serializedAttribute =
              AttributeFormatter.getAttributeProto(
                  attribute,
                  attributeValue,
                  /*explicitlySpecified=*/ false,
                  /*encodeBooleanAndTriStateAsIntegerAndString=*/ true);
          attributes.add(serializedAttribute);
        }
        rulePb.addAllAttribute(
            attributes.stream().distinct().sorted(ATTRIBUTE_NAME).collect(Collectors.toList()));
      }
      if (includeRuleInputsAndOutputs) {
        // Add all deps from aspects as rule inputs of current target.
        if (!aspectsDependencies.isEmpty()) {
          aspectsDependencies.values().stream()
              .distinct()
              .forEach(dep -> rulePb.addRuleInput(dep.toString()));
        }
        // Include explicit elements for all direct inputs and outputs of a rule; this goes beyond
        // what is available from the attributes above, since it may also (depending on options)
        // include implicit outputs, host-configuration outputs, and default values.
        rule.getLabels(dependencyFilter).stream()
            .distinct()
            .forEach(input -> rulePb.addRuleInput(input.toString()));
        rule.getOutputFiles()
            .stream()
            .distinct()
            .forEach(output -> rulePb.addRuleOutput(output.getLabel().toString()));
      }
      for (String feature : rule.getPackage().getFeatures()) {
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

      if (includeLocations) {
        output.setLocation(FormatUtils.getLocation(target, relativeLocations));
      }
      targetPb.setType(GENERATED_FILE);
      targetPb.setGeneratedFile(output.build());
    } else if (target instanceof InputFile) {
      InputFile inputFile = (InputFile) target;
      Label label = inputFile.getLabel();

      Build.SourceFile.Builder input = Build.SourceFile.newBuilder()
          .setName(label.toString());

      if (includeLocations) {
        input.setLocation(FormatUtils.getLocation(target, relativeLocations));
      }

      if (inputFile.getName().equals("BUILD")) {
        Iterable<Label> skylarkLoadLabels = aspectResolver == null
            ? inputFile.getPackage().getSkylarkFileDependencies()
            : aspectResolver.computeBuildFileDependencies(inputFile.getPackage());

        for (Label skylarkLoadLabel : skylarkLoadLabels) {
          input.addSubinclude(skylarkLoadLabel.toString());
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

      if (includeLocations) {
        input.setLocation(FormatUtils.getLocation(target, relativeLocations));
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

  protected void addAttributes(Build.Rule.Builder rulePb, Rule rule, Object extraDataForAttrHash)
      throws InterruptedException {
    Map<Attribute, Build.Attribute> serializedAttributes = Maps.newHashMap();
    AggregatingAttributeMapper attributeMapper = AggregatingAttributeMapper.of(rule);
    for (Attribute attr : rule.getAttributes()) {
      if (!shouldIncludeAttribute(rule, attr)) {
        continue;
      }
      Object attributeValue;
      if (flattenSelects || !attributeMapper.isConfigurable(attr.getName())) {
        attributeValue =
            flattenAttributeValues(
                attr.getType(), PossibleAttributeValues.forRuleAndAttribute(rule, attr));
      } else {
        attributeValue = attributeMapper.getSelectorList(attr.getName(), attr.getType());
      }
      Build.Attribute serializedAttribute =
          AttributeFormatter.getAttributeProto(
              attr,
              attributeValue,
              rule.isAttributeValueExplicitlySpecified(attr),
              /*encodeBooleanAndTriStateAsIntegerAndString=*/ true);
      serializedAttributes.put(attr, serializedAttribute);
    }
    rulePb.addAllAttribute(
        serializedAttributes
            .values()
            .stream()
            .distinct()
            .sorted(ATTRIBUTE_NAME)
            .collect(Collectors.toList()));

    if (includeSyntheticAttributeHash()) {
      rulePb.addAttribute(
          Build.Attribute.newBuilder()
              .setName("$internal_attr_hash")
              .setStringValue(
                  SyntheticAttributeHashCalculator.compute(
                      rule, serializedAttributes, extraDataForAttrHash))
              .setType(Discriminator.STRING));
    }
  }

  protected boolean shouldIncludeAttribute(Rule rule, Attribute attr) {
    return (includeDefaultValues || rule.isAttributeValueExplicitlySpecified(attr))
        && ruleAttributePredicate.apply(attr.getName());
  }

  private Object getAspectAttributeValue(
      Target target, Attribute attribute, Collection<Label> labels) {
    Type<?> attributeType = attribute.getType();
    if (attributeType.equals(BuildType.LABEL)) {
      Preconditions.checkState(labels.size() == 1, "attribute=%s, labels=%s", attribute, labels);
      return Iterables.getOnlyElement(labels);
    } else if (attributeType.equals(BuildType.LABEL_KEYED_STRING_DICT)) {
      // Ideally we'd support LABEL_KEYED_STRING_DICT by getting the value directly from the aspect
      // definition vs. trying to reverse-construct it from the flattened labels as this method
      // does. Unfortunately any proper support surfaces a latent bug between --output=proto and
      // aspect attributes: "{@code labels} isn't the set of labels for a single attribute value but
      // for all values of all attributes with the same name. We can have multiple attributes with
      // the same name because multiple aspects may attach to a rule, and nothing is stopping them
      // from defining the same attribute names. That means the "Attribute" proto message doesn't
      // really represent a single attribute, in spite of its documented purpose. This all calls for
      // an API design upgrade to properly consider these relationships. Details at b/149982967.
      if (eventHandler != null) {
        eventHandler.handle(
            Event.error(
                String.format(
                    "Target \"%s\", aspect attribute \"%s\": type \"%s\" not yet supported with"
                        + " --output=proto.",
                    target.getLabel(), attribute.getName(), BuildType.LABEL_KEYED_STRING_DICT)));
      }
      // This return value is misleading when the above error isn't get triggered: it implies an
      // empty result with no signal that that result isn't accurate.
      // TODO(bazel-team): either make the result accurate or trigger an error universally. Letting
      // OutputFormatter.output() throw a QueryException is a promising approach.
      return ImmutableMap.of();
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

  /** Allow filtering of aspect attributes. */
  protected boolean includeAspectAttribute(Attribute attr, Collection<Label> value) {
    return true;
  }

  protected boolean includeRuleDefinitionEnvironment() {
    return true;
  }

  /** Returns if the "$internal_attr_hash" should be computed and added to the result. */
  protected boolean includeSyntheticAttributeHash() {
    return includeSyntheticAttributeHash;
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
        for (Map.Entry<Object, Object> entry : stringDict.entrySet()) {
          mergedDict.put(entry.getKey(), entry.getValue());
        }
      }
      return mergedDict;
    }

    throw new AssertionError("Unknown type: " + attrType);
  }

  /**
   * Specialized {@link OutputFormatterCallback} implementation which produces a valid {@link
   * QueryResult} in streaming fashion. Internally this class makes some reasonably sound and stable
   * assumptions about the format of serialized protos in order to improve memory overhead and
   * performance.
   */
  private class StreamedQueryResultFormatter extends OutputFormatterCallback<Target> {

    /**
     * Pseudo-arbitrarily chosen buffer size for output. Chosen to be large enough to fit a handful
     * of targets without needing to flush to the underlying output, which may not be buffered.
     */
    private static final int OUTPUT_BUFFER_SIZE = 16384;

    private final CodedOutputStream codedOut;

    private StreamedQueryResultFormatter(OutputStream out) {
      this.codedOut = CodedOutputStream.newInstance(out, OUTPUT_BUFFER_SIZE);
    }

    @Override
    public void processOutput(Iterable<Target> partialResult)
        throws IOException, InterruptedException {
      // Write out targets with their tag (field number) as if they were serialized as part of a
      // QueryResult proto. The assumptions we make about this being compatible with actually
      // constructing and serializing a QueryResult proto are protected by test coverage and proto
      // best practices.
      for (Target target : partialResult) {
        codedOut.writeMessage(QueryResult.TARGET_FIELD_NUMBER, toTargetProtoBuffer(target));
      }
    }

    @Override
    public void close(boolean failFast) throws IOException {
      codedOut.flush();
    }
  }
}
