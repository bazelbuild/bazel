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
package com.google.devtools.build.lib.packages;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.License.DistributionType;
import com.google.devtools.build.lib.packages.MakeEnvironment.Binding;
import com.google.devtools.build.lib.query2.proto.proto2api.Build;
import com.google.devtools.build.lib.query2.proto.proto2api.Build.Rule.Builder;
import com.google.protobuf.CodedOutputStream;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;

/**
 * Functionality to serialize loaded packages.
 */
public class PackageSerializer {
  /** Allows custom serialization logic to be injected. */
  public interface PackageSerializationEnvironment {
    /**
     * Called right before the given builder's {@link Build.Rule.Builder#build} method is called.
     * Implementations can use this hook to serialize additional data in the proto.
     */
    void maybeSerializeAdditionalDataForRule(Rule rule, Build.Rule.Builder builder);
  }

  // Workaround for Java serialization making it tough to pass in a serialization environment
  // manually.
  // volatile is needed to ensure that the objects are published safely.
  public static volatile PackageSerializationEnvironment defaultPackageSerializationEnvironment =
      new PackageSerializationEnvironment() {
        @Override
        public void maybeSerializeAdditionalDataForRule(Rule rule, Builder builder) {
        }
      };

  private final PackageSerializationEnvironment env;

  public PackageSerializer() {
    this(defaultPackageSerializationEnvironment);
  }

  public PackageSerializer(PackageSerializationEnvironment env) {
    this.env = Preconditions.checkNotNull(env);
  }

  /**
   * Serialize a package to {@code out}. The inverse of {@link PackageDeserializer#deserialize}.
   *
   * <p>Writes pkg as a single
   * {@link com.google.devtools.build.lib.query2.proto.proto2api.Build.Package} protocol buffer
   * message followed by a series of
   * {@link com.google.devtools.build.lib.query2.proto.proto2api.Build.TargetOrTerminator} messages
   * encoding the targets.
   *
   * @param pkg the {@link Package} to be serialized
   * @param codedOut the stream to pkg's serialized representation to
   * @throws IOException on failure writing to {@code out}
   */
  public void serialize(Package pkg, CodedOutputStream codedOut) throws IOException {
    Build.Package.Builder builder = Build.Package.newBuilder();
    builder.setName(pkg.getName());
    builder.setRepository(pkg.getPackageIdentifier().getRepository().getName());
    builder.setBuildFilePath(pkg.getFilename().getPathString());
    // The extra bit is needed to handle the corner case when the default visibility is [], i.e.
    // zero labels.
    builder.setDefaultVisibilitySet(pkg.isDefaultVisibilitySet());
    if (pkg.isDefaultVisibilitySet()) {
      for (Label visibilityLabel : pkg.getDefaultVisibility().getDeclaredLabels()) {
        builder.addDefaultVisibilityLabel(visibilityLabel.toString());
      }
    }

    builder.setDefaultTestonly(pkg.getDefaultTestOnly());
    if (pkg.getDefaultDeprecation() != null) {
      builder.setDefaultDeprecation(pkg.getDefaultDeprecation());
    }

    for (String defaultCopt : pkg.getDefaultCopts()) {
      builder.addDefaultCopt(defaultCopt);
    }

    if (pkg.isDefaultHdrsCheckSet()) {
      builder.setDefaultHdrsCheck(pkg.getDefaultHdrsCheck());
    }

    builder.setDefaultLicense(serializeLicense(pkg.getDefaultLicense()));

    for (DistributionType distributionType : pkg.getDefaultDistribs()) {
      builder.addDefaultDistrib(distributionType.toString());
    }

    for (String feature : pkg.getFeatures()) {
      builder.addDefaultSetting(feature);
    }

    for (Label subincludeLabel : pkg.getSubincludeLabels()) {
      builder.addSubincludeLabel(subincludeLabel.toString());
    }

    for (Label skylarkLabel : pkg.getSkylarkFileDependencies()) {
      builder.addSkylarkLabel(skylarkLabel.toString());
    }

    for (Build.MakeVar makeVar :
         serializeMakeEnvironment(pkg.getMakeEnvironment())) {
      builder.addMakeVariable(makeVar);
    }

    for (Event event : pkg.getEvents()) {
      builder.addEvent(serializeEvent(event));
    }

    builder.setContainsErrors(pkg.containsErrors());

    builder.setWorkspaceName(pkg.getWorkspaceName());

    codedOut.writeMessageNoTag(builder.build());

    // Targets are emitted separately as individual protocol buffers as to prevent overwhelming
    // protocol buffer deserialization size limits.
    emitTargets(pkg.getTargets(), codedOut);
  }

  private Build.Target serializeInputFile(InputFile inputFile) {
    Build.SourceFile.Builder builder = Build.SourceFile.newBuilder();
    builder.setName(inputFile.getLabel().getName());
    if (inputFile.isVisibilitySpecified()) {
      for (Label visibilityLabel : inputFile.getVisibility().getDeclaredLabels()) {
        builder.addVisibilityLabel(visibilityLabel.toString());
      }
    }
    if (inputFile.isLicenseSpecified()) {
      builder.setLicense(serializeLicense(inputFile.getLicense()));
    }

    return Build.Target.newBuilder()
        .setType(Build.Target.Discriminator.SOURCE_FILE)
        .setSourceFile(builder.build())
        .build();
  }

  private Build.Target serializePackageGroup(PackageGroup packageGroup) {
    Build.PackageGroup.Builder builder = Build.PackageGroup.newBuilder();

    builder.setName(packageGroup.getLabel().getName());

    for (PackageSpecification packageSpecification : packageGroup.getPackageSpecifications()) {
      builder.addContainedPackage(packageSpecification.toString());
    }

    for (Label include : packageGroup.getIncludes()) {
      builder.addIncludedPackageGroup(include.toString());
    }

    return Build.Target.newBuilder()
        .setType(Build.Target.Discriminator.PACKAGE_GROUP)
        .setPackageGroup(builder.build())
        .build();
  }

  private Build.Target serializeRule(Rule rule) {
    Build.Rule.Builder builder = Build.Rule.newBuilder();
    builder.setName(rule.getLabel().getName());
    builder.setRuleClass(rule.getRuleClass());
    builder.setPublicByDefault(rule.getRuleClassObject().isPublicByDefault());
    for (Attribute attribute : rule.getAttributes()) {
      builder.addAttribute(
          AttributeSerializer.getAttributeProto(
              attribute,
              AttributeSerializer.getAttributeValues(rule, attribute),
              rule.isAttributeValueExplicitlySpecified(attribute),
              /*includeGlobs=*/ true));
    }
    env.maybeSerializeAdditionalDataForRule(rule, builder);

    return Build.Target.newBuilder()
        .setType(Build.Target.Discriminator.RULE)
        .setRule(builder.build())
        .build();
  }

  private static List<Build.MakeVar> serializeMakeEnvironment(MakeEnvironment makeEnv) {
    List<Build.MakeVar> result = new ArrayList<>();

    for (Map.Entry<String, ImmutableList<Binding>> var : makeEnv.getBindings().entrySet()) {
      Build.MakeVar.Builder varPb = Build.MakeVar.newBuilder();
      varPb.setName(var.getKey());
      for (Binding binding : var.getValue()) {
        Build.MakeVarBinding.Builder bindingPb = Build.MakeVarBinding.newBuilder();
        bindingPb.setValue(binding.getValue());
        bindingPb.setPlatformSetRegexp(binding.getPlatformSetRegexp());
        varPb.addBinding(bindingPb);
      }

      result.add(varPb.build());
    }

    return result;
  }

  private static Build.License serializeLicense(License license) {
    Build.License.Builder result = Build.License.newBuilder();

    for (License.LicenseType licenseType : license.getLicenseTypes()) {
      result.addLicenseType(licenseType.toString());
    }

    for (Label exception : license.getExceptions()) {
      result.addException(exception.toString());
    }
    return result.build();
  }

  private Build.Event serializeEvent(Event event) {
    Build.Event.Builder result = Build.Event.newBuilder();
    result.setMessage(event.getMessage());

    Build.Event.EventKind kind;
    switch (event.getKind()) {
      case ERROR:
        kind = Build.Event.EventKind.ERROR;
        break;
      case WARNING:
        kind = Build.Event.EventKind.WARNING;
        break;
      case INFO:
        kind = Build.Event.EventKind.INFO;
        break;
      case PROGRESS:
        kind = Build.Event.EventKind.PROGRESS;
        break;
      default: throw new IllegalArgumentException("unexpected event type: " + event.getKind());
    }

    result.setKind(kind);
    return result.build();
  }

  /** Writes targets as a series of separate TargetOrTerminator messages to out. */
  private void emitTargets(Collection<Target> targets, CodedOutputStream codedOut)
      throws IOException {
    for (Target target : targets) {
      if (target instanceof InputFile) {
        emitTarget(serializeInputFile((InputFile) target), codedOut);
      } else if (target instanceof OutputFile) {
        // Output files are not serialized; they are recreated by the RuleClass on deserialization.
      } else if (target instanceof PackageGroup) {
        emitTarget(serializePackageGroup((PackageGroup) target), codedOut);
      } else if (target instanceof Rule) {
        emitTarget(serializeRule((Rule) target), codedOut);
      }
    }

    // Terminate stream with isTerminator = true.
    codedOut.writeMessageNoTag(Build.TargetOrTerminator.newBuilder()
        .setIsTerminator(true)
        .build());
  }

  private static void emitTarget(Build.Target target, CodedOutputStream codedOut)
      throws IOException {
    codedOut.writeMessageNoTag(Build.TargetOrTerminator.newBuilder()
        .setTarget(target)
        .build());
  }

}
