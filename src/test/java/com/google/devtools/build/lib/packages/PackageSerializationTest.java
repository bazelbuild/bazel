// Copyright 2015 The Bazel Authors. All rights reserved.
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

import static com.google.common.truth.Truth.assertThat;

import com.google.common.base.Strings;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.collect.Multimap;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.PackageFactory.EnvironmentExtension;
import com.google.devtools.build.lib.packages.util.PackageSerializationTestCase;
import com.google.devtools.build.lib.query2.proto.proto2api.Build;
import com.google.devtools.build.lib.query2.proto.proto2api.Build.Attribute.Discriminator;
import com.google.devtools.build.lib.syntax.GlobCriteria;
import com.google.devtools.build.lib.syntax.GlobList;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.ExtensionRegistryLite;

import java.io.IOException;
import java.util.List;

/**
 * Unit tests for package serialization and deserialization.
 */
public class PackageSerializationTest extends PackageSerializationTestCase {
  @Override
  protected List<EnvironmentExtension> getPackageEnvironmentExtensions() {
    return ImmutableList.of();
  }

  public void testLocationsOmitted() throws Exception {
    Package pkg = scratchPackage("bacon",
        "sh_library(name='bacon',",
        "           srcs=['bacon.sh'])");
    // By default we keep full location.
    Package pkg2 = deserializer.deserialize(codedInputFromPackage(pkg));
    Rule rule2 = pkg2.getRule("bacon");

    assertEmpty(rule2.getLocation());
    assertEmpty(rule2.getAttributeLocation("name"));
    assertEmpty(rule2.getAttributeLocation("srcs"));
  }

  public void testGlobInformationKept() throws Exception {
    scratch.file("ham/head.sh");
    scratch.file("ham/tbone.sh");
    Package pkg = scratchPackage("ham",
        "sh_library(name='ham',",
        "           srcs=glob(['*.sh'], exclude=['tbo*.sh']))");
    Package pkg2 = deserializer.deserialize(codedInputFromPackage(pkg));
    Rule ham = pkg2.getRule("ham");
    GlobList<?> globs = Rule.getGlobInfo(RawAttributeMapper.of(ham)
        .get("srcs", BuildType.LABEL_LIST));
    assertThat(globs).containsExactly(Label.parseAbsolute("//ham:head.sh"));
    List<GlobCriteria> criteria = globs.getCriteria();
    assertThat(criteria).hasSize(1);
    assertThat(criteria.get(0).getIncludePatterns()).containsExactly("*.sh");
    assertThat(criteria.get(0).getExcludePatterns()).containsExactly("tbo*.sh");
    assertTrue(criteria.get(0).isGlob());
  }

  public void testCanConcatenateGlobs() throws Exception {
    scratch.file("serrano/sinew.sh");
    scratch.file("serrano/muscle.sh");
    scratch.file("serrano/bone.sh");
    Package pkg = scratchPackage("serrano",
        "sh_library(name='serrano',",
        "           srcs=glob(['s*.sh']) + ['muscle.sh'] + glob(['b*.sh']))");
    Package pkg2 = deserializer.deserialize(codedInputFromPackage(pkg));
    Rule ham = pkg2.getRule("serrano");
    GlobList<?> globs = Rule.getGlobInfo(RawAttributeMapper.of(ham)
        .get("srcs", BuildType.LABEL_LIST));
    assertThat(globs).containsExactly(
        Label.parseAbsolute("//serrano:sinew.sh"),
        Label.parseAbsolute("//serrano:muscle.sh"),
        Label.parseAbsolute("//serrano:bone.sh"));
    List<GlobCriteria> criteria = globs.getCriteria();
    assertThat(criteria).hasSize(3);
    assertThat(criteria.get(0).getIncludePatterns()).containsExactly("s*.sh");
    assertTrue(criteria.get(0).isGlob());
    assertFalse(criteria.get(1).isGlob());
    assertThat(criteria.get(2).getIncludePatterns()).containsExactly("b*.sh");
    assertTrue(criteria.get(2).isGlob());
  }

  public void testEvents() throws Exception {
    Package pkg = scratchPackage("j", "sh_library(name='s', srcs='s.sh')");
    Package pkg2 = deserializer.deserialize(codedInputFromPackage(pkg));
    assertThat(pkg2.getEvents()).hasSize(1);
    Event ev = pkg2.getEvents().get(0);
    assertThat(ev.getMessage()).contains("expected value of type 'list(label)'");
  }

  public void testPermanentErrorBitIsKept() throws Exception {
    Package pkg = scratchPackage("g", "sh_library(name='g', srcs=g.sh)");
    Package pkg2 = deserializer.deserialize(codedInputFromPackage(pkg));
    assertTrue(pkg2.containsErrors());
  }

  public void testBasicSerialization() throws Exception {
    checkSerialization(scratchPackage("bacon",
        "sh_library(name='bacon',",
        "           srcs=['bacon.sh'],",
        "           data=glob(['*.txt']))"));
  }

  public void testIntList() throws Exception {
    Build.Attribute.Builder attrPb = Build.Attribute.newBuilder();
    attrPb.setName("has_int_list").setType(Discriminator.INTEGER_LIST);
    attrPb.addIntListValue(1).addIntListValue(2).addIntListValue(3).addIntListValue(4);
    assertThat(PackageDeserializer.deserializeAttributeValue(
        com.google.devtools.build.lib.syntax.Type.INTEGER_LIST, attrPb.build()))
        .isEqualTo(ImmutableList.of(1, 2, 3, 4));
  }

  public void testExternalPackageLabel() throws Exception {
    PackageIdentifier packageId = PackageIdentifier.create("@foo", new PathFragment("p"));
    Package pkg = scratchPackage(packageId, "filegroup(name = 'rumple', srcs = ['dark_one'])");
    assertEquals(packageId, pkg.getPackageIdentifier());
    Package pkg2 = deserializer.deserialize(codedInputFromPackage(pkg));
    assertEquals(pkg.getPackageIdentifier(), pkg2.getPackageIdentifier());
  }

  public void testConfigurableAttributes() throws Exception {
    Package pkg = scratchPackage("pkg",
        "sh_library(",
        "    name = 'bread',",
        "    srcs = select({",
        "        ':one': ['rye.sh'],",
        "        ':two': ['wheat.sh'],",
        "    }))");
    Package deserialized = deserializer.deserialize(codedInputFromPackage(pkg));
    AttributeMap attrs = RawAttributeMapper.of(deserialized.getRule("bread"));
    // We expect package serialization to "flatten" configurable attributes, e.g. the deserialized
    // rule should look like "srcs = ['rye.sh', 'wheat.sh']" (without the select). Eventually
    // we'll want to preserve the original structure, but that requires syntactic changes to the
    // proto which we'll need to ensure the depserver understands.
    assertThat(attrs.get("srcs", BuildType.LABEL_LIST)).containsExactly(
        Label.parseAbsolute("//pkg:rye.sh"), Label.parseAbsolute("//pkg:wheat.sh"));
  }

  public void testConfigurableDictionaryAttribute() throws Exception {
    // Although we expect package serialization to "flatten" configurable attributes, as described
    // above, package deserialization should not crash if multiple configuration entries for a
    // map valued attribute have keys in common.
    checkSerialization(scratchPackage("pkg",
        "cc_library(",
        "    name = 'bread',",
        "    srcs = ['PROTECTED/rye.cc'],",
        "    abi_deps = select({",
        "        ':one': {'duplicated_key': [':value1']},",
        "        ':two': {'duplicated_key': [':value2']},",
        "    }))"));
  }

  public void testRuleAttributesWithNullValuesAreIncludedInSerializedRepresentation()
      throws Exception {
    Package pkg = scratchPackage("lettuce",
        "sh_library(name='lettuce',",
        "           srcs=['romaine.sh'])");

    // Manually parse stream, we're interested in wire representation.
    CodedInputStream codedIn = codedInputFromPackage(pkg);
    readPackage(codedIn);
    Multimap<Build.Target.Discriminator, Build.Target> targets = readTargets(codedIn);

    // Check that we encoded "deprecation" in the rule proto output even though it had no value.
    Build.Rule pbRule =
        Iterables.getOnlyElement(targets.get(Build.Target.Discriminator.RULE)).getRule();
    boolean foundEmptyAttribute = false;
    for (Build.Attribute attribute : pbRule.getAttributeList()) {
      if (attribute.getName().equals("deprecation")) {
        assertFalse(attribute.hasStringValue());
        foundEmptyAttribute = true;
      }
    }
    assertTrue(foundEmptyAttribute);
  }

  public void testTargetsIndividuallySerializedAndDeserialized() throws Exception {
    scratchPackage("empty",
        "sh_library(name='noop')");

    Package pkg = scratchPackage("food",
        "sh_library(name='pork',",
        "           srcs=['pig.sh'])",
        "sh_library(name='salmon',",
        "           srcs=['river.sh'])",
        "package_group(name='self',",
        "           packages=['//food'])");

    // Manually parse stream, we're interested in wire representation.
    CodedInputStream codedIn = codedInputFromPackage(pkg);
    readPackage(codedIn);

    Multimap<Build.Target.Discriminator, Build.Target> targets = readTargets(codedIn);
    assertThat(targets).hasSize(6);
    assertThat(targets.get(Build.Target.Discriminator.SOURCE_FILE)).hasSize(3);
    assertThat(targets.get(Build.Target.Discriminator.RULE)).hasSize(2);
    assertThat(targets.get(Build.Target.Discriminator.PACKAGE_GROUP)).hasSize(1);

    // Make sure we see the same thing when deserializing all the way.
    Package deserialized = deserializer.deserialize(codedInputFromPackage(pkg));
    assertThat(deserialized.getTargets(InputFile.class)).hasSize(3);
    assertThat(deserialized.getTargets(Rule.class)).hasSize(2);
    assertThat(deserialized.getTargets(PackageGroup.class)).hasSize(1);
  }

  public void testMassivePackageDeserializesFine() throws Exception {
    // Create a package definition which exports 2^16 files with name lengths 2^10 each, which
    // should result in 2^26 (64MB) of targets. With overhead this should push us comfortably
    // over the 64MB default protocol buffer deserialization limit.
    StringBuilder sb = new StringBuilder();
    sb.append("exports_files([");
    String srcName = Strings.repeat("x", 1 << 10);
    for (int i = 0; i < (1 << 16); i++) {
      sb.append("'").append(srcName).append(i).append("',");
    }
    sb.append("'last'])");

    Package pkg = scratchPackage("meat", sb.toString());

    // Check that we created our package correctly. There should be 2^16 + 1 dummy targets from
    // our exports_files, plus the BUILD file.
    assertThat(pkg.containsErrors()).isFalse();
    assertThat(pkg.getTargets()).hasSize((1 << 16) + 2);

    checkSerialization(pkg);
  }

  private static Multimap<Build.Target.Discriminator, Build.Target> readTargets(
      CodedInputStream codedIn) throws IOException {
    Multimap<Build.Target.Discriminator, Build.Target> targets = ArrayListMultimap.create();
    Build.TargetOrTerminator tot;
    while (!(tot = readNext(codedIn)).getIsTerminator()) {
      Build.Target pbTarget = tot.getTarget();
      targets.put(pbTarget.getType(), pbTarget);
    }
    return targets;
  }

  private static Build.Package readPackage(CodedInputStream codedIn) throws IOException {
    Build.Package.Builder builder = Build.Package.newBuilder();
    codedIn.readMessage(builder, ExtensionRegistryLite.getEmptyRegistry());
    return builder.build();
  }

  private static Build.TargetOrTerminator readNext(CodedInputStream codedIn) throws IOException {
    Build.TargetOrTerminator.Builder builder = Build.TargetOrTerminator.newBuilder();
    codedIn.readMessage(builder, ExtensionRegistryLite.getEmptyRegistry());
    return builder.build();
  }

  private void assertEmpty(Location location) {
    assertEquals(0, location.getStartOffset());
    assertEquals(0, location.getEndOffset());
    assertEquals(new PathFragment("/dev/null"), location.getPath());
    assertEquals(0, location.getStartLineAndColumn().getLine());
    assertEquals(0, location.getStartLineAndColumn().getColumn());
    assertEquals(0, location.getEndLineAndColumn().getLine());
    assertEquals(0, location.getEndLineAndColumn().getColumn());
  }
}
