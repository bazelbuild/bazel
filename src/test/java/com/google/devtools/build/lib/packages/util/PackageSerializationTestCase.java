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

package com.google.devtools.build.lib.packages.util;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.packages.CachingPackageLocator;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.PackageDeserializer;
import com.google.devtools.build.lib.packages.PackageDeserializer.AttributesToDeserialize;
import com.google.devtools.build.lib.packages.PackageDeserializer.PackageDeserializationEnvironment;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.packages.PackageFactory.EnvironmentExtension;
import com.google.devtools.build.lib.packages.PackageSerializer;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClassProvider;
import com.google.devtools.build.lib.query2.proto.proto2api.Build;
import com.google.devtools.build.lib.testutil.FoundationTestCase;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import com.google.devtools.build.lib.vfs.Path;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/** Provides test infrastructure for package serialization tests. */
public abstract class PackageSerializationTestCase extends FoundationTestCase {

  private CachingPackageLocator packageLocator;
  private Map<PackageIdentifier, Path> buildFileMap;
  private PackageFactory packageFactory;
  private RuleClassProvider ruleClassProvider;

  protected PackageSerializer serializer;
  protected PackageDeserializer deserializer;

  protected abstract List<EnvironmentExtension> getPackageEnvironmentExtensions();

  @Override
  protected void setUp() throws Exception {
    super.setUp();

    reporter = new Reporter();
    buildFileMap = new HashMap<>();

    packageLocator = new CachingPackageLocator() {
      @Override
      public Path getBuildFileForPackage(PackageIdentifier packageName) {
        return buildFileMap.get(packageName);
      }
    };

    ruleClassProvider = TestRuleClassProvider.getRuleClassProvider();
    packageFactory = new PackageFactory(ruleClassProvider, getPackageEnvironmentExtensions());
    serializer = new PackageSerializer();
    deserializer = new PackageDeserializer(createDeserializationEnvironment());

  }

  protected PackageDeserializationEnvironment createDeserializationEnvironment() {
    return new TestPackageDeserializationEnvironment();
  }

  private void registerBuildFile(PackageIdentifier packageName, Path path) {
    buildFileMap.put(packageName, path);
  }

  protected Package scratchPackage(String name, String... lines) throws Exception {
    return scratchPackage(PackageIdentifier.createInDefaultRepo(name), lines);
  }

  protected Package scratchPackage(PackageIdentifier packageId, String... lines)
      throws Exception {
    Path buildFile = scratch.file("" + packageId.getPathFragment() + "/BUILD", lines);
    registerBuildFile(packageId, buildFile);
    Package.Builder externalPkg =
        Package.newExternalPackageBuilder(buildFile.getRelative("WORKSPACE"), "TESTING");
    externalPkg.setWorkspaceName(TestConstants.WORKSPACE_NAME);
    return packageFactory.createPackageForTesting(
        packageId, externalPkg.build(), buildFile, packageLocator, reporter);
  }

  protected Package checkSerialization(Package pkg) throws Exception {
    ByteArrayOutputStream bytesOut = new ByteArrayOutputStream();
    CodedOutputStream codedOut = CodedOutputStream.newInstance(bytesOut);
    serializer.serialize(pkg, codedOut);
    codedOut.flush();

    Package deserializedPkg = deserializer.deserialize(
        CodedInputStream.newInstance(bytesOut.toByteArray()));

    // Check equality of an arbitrary sample of properties.
    assertEquals(pkg.getName(), deserializedPkg.getName());
    assertEquals(pkg.getPackageIdentifier(), deserializedPkg.getPackageIdentifier());
    assertEquals(pkg.getBuildFileLabel(), deserializedPkg.getBuildFileLabel());
    assertEquals(pkg.getFilename(), deserializedPkg.getFilename());
    assertEquals(pkg.toString(), deserializedPkg.toString());
    // Not all implementations of Target implement equals, so just check sizes match up.
    assertThat(deserializedPkg.getTargets()).hasSize(pkg.getTargets().size());
    return deserializedPkg;
  }

  private class TestPackageDeserializationEnvironment implements PackageDeserializationEnvironment {

    @Override
    public Path getPath(String buildFilePath) {
      return scratch.getFileSystem().getPath(buildFilePath);
    }

    @Override
    public RuleClass getRuleClass(Build.Rule rulePb, Location ruleLocation) {
      return ruleClassProvider.getRuleClassMap().get(rulePb.getRuleClass());
    }

    @Override
    public AttributesToDeserialize attributesToDeserialize() {
      return PackageDeserializer.DESERIALIZE_ALL_ATTRS;
    }
  }

  protected CodedInputStream codedInputFromPackage(Package pkg) throws IOException {
    ByteArrayOutputStream bytesOut = new ByteArrayOutputStream();
    CodedOutputStream codedOut = CodedOutputStream.newInstance(bytesOut);
    serializer.serialize(pkg, codedOut);
    codedOut.flush();
    return CodedInputStream.newInstance(bytesOut.toByteArray());
  }
}
