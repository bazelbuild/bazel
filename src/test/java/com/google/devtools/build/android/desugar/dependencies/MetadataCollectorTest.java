// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android.desugar.dependencies;

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.android.desugar.proto.DesugarDeps;
import com.google.devtools.build.android.desugar.proto.DesugarDeps.DesugarDepsInfo;
import com.google.devtools.build.android.desugar.proto.DesugarDeps.InterfaceDetails;
import java.util.Arrays;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link MetadataCollector}. */
@RunWith(JUnit4.class)
public class MetadataCollectorTest {

  @Test
  public void testEmptyAvoidsOutput() {
    assertThat(new MetadataCollector(false).toByteArray()).isNull();
  }

  @Test
  public void testAssumeCompanionClass() throws Exception {
    MetadataCollector collector = new MetadataCollector(false);
    collector.assumeCompanionClass("a", "b$$CC");
    collector.assumeCompanionClass("b", "b$$CC");
    collector.assumeCompanionClass("a", "a$$CC");

    DesugarDepsInfo info = extractProto(collector);

    assertThat(info.getAssumePresentList())
        .containsExactly(
            dependency("a", "a$$CC"), dependency("a", "b$$CC"), dependency("b", "b$$CC"));
  }

  @Test
  public void testMissingImplementedInterface() throws Exception {
    MetadataCollector collector = new MetadataCollector(true);
    collector.missingImplementedInterface("a", "b");
    collector.missingImplementedInterface("c", "b");
    collector.missingImplementedInterface("a", "c");

    DesugarDepsInfo info = extractProto(collector);
    assertThat(info.getMissingInterfaceList().get(0)).isEqualTo(dependency("a", "b"));
    assertThat(info.getMissingInterfaceList().get(1)).isEqualTo(dependency("a", "c"));
    assertThat(info.getMissingInterfaceList().get(2)).isEqualTo(dependency("c", "b"));
  }

  @Test
  public void testRecordExtendedInterfaces() throws Exception {
    MetadataCollector collector = new MetadataCollector(false);
    collector.recordExtendedInterfaces("c", "d");
    collector.recordExtendedInterfaces("a", "c", "b");
    collector.recordExtendedInterfaces("b");

    DesugarDepsInfo info = extractProto(collector);

    assertThat(info.getInterfaceWithSupertypesList().get(0))
        .isEqualTo(interfaceDetails("a", "b", "c"));
    assertThat(info.getInterfaceWithSupertypesList().get(0).getExtendedInterfaceList().get(0))
        .isEqualTo(wrapType("b"));
    assertThat(info.getInterfaceWithSupertypesList().get(0).getExtendedInterfaceList().get(1))
        .isEqualTo(wrapType("c"));

    assertThat(info.getInterfaceWithSupertypesList().get(1)).isEqualTo(interfaceDetails("c", "d"));
  }

  @Test
  public void testRecordDefaultMethods() throws Exception {
    MetadataCollector collector = new MetadataCollector(false);
    collector.recordDefaultMethods("b", 1);
    collector.recordDefaultMethods("a", 0);

    DesugarDepsInfo info = extractProto(collector);
    assertThat(info.getInterfaceWithCompanionList().get(0))
        .isEqualTo(interfaceWithCompanion("a", 0));
    assertThat(info.getInterfaceWithCompanionList().get(1))
        .isEqualTo(interfaceWithCompanion("b", 1));
  }

  private DesugarDeps.InterfaceWithCompanion interfaceWithCompanion(String origin, int count) {
    return DesugarDeps.InterfaceWithCompanion.newBuilder()
        .setOrigin(wrapType(origin))
        .setNumDefaultMethods(count)
        .build();
  }

  private DesugarDeps.InterfaceDetails interfaceDetails(String originName, String... interfaces) {
    return InterfaceDetails.newBuilder()
        .setOrigin(wrapType(originName))
        .addAllExtendedInterface(
            Arrays.stream(interfaces)
                .map(MetadataCollectorTest::wrapType)
                .collect(toImmutableList()))
        .build();
  }

  private DesugarDeps.Dependency dependency(String origin, String target) {
    return DesugarDeps.Dependency.newBuilder()
        .setOrigin(wrapType(origin))
        .setTarget(wrapType(target))
        .build();
  }

  private static DesugarDeps.Type wrapType(String name) {
    return DesugarDeps.Type.newBuilder().setBinaryName(name).build();
  }

  private DesugarDepsInfo extractProto(MetadataCollector collector) throws Exception {
    return DesugarDepsInfo.parseFrom(collector.toByteArray());
  }
}
