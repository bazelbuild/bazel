// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.android.desugar.nest;

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.devtools.build.android.desugar.langmodel.LangModelConstants.NEST_COMPANION_CLASS_SIMPLE_NAME;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Streams;
import com.google.devtools.build.android.desugar.io.FileContentProvider;
import com.google.devtools.build.android.desugar.langmodel.ClassAttributeRecord;
import com.google.devtools.build.android.desugar.langmodel.ClassMemberRecord;
import java.io.ByteArrayInputStream;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import javax.annotation.Nullable;
import org.objectweb.asm.ClassWriter;

/** Manages the creation and IO stream for nest-companion classes. */
public class NestCompanions {

  private final ClassMemberRecord classMemberRecord;
  private final ClassAttributeRecord classAttributeRecord;
  private final Map<String, String> nestCompanionToHostMap;

  /**
   * A map from the class binary names of nest hosts to the associated class writer of the nest's
   * companion.
   */
  private ImmutableMap<String, ClassWriter> companionWriters;

  public static NestCompanions create(
      ClassMemberRecord classMemberRecord, ClassAttributeRecord classAttributeRecord) {
    return new NestCompanions(classMemberRecord, classAttributeRecord, new HashMap<>());
  }

  private NestCompanions(
      ClassMemberRecord classMemberRecord,
      ClassAttributeRecord classAttributeRecord,
      HashMap<String, String> nestCompanionToHostMap) {
    this.classMemberRecord = classMemberRecord;
    this.classAttributeRecord = classAttributeRecord;
    this.nestCompanionToHostMap = nestCompanionToHostMap;
  }

  /**
   * Generates the nest companion class writers. The nest companion classes will be generated as the
   * last placeholder class type for the synthetic constructor, whose originating constructor has
   * any invocation in other classes in nest.
   */
  void prepareCompanionClasses() {
    ImmutableList<String> nestHostsWithCompanion =
        classMemberRecord.findAllConstructorMemberKeys().stream()
            .map(
                constructor ->
                    nestHost(constructor.owner(), classAttributeRecord, nestCompanionToHostMap))
            .flatMap(Streams::stream)
            .distinct()
            .collect(toImmutableList());
    ImmutableMap.Builder<String, ClassWriter> companionWriterBuilders = ImmutableMap.builder();
    for (String nestHost : nestHostsWithCompanion) {
      String nestCompanion = nestHost + '$' + NEST_COMPANION_CLASS_SIMPLE_NAME;
      nestCompanionToHostMap.put(nestCompanion, nestHost);
      companionWriterBuilders.put(nestHost, new ClassWriter(ClassWriter.COMPUTE_MAXS));
    }
    companionWriters = companionWriterBuilders.build();
  }

  /**
   * The public API that finds the nest host for a given class. It is expected {@link
   * #prepareCompanionClasses()} executed before this API is ready. The method returns {@link
   * Optional#empty()} if the class is not part of a nest. A generated nest companion class and its
   * nest host are considered to be a nest host/member relationship.
   */
  public Optional<String> nestHost(String classBinaryName) {
    // Ensures prepareCompanionClasses has been executed.
    checkNotNull(companionWriters);
    return nestHost(classBinaryName, classAttributeRecord, nestCompanionToHostMap);
  }

  /**
   * The internal method finds the nest host for a given class from a class attribute record. The
   * method returns {@link * Optional#empty()} if the class is not part of a nest. A generated nest
   * companion class and its * nest host are considered to be a nest host/member relationship.
   *
   * <p>In addition to exam the NestHost_attribute from the class file, this method returns the
   * class under investigation itself for a class with NestMembers_attribute but without
   * NestHost_attribute.
   */
  private static Optional<String> nestHost(
      String classBinaryName,
      ClassAttributeRecord classAttributeRecord,
      Map<String, String> companionToHostMap) {
    if (companionToHostMap.containsKey(classBinaryName)) {
      return Optional.of(companionToHostMap.get(classBinaryName));
    }
    Optional<String> nestHost = classAttributeRecord.getNestHost(classBinaryName);
    if (nestHost.isPresent()) {
      return nestHost;
    }
    Set<String> nestMembers = classAttributeRecord.getNestMembers(classBinaryName);
    if (!nestMembers.isEmpty()) {
      return Optional.of(classBinaryName);
    }
    return Optional.empty();
  }

  /**
   * Returns the internal name of the nest companion class for a given class.
   *
   * <p>e.g. The nest host of a/b/C$D is a/b/C$NestCC
   */
  public String nestCompanion(String classBinaryName) {
    return nestHost(classBinaryName)
        .map(nestHost -> nestHost + '$' + NEST_COMPANION_CLASS_SIMPLE_NAME)
        .orElseThrow(
            () ->
                new IllegalStateException(
                    String.format(
                        "Expected the presence of NestHost attribute of %s to get nest companion.",
                        classBinaryName)));
  }

  /**
   * Gets the class visitor of the affiliated nest host of the given class. E.g For the given class
   * com/google/a/b/Delta$Echo, it returns the class visitor of com/google/a/b/Delta$NestCC
   */
  @Nullable
  public ClassWriter getCompanionClassWriter(String classInternalName) {
    return nestHost(classInternalName).map(nestHost -> companionWriters.get(nestHost)).orElse(null);
  }

  /** Gets all nest companion classes required to be generated. */
  public ImmutableList<String> getAllCompanionClasses() {
    return companionWriters.keySet().stream().map(this::nestCompanion).collect(toImmutableList());
  }

  /** Gets all nest companion files required to be generated. */
  public ImmutableList<FileContentProvider<ByteArrayInputStream>> getCompanionFileProviders() {
    ImmutableList.Builder<FileContentProvider<ByteArrayInputStream>> fileContents =
        ImmutableList.builder();
    for (String companion : getAllCompanionClasses()) {
      fileContents.add(
          new FileContentProvider<>(
              companion + ".class", () -> getByteArrayInputStreamOfCompanionClass(companion)));
    }
    return fileContents.build();
  }

  private ByteArrayInputStream getByteArrayInputStreamOfCompanionClass(String companion) {
    ClassWriter companionClassWriter = getCompanionClassWriter(companion);
    checkNotNull(
        companionClassWriter,
        "Expected companion class (%s) to be present in (%s)",
        companionWriters);
    return new ByteArrayInputStream(companionClassWriter.toByteArray());
  }
}
