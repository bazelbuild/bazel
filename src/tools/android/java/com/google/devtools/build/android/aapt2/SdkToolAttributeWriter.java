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

package com.google.devtools.build.android.aapt2;

import com.android.SdkConstants;
import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Multimap;
import com.google.common.collect.MultimapBuilder;
import com.google.devtools.build.android.AndroidDataWritingVisitor;
import com.google.devtools.build.android.AndroidResourceMerger.MergingException;
import com.google.devtools.build.android.FullyQualifiedName;
import com.google.devtools.build.android.proto.SerializeFormat.ToolAttributes;
import com.google.devtools.build.android.xml.Namespaces;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collection;
import java.util.Map.Entry;
import java.util.function.Function;
import javax.xml.namespace.QName;

/**
 * An AndroidDataWritingVisitor that only records {@link SdkConstants#TOOLS_URI} attributes to the
 * {@link ToolAttributes} proto format. Comma delimited values {@link SdkConstants#ATTR_KEEP} and
 * {@link SdkConstants#ATTR_DISCARD} will be expanded into separate values.
 */
class SdkToolAttributeWriter implements AndroidDataWritingVisitor {

  private static final Splitter COMMA_SPLITTER = Splitter.on(',');

  private static final Function<String, Iterable<String>> DEFAULT_ATTRIBUTE_PROCESSOR =
      ImmutableList::of;

  private static final ImmutableMap<String, Function<String, Iterable<String>>>
      ATTRIBUTE_PROCESSORS =
          ImmutableMap.of(
              SdkConstants.ATTR_KEEP, COMMA_SPLITTER::split,
              SdkConstants.ATTR_DISCARD, COMMA_SPLITTER::split);

  final Multimap<String, String> attributes = MultimapBuilder.hashKeys().hashSetValues().build();
  private final Path out;

  SdkToolAttributeWriter(Path out) {
    this.out = out;
  }

  @Override
  public void flush() throws IOException {
    ToolAttributes.Builder builder = ToolAttributes.newBuilder();
    for (Entry<String, Collection<String>> entry : attributes.asMap().entrySet()) {
      builder.putAttributes(
          entry.getKey(),
          ToolAttributes.ToolAttributeValues.newBuilder().addAllValues(entry.getValue()).build());
    }
    try (OutputStream stream = new BufferedOutputStream(Files.newOutputStream(out))) {
      builder.build().writeTo(stream);
    }
  }

  @Override
  public Path copyManifest(Path sourceManifest) throws IOException {
    throw new UnsupportedOperationException();
  }

  @Override
  public void copyAsset(Path source, String relativeDestinationPath) throws IOException {
    throw new UnsupportedOperationException();
  }

  @Override
  public void copyResource(Path source, String relativeDestinationPath) throws MergingException {
    throw new UnsupportedOperationException();
  }

  @Override
  public void defineAttribute(FullyQualifiedName fqn, String name, String value) {
    final QName qName = QName.valueOf(fqn.name());
    if (SdkConstants.TOOLS_URI.equals(qName.getNamespaceURI())) {
      attributes.putAll(
          qName.getLocalPart(),
          ATTRIBUTE_PROCESSORS
              .getOrDefault(qName.getLocalPart(), DEFAULT_ATTRIBUTE_PROCESSOR)
              .apply(value));
    }
  }

  @Override
  public void defineNamespacesFor(FullyQualifiedName fqn, Namespaces namespaces) {}

  @Override
  public ValueResourceDefinitionMetadata define(FullyQualifiedName fqn) {
    throw new UnsupportedOperationException();
  }
}
