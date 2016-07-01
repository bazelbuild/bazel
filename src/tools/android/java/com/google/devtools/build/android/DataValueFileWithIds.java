// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.android.ParsedAndroidData.KeyValueConsumer;
import com.google.devtools.build.android.xml.IdXmlResourceValue;

import com.android.SdkConstants;
import com.android.resources.ResourceType;

import java.io.BufferedInputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Iterator;

import javax.xml.stream.XMLEventReader;
import javax.xml.stream.XMLInputFactory;
import javax.xml.stream.XMLStreamException;
import javax.xml.stream.events.Attribute;
import javax.xml.stream.events.StartElement;
import javax.xml.stream.events.XMLEvent;

/**
 * Parses an XML file for "@+id/foo" and creates {@link IdXmlResourceValue} from parsed IDs.
 * This can be a layout file, menu, drawable, etc.
 */
public class DataValueFileWithIds {

  public static void parse(
      XMLInputFactory xmlInputFactory,
      Path source,
      FullyQualifiedName fileKey,
      FullyQualifiedName.Factory fqnFactory,
      KeyValueConsumer<DataKey, DataResource> overwritingConsumer,
      KeyValueConsumer<DataKey, DataResource> combiningConsumer)
      throws IOException, XMLStreamException {
    ImmutableSet.Builder<String> newIds = ImmutableSet.builder();
    try (BufferedInputStream inStream = new BufferedInputStream(Files.newInputStream(source))) {
      XMLEventReader eventReader =
          xmlInputFactory.createXMLEventReader(inStream, StandardCharsets.UTF_8.toString());
      // Go through every start tag and look at the values of all attributes (the attribute does
      // not need to be android:id, e.g., android:layout_above="@+id/UpcomingView").
      // If the any attribute's value begins with @+id/, then track it, since aapt seems to be
      // forgiving and allow even non-android namespaced attributes to define a new ID.
      while (eventReader.hasNext()) {
        XMLEvent event = eventReader.nextEvent();
        if (event.isStartElement()) {
          StartElement start = event.asStartElement();
          Iterator<Attribute> attributes = XmlResourceValues.iterateAttributesFrom(start);
          while (attributes.hasNext()) {
            Attribute attribute = attributes.next();
            String value = attribute.getValue();
            if (value.startsWith(SdkConstants.NEW_ID_PREFIX)) {
              String idName = value.substring(SdkConstants.NEW_ID_PREFIX.length());
              newIds.add(idName);
            }
          }
        }
      }
      eventReader.close();
    }
    ImmutableSet<String> idResources = newIds.build();
    overwritingConsumer.consume(fileKey, DataValueFile.of(source));
    for (String id : idResources) {
      combiningConsumer.consume(
          fqnFactory.create(ResourceType.ID, id),
          DataResourceXml.of(source, IdXmlResourceValue.of()));
    }
  }

}
