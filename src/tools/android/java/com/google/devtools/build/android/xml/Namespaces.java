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
package com.google.devtools.build.android.xml;

import com.google.common.base.MoreObjects;
import com.google.common.collect.ImmutableSortedMap;
import com.google.devtools.build.android.DataResourceXml;
import com.google.devtools.build.android.XmlResourceValue;
import com.google.devtools.build.android.XmlResourceValues;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.logging.Logger;
import javax.xml.namespace.QName;
import javax.xml.stream.events.Attribute;
import javax.xml.stream.events.Namespace;
import javax.xml.stream.events.StartElement;

/**
 * Represents a collection of xml namespaces.
 *
 * <p>Each &lt;resources&gt; can have xmlns declarations. Since the merging process generates the
 * resources tag to combining multiple {@link DataResourceXml}s, the Namespaces must be tracked and
 * kept with each value.
 */
public class Namespaces implements Iterable<Map.Entry<String, String>> {
  private static final Logger logger = Logger.getLogger(Namespaces.class.getCanonicalName());
  private static final Namespaces EMPTY_INSTANCE =
      new Namespaces(ImmutableSortedMap.<String, String>of());

  /** Collects prefix and uri pairs from elements. */
  public static class Collector {
    private Map<String, String> prefixToUri = new HashMap<>();

    public Namespaces toNamespaces() {
      return Namespaces.from(prefixToUri);
    }

    /**
     * Collects all the prefix and uri pairs from a start element.
     *
     * <p>Since {@link Namespaces} represents top level declarations, collectFrom ignores any prefix
     * that is declared on the element. Those will be handled by the {@link XmlResourceValue}
     * individually.
     *
     * @param start The element to collect prefix and uris from.
     * @return The current namespace builder.
     */
    public Collector collectFrom(StartElement start) {
      Iterator<Attribute> attributes = XmlResourceValues.iterateAttributesFrom(start);
      Iterator<Namespace> localNamespaces = XmlResourceValues.iterateNamespacesFrom(start);
      // Collect the local prefixes to make sure a prefix isn't declared locally.
      Set<String> prefixes = new HashSet<>();
      while (localNamespaces.hasNext()) {
        prefixes.add(localNamespaces.next().getPrefix());
      }
      collectFrom(start.getName(), prefixes);
      while (attributes.hasNext()) {
        collectFrom(attributes.next().getName(), prefixes);
      }
      return this;
    }

    void collectFrom(QName name, Set<String> localPrefixes) {
      String prefix = name.getPrefix();
      if (!prefix.isEmpty() && !localPrefixes.contains(prefix)) {
        // If the prefix exists and is not a locally declared prefix, add the prefix and uri.
        prefixToUri.put(prefix, name.getNamespaceURI());
      }
    }
  }

  public static Collector collector() {
    return new Collector();
  }

  public static Namespaces from(Map<String, String> prefixToUri) {
    if (prefixToUri.isEmpty()) {
      return empty();
    }
    return new Namespaces(ImmutableSortedMap.copyOf(prefixToUri));
  }

  /**
   * Create a {@link Namespaces} containing the singular namespace used by the name, or an empty
   * one.
   */
  public static Namespaces from(QName name) {
    if (name.getPrefix().isEmpty()) {
      return empty();
    }
    return new Namespaces(ImmutableSortedMap.of(name.getPrefix(), name.getNamespaceURI()));
  }

  public static Namespaces empty() {
    return EMPTY_INSTANCE;
  }

  // Keep the prefixes in a sorted map so that when this object is iterated over, the order is
  // deterministic.
  private final ImmutableSortedMap<String, String> prefixToUri;

  private Namespaces(ImmutableSortedMap<String, String> prefixToUri) {
    this.prefixToUri = prefixToUri;
  }

  /** Combines two {@link Namespaces} into a new instance and returns it. */
  public Namespaces union(Namespaces other) {
    // No prefixes to add, return the other.
    if (prefixToUri.isEmpty()) {
      return other;
    }
    // TODO(corysmith): Issue error when prefixes are mapped to different uris.
    // Keeping behavior for backwards compatibility.
    Map<String, String> combinedNamespaces = new LinkedHashMap<>();
    combinedNamespaces.putAll(other.prefixToUri);
    for (Map.Entry<String, String> namespace : prefixToUri.entrySet()) {
      String prefix = namespace.getKey();
      String namespaceUri = namespace.getValue();
      if (combinedNamespaces.containsKey(prefix)
          && !combinedNamespaces.get(prefix).equals(namespaceUri)) {
        logger.warning(
            String.format(
                "%s has multiple namespaces: %s and %s. Using %s."
                    + " This will be an error in the future.",
                prefix, namespaceUri, combinedNamespaces.get(prefix), namespaceUri));
      }
      combinedNamespaces.put(prefix, namespaceUri);
    }
    return Namespaces.from(combinedNamespaces);
  }

  @Override
  public Iterator<Map.Entry<String, String>> iterator() {
    return prefixToUri.entrySet().iterator();
  }

  public Map<String, String> asMap() {
    return prefixToUri;
  }

  @Override
  public int hashCode() {
    return prefixToUri.hashCode();
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(getClass()).add("prefixToUri", prefixToUri).toString();
  }

  @Override
  public boolean equals(Object obj) {
    if (obj instanceof Namespaces) {
      Namespaces other = (Namespaces) obj;
      return Objects.equals(prefixToUri, other.prefixToUri);
    }
    return false;
  }
}
