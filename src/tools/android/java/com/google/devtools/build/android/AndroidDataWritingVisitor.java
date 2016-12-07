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

import com.android.ide.common.res2.MergingException;
import com.google.devtools.build.android.xml.AttrXmlResourceValue;
import com.google.devtools.build.android.xml.Namespaces;
import com.google.devtools.build.android.xml.StyleableXmlResourceValue;
import java.io.Flushable;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Map.Entry;
import javax.annotation.CheckReturnValue;
import javax.xml.namespace.QName;

/** An interface for visiting android data for writing. */
public interface AndroidDataWritingVisitor extends Flushable {
  /**
   * Copies the AndroidManifest to the destination directory.
   */
  Path copyManifest(Path sourceManifest) throws IOException;

  /**
   * Copies the source asset to the relative destination path.
   *
   * @param source The source file to copy.
   * @param relativeDestinationPath The relative destination path to write the asset to.
   * @throws IOException if there are errors during copying.
   */
  void copyAsset(Path source, String relativeDestinationPath) throws IOException;

  /**
   * Copies the source resource to the relative destination path.
   *
   * @param source The source file to copy.
   * @param relativeDestinationPath The relative destination path to write the resource to.
   * @throws IOException if there are errors during copying.
   * @throws MergingException for errors during png crunching.
   */
  void copyResource(Path source, String relativeDestinationPath)
      throws IOException, MergingException;

  /**
   * Adds the namespaces associated with a {@link FullyQualifiedName}.
   *
   * <p>An xml namespace consists of a prefix and a uri. They are common declared on the root
   * &lt;resources&gt; tag of each resource. The namespaces collected here will be merged together,
   * with the last uri added taking precedence over the prefix key.
   */
  void defineNamespacesFor(FullyQualifiedName fqn, Namespaces namespaces);

  /**
   * Provides a fluent interface to generate an xml resource for the values directory.
   *
   * <p>Example usage: 
   * <code>
   *    writer.define(key)
   *        .derivedFrom(source)
   *        .startTag(tagName)
   *        .named(key)
   *        .closeTag()
   *        .write(stringValue)
   *        .endTag()
   *        .save();
   * </code>
   */
  // Check return value will ensure that the value is finished being written.
  @CheckReturnValue
  ValueResourceDefinitionMetadata define(FullyQualifiedName fqn);

  /** Represents the xml values resource meta data. */
  @CheckReturnValue
  interface ValueResourceDefinitionMetadata {
    ValuesResourceDefinition derivedFrom(DataSource source);
  }

  /** Fluent interface to define the xml value for a {@link FullyQualifiedName}. */
  @CheckReturnValue
  interface ValuesResourceDefinition {
    /** Starts an xml tag with a prefix and localName. */
    StartTag startTag(String prefix, String localName);

    /** Starts an xml tag with a localName. */
    StartTag startTag(String localName);

    /** Starts an xml tag with a QName. */
    StartTag startTag(QName name);

    /** Starts an xml tag with the name "item" */
    StartTag startItemTag();

    /**
     * Takes another values xml resource and writes it as a child tag here.
     *
     * <p>This allows xml elements from other {@link XmlResourceValue} to be moved in the stream.
     * Currently, this is only necessary for {@link StyleableXmlResourceValue} which can have 
     * {@link AttrXmlResourceValue} defined as child elements (yet, they are merged and treated as
     * independent resources.)
     *
     * @param fqn The {@link FullyQualifiedName} of the {@link XmlResourceValue} to be adopted. This
     *     resource doesn't have to be defined for the adopt invocation, but it must exist when
     *     {@link AndroidDataWritingVisitor#flush()} is called.
     * @return The current definition.
     */
    ValuesResourceDefinition adopt(FullyQualifiedName fqn);

    /** Adds a string as xml characters to the definition. */
    ValuesResourceDefinition addCharactersOf(String characters);

    /** Ends the last {@link StartTag}. */
    ValuesResourceDefinition endTag();

    /** Saves and validates the xml resource definition. */
    void save();
  }

  /** Represents the start of opening tag of a resource xml. */
  @CheckReturnValue
  interface StartTag {
    /** Adds name="{@link FullyQualifiedName}#name()" attribute. */
    StartTag named(FullyQualifiedName key);
    /** Adds "name" attribute to the {@link StartTag}. */
    StartTag named(String key);
    /** Adds all the {@link Entry} as key="value" to the {@link StartTag}. */
    StartTag addAttributesFrom(Iterable<Entry<String, String>> entries);
    /** Starts an attribute of prefix:name. */
    Attribute attribute(String prefix, String name);
    /** Starts an attribute of name. */
    Attribute attribute(String string);
    /** Indicates the next attribute will only be written if the value is not null. */
    Optional optional();
    /** Closes the {@link StartTag} as ">" */
    ValuesResourceDefinition closeTag();
    /** Closes the {@link StartTag} as "/>", indicating it is a unary xml element. */
    ValuesResourceDefinition closeUnaryTag();
  }

  /** Adjective for an optional attribute. */
  @CheckReturnValue
  interface Optional {
    /** Starts an attribute of prefix:name. */
    Attribute attribute(String prefix, String name);
    /** Starts an attribute of name. */
    Attribute attribute(String string);
  }

  /** Represents an xml attribute of a start tag. */
  @CheckReturnValue
  interface Attribute {
    /** Sets the attribute value. */
    StartTag setTo(String value);
    /** Sets the attribute value to {@linkplain FullyQualifiedName#name()}. */
    StartTag setTo(FullyQualifiedName fqn);
    /** Start the process of setting an attribute value from an iterable of strings. * */
    ValueJoiner setFrom(Iterable<String> values);
  }

  /** Represents the concatenation step of turning an {@link Iterable} into a string. */
  @CheckReturnValue
  interface ValueJoiner {
    StartTag joinedBy(String separator);
  }
}
