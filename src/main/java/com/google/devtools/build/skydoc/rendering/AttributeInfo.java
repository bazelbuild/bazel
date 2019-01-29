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

package com.google.devtools.build.skydoc.rendering;

/**
 * Stores information about a skylark attribute definition.
 */
public class AttributeInfo {

  private final String name;
  private final String docString;
  private final Type type;
  private final boolean mandatory;

  public AttributeInfo(String name, String docString, Type type, boolean mandatory) {
    this.name = name;
    this.docString = docString.trim();
    this.type = type;
    this.mandatory = mandatory;
  }

  @SuppressWarnings("unused") // Used by markdown template.
  public String getName() {
    return name;
  }

  @SuppressWarnings("unused") // Used by markdown template.
  public String getDocString() {
    return docString;
  }

  @SuppressWarnings("unused") // Used by markdown template.
  public Type getType() {
    return type;
  }

  /**
   * Returns a string representing whether this attribute is required or optional.
   */
  @SuppressWarnings("unused") // Used by markdown template.
  public String getMandatoryString() {
    return mandatory ? "required" : "optional";
  }

  /**
   * Attribute type. For example, an attribute described by attr.label() will be of type LABEL.
   */
  public enum Type {
    NAME("Name"),
    INT("Integer"),
    LABEL("Label"),
    STRING("String"),
    STRING_LIST("List of strings"),
    INT_LIST("List of integers"),
    LABEL_LIST("List of labels"),
    BOOLEAN("Boolean"),
    LICENSE("List of strings"),
    LABEL_STRING_DICT("Dictionary: Label -> String"),
    STRING_DICT("Dictionary: String -> String"),
    STRING_LIST_DICT("Dictionary: String -> List of strings"),
    OUTPUT("Label"),
    OUTPUT_LIST("List of labels");

    private final String description;

    Type(String description) {
      this.description = description;
    }

    /**
     * Returns a human-readable string representing this attribute type.
     */
    public String getDescription() {
      return description;
    }
  }
}
