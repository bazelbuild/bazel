// Copyright 2020 The Bazel Authors. All rights reserved.
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

import static com.google.common.base.Verify.verify;

import com.android.aapt.Resources.Reference;
import com.android.aapt.Resources.XmlAttribute;
import com.android.aapt.Resources.XmlNode;
import com.android.resources.ResourceType;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import java.util.Optional;

/** Utilities for manipulating XML (as represented by aapt2's protobuf definitions). */
public final class ProtoXmlUtils {

  private static final String SCHEMA_AUTO = "http://schemas.android.com/apk/res-auto";
  private static final String SCHEMA_PUBLIC_PREFIX = "http://schemas.android.com/apk/res/";
  private static final String SCHEMA_PRIVATE_PREFIX = "http://schemas.android.com/apk/prv/res/";

  /** Returns all resources referenced from XML, including attribute references. */
  // Implementation notes:
  // * We return Reference objects instead of Strings for consistency with the rest of protobuf-XML
  //   usage, and to avoid ambiguity with e.g. whether '@' prefixes should be included.
  // * The aapt2 source of truth is
  // https://android.googlesource.com/platform/frameworks/base.git/+/android-10.0.0_r1/tools/aapt2/link/XmlReferenceLinker.cpp#8
  public static ImmutableList<Reference> getAllResourceReferences(XmlNode root) {
    ImmutableList.Builder<Reference> refs = ImmutableList.builder();
    getAllResourceReferences(root, refs);
    return refs.build();
  }

  private static void getAllResourceReferences(
      XmlNode xmlNode, ImmutableList.Builder<Reference> refs) {
    if (!xmlNode.hasElement()) {
      return;
    }
    for (XmlAttribute attribute : xmlNode.getElement().getAttributeList()) {
      parseAttributeNameReference(attribute.getNamespaceUri(), attribute.getName())
          .ifPresent(refs::add);
      // Note: there's a field called "compiled_item" with a Reference inside, but it's only filled
      // in *after* running "aapt2 link".
      parseResourceReference(attribute.getValue()).ifPresent(refs::add);
    }
    for (XmlNode node : xmlNode.getElement().getChildList()) {
      getAllResourceReferences(node, refs);
    }
  }

  @VisibleForTesting
  static Optional<Reference> parseAttributeNameReference(String uri, String name) {
    if (uri.isEmpty()) {
      return Optional.empty();
    }

    // See
    // https://android.googlesource.com/platform/frameworks/base.git/+/android-10.0.0_r1/tools/aapt2/xml/XmlUtil.cpp#37
    boolean isPrivate = false;
    String pkg;
    if (uri.equals(SCHEMA_AUTO)) {
      pkg = "";
    } else if (uri.startsWith(SCHEMA_PUBLIC_PREFIX)) {
      pkg = uri.substring(SCHEMA_PUBLIC_PREFIX.length());
      verify(!pkg.isEmpty());
    } else if (uri.startsWith(SCHEMA_PRIVATE_PREFIX)) {
      pkg = uri.substring(SCHEMA_PRIVATE_PREFIX.length());
      isPrivate = true;
      verify(!pkg.isEmpty());
    } else {
      return Optional.empty();
    }

    return Optional.of(
        Reference.newBuilder()
            // Reference.Type omitted for consistency with how aapt2 serializes attributes set by
            // styles; it won't be used by anything anyway.
            .setPrivate(isPrivate)
            .setName((pkg.isEmpty() ? "" : pkg + ":") + "attr/" + name)
            .build());
  }

  @VisibleForTesting
  static Optional<Reference> parseResourceReference(String value) {
    if (value.startsWith("@") || value.startsWith("?")) {
      boolean isPrivate = false;
      int startIndex = 1;

      if (value.length() <= startIndex) {
        return Optional.empty();
      }
      if (value.charAt(startIndex) == '+') {
        startIndex++;
      } else if (value.charAt(startIndex) == '*') {
        startIndex++;
        isPrivate = true;
      }
      if (value.length() <= startIndex) {
        return Optional.empty();
      }

      int slashIndex = value.indexOf('/', startIndex);
      if (slashIndex == -1) {
        return Optional.empty();
      }

      int colonIndex = value.lastIndexOf(':', slashIndex);
      int typeBegin = colonIndex == -1 ? startIndex : colonIndex + 1;
      if (ResourceType.getEnum(value.substring(typeBegin, slashIndex)) == null) {
        // aapt2 will treat something like "@x/foo" as a string instead of throwing an error.
        return Optional.empty();
      }

      return Optional.of(
          Reference.newBuilder()
              .setType(value.startsWith("@") ? Reference.Type.REFERENCE : Reference.Type.ATTRIBUTE)
              .setPrivate(isPrivate)
              .setName(value.substring(startIndex))
              .build());
    }
    return Optional.empty();
  }

  private ProtoXmlUtils() {}
}
