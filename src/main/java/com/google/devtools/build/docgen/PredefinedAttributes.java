// Copyright 2014 Google Inc. All rights reserved.
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
package com.google.devtools.build.docgen;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableMap.Builder;
import com.google.common.io.ByteStreams;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.Map;

/**
 * A class to contain the base definition of common BUILD rule attributes.
 * 
 * <p>It looks for the {@code {common,binary,test}} directory of the attributes template directory.
 * Each file in that directory should be named after the attribute name and contains the HTML
 * description of the attribute.
 */
public class PredefinedAttributes {

  private static ImmutableMap<String, RuleDocumentationAttribute> generateAttributeMap(
      String commonType, String... names) {
    Builder<String, RuleDocumentationAttribute> builder =
        ImmutableMap.<String, RuleDocumentationAttribute>builder();
    for (String name : names) {
      String filename = "templates/attributes/" + commonType + "/" + name + ".html";
      try {
        InputStream stream = PredefinedAttributes.class.getResourceAsStream(filename);
        String content = new String(ByteStreams.toByteArray(stream), StandardCharsets.UTF_8);
        builder.put(name, RuleDocumentationAttribute.create(name, commonType, content));
      } catch (IOException e) {
        System.err.println("Exception while reading " + filename + ", skipping!");
        e.printStackTrace();
      }
    }
    return builder.build();
  }

  public static final Map<String, RuleDocumentationAttribute> COMMON_ATTRIBUTES =
      generateAttributeMap(DocgenConsts.COMMON_ATTRIBUTES, "deps", "data", "licenses",
          "distribs", "deprecation", "obsolete", "testonly", "tags", "visibility");

  public static final Map<String, RuleDocumentationAttribute> BINARY_ATTRIBUTES =
      generateAttributeMap(DocgenConsts.BINARY_ATTRIBUTES, "args", "output_licenses");

  public static final Map<String, RuleDocumentationAttribute> TEST_ATTRIBUTES =
      generateAttributeMap(DocgenConsts.TEST_ATTRIBUTES, "args", "size", "timeout", "flaky",
          "shard_count", "local");
}
