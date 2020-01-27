// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules.java;

import com.google.devtools.build.lib.util.Fingerprint;
import java.util.Map;
import java.util.Properties;

/** The generic implementation of {@link BuildInfoPropertiesTranslator} */
public class GenericBuildInfoPropertiesTranslator implements BuildInfoPropertiesTranslator {

  private static final String GUID = "e71fe4a8-11af-4ec0-9b38-1d3e7f542f51";

  // syntax is %ID% for a property that depends on the ID key, %ID|default% to
  // always add the property with the "default" key, %% is to add a percent sign
  private final Map<String, String> translationKeys;

  /**
   * Create a generic translator, for each key,value pair in {@code translationKeys}, the key
   * represents the property key that will be written and the value, its value. Inside value every
   * %ID% is replaced by the corresponding build information with the same ID key. The property
   * won't be added if it's depends on an unresolved build information. Adding a property can be
   * forced even if a build information is missing by specifying a default value using the
   * %ID|default% syntax. Finally to add a percent sign, just use the %% syntax.
   */
  public GenericBuildInfoPropertiesTranslator(Map<String, String> translationKeys) {
    this.translationKeys = translationKeys;
  }

  @Override
  public String computeKey() {
    Fingerprint f = new Fingerprint();
    f.addString(GUID);
    f.addStringMap(translationKeys);
    return f.hexDigestAndReset();
  }

  @Override
  public void translate(Map<String, String> buildInfo, Properties properties) {
    for (Map.Entry<String, String> entry : translationKeys.entrySet()) {
      String translatedValue = translateValue(entry.getValue(), buildInfo);
      if (translatedValue != null) {
        properties.put(entry.getKey(), translatedValue);
      }
    }
  }

  private String translateValue(String valueDescription, Map<String, String> buildInfo) {
    String[] split = valueDescription.split("%");
    StringBuilder result = new StringBuilder();
    boolean isInsideKey = false;
    for (String key : split) {
      if (isInsideKey) {
        if (key.isEmpty()) {
          result.append("%"); // empty key means %%
        } else {
          String defaultValue = null;
          int i = key.lastIndexOf('|');
          if (i >= 0) {
            defaultValue = key.substring(i + 1);
            key = key.substring(0, i);
          }
          if (buildInfo.containsKey(key)) {
            result.append(buildInfo.get(key));
          } else if (defaultValue != null) {
            result.append(defaultValue);
          } else { // we haven't found the requested key so we ignore the whole value
            return null;
          }
        }
      } else {
        result.append(key);
      }
      isInsideKey = !isInsideKey;
    }
    return result.toString();
  }
}
