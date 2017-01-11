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

import java.io.IOException;
import java.lang.reflect.Field;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;

/**
 * Loads android.R.attr resource IDs from an android jar. An alternative may be to parse the
 * res/values/public.xml from the SDK, but the class loading approach is ~20ms, right now.
 */
public class AndroidFrameworkAttrIdJar implements AndroidFrameworkAttrIdProvider {

  private static final String ANDROID_ATTR_CLASS = "android.R$attr";
  private final Path androidJar;
  private Map<String, Integer> cachedFields;

  public AndroidFrameworkAttrIdJar(Path androidJar) {
    this.androidJar = androidJar;
  }

  @Override
  public int getAttrId(String fieldName) throws AttrLookupException {
    // Lazily load the ANDROID_ATTR_CLASS from the androidJar, to save time if never end up
    // needing the android framework attributes. This provider can only work for one given
    // androidJar path, since we never invalidate the lazily filled cache.
    if (cachedFields == null) {
      cachedFields = getAttrFields();
    }
    Integer result = cachedFields.get(fieldName);
    if (result == null) {
      throw new AttrLookupException("Android attribute not found: " + fieldName);
    }
    return result;
  }

  private Map<String, Integer> getAttrFields() throws AttrLookupException {
    try (URLClassLoader urlClassLoader =
        new URLClassLoader(new URL[] {androidJar.toUri().toURL()})) {
      Class<?> attrClass = urlClassLoader.loadClass(ANDROID_ATTR_CLASS);
      Map<String, Integer> attributeIds = new HashMap<>();
      for (Field field : attrClass.getFields()) {
        attributeIds.put(field.getName(), field.getInt(null));
      }
      return attributeIds;
    } catch (IOException | ClassNotFoundException | IllegalAccessException e) {
      throw new AttrLookupException(e);
    }
  }
}
