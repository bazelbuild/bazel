// Copyright 2023 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.bzlmod;

import com.google.auto.value.AutoValue;
import com.google.common.base.Splitter;
import com.ryanharter.auto.value.gson.GenerateTypeAdapter;
import java.util.ArrayList;
import java.util.List;

/**
 * This object holds the evaluation factors for module extensions in the lockfile, such as the
 * operating system and architecture it depends on. If an extension has no dependencies in this
 * regard, the object remains empty
 */
@AutoValue
@GenerateTypeAdapter
public abstract class ModuleExtensionEvalFactors implements Comparable<ModuleExtensionEvalFactors> {

  private static final String OS_KEY = "os:";
  private static final String ARCH_KEY = "arch:";

  // This is used when the module extension doesn't depend on os or arch, to indicate that
  // its value is "general" and can be used with any platform
  private static final String GENERAL_EXTENSION = "general";

  /** Returns the OS this extension is evaluated on, or empty if it doesn't depend on the os */
  public abstract String getOs();

  /**
   * Returns the architecture this extension is evaluated on, or empty if it doesn't depend on the
   * architecture
   */
  public abstract String getArch();

  public boolean isEmpty() {
    return getOs().isEmpty() && getArch().isEmpty();
  }

  public boolean hasSameDependenciesAs(ModuleExtensionEvalFactors other) {
    return getOs().isEmpty() == other.getOs().isEmpty()
        && getArch().isEmpty() == other.getArch().isEmpty();
  }

  @Override
  public String toString() {
    if (isEmpty()) {
      return GENERAL_EXTENSION;
    }

    List<String> parts = new ArrayList<>();
    if (!getOs().isEmpty()) {
      parts.add(OS_KEY + getOs());
    }
    if (!getArch().isEmpty()) {
      parts.add(ARCH_KEY + getArch());
    }
    return String.join(",", parts);
  }

  @Override
  public int compareTo(ModuleExtensionEvalFactors o) {
    return toString().compareTo(o.toString());
  }

  public static ModuleExtensionEvalFactors create(String os, String arch) {
    return new AutoValue_ModuleExtensionEvalFactors(os, arch);
  }

  public static ModuleExtensionEvalFactors parse(String s) {
    if (s.equals(GENERAL_EXTENSION)) {
      return ModuleExtensionEvalFactors.create("", "");
    }

    String os = "";
    String arch = "";
    var extParts = Splitter.on(',').splitToList(s);
    for (String part : extParts) {
      if (part.startsWith(OS_KEY)) {
        os = part.substring(OS_KEY.length());
      } else if (part.startsWith(ARCH_KEY)) {
        arch = part.substring(ARCH_KEY.length());
      }
    }
    return ModuleExtensionEvalFactors.create(os, arch);
  }
}
