// Copyright 2019 The Bazel Authors. All rights reserved.
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

import static com.google.common.base.Verify.verify;

import com.android.resources.ResourceType;
import com.google.auto.value.AutoValue;
import com.google.common.base.Strings;
import java.util.regex.Matcher;

/**
 * Represents an Android resource for the purposes of declarations/linkage. Resource definitions use
 * a ResourceName with a set of qualifiers; this is currently modeled using {@FullyQualifiedName}.
 */
@AutoValue
abstract class ResourceName {
  /** Package name; this is usually left empty to denote the default/current package. */
  abstract String pkg();

  /**
   * Type of resource, e.g. string, drawable, et cetera. For example, {@code @string/foo} references
   * a resource of type {@code string}.
   */
  abstract ResourceType type();

  /** Entry name of a resource. For example, {@code @string/foo} references entry {@code foo}. */
  abstract String entry();

  static ResourceName create(String pkg, ResourceType type, String entry) {
    return new AutoValue_ResourceName(pkg, type, entry);
  }

  /**
   * Constructs a ResourceName from the canonical form used throughout Android ({@code
   * [<pkg>:]<type>/<entry>}).
   */
  // While we use aapt2 to convert XML data to protobuf, we still need to do some parsing:
  //
  // https://android.googlesource.com/platform/frameworks/base.git/+/refs/tags/platform-tools-29.0.2/tools/aapt2/Resources.proto#232
  static ResourceName parse(String name) {
    Matcher matcher = FullyQualifiedName.QUALIFIED_REFERENCE.matcher(name);
    verify(
        matcher.matches(),
        "%s is not a valid resource name. Expected %s",
        name,
        FullyQualifiedName.QUALIFIED_REFERENCE);

    return create(
        Strings.nullToEmpty(matcher.group("package")),
        ResourceType.getEnum(matcher.group("type")),
        matcher.group("name"));
  }
}
