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
package com.google.devtools.build.android.r8;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.devtools.build.android.r8.FileUtils.CLASS_EXTENSION;

import org.objectweb.asm.Type;

/** Utilities for handling Java descriptor and binary name strings */
public class DescriptorUtils {

  public static final String DESCRIPTOR_PACKAGE_SEPARATOR = "/";
  public static final String JAVA_PACKAGE_SEPARATOR = ".";

  /**
   * Basic check of if a string is a class descriptor.
   *
   * @param descriptor a possible class descriptor i.e. "Ljava/lang/Object;"
   * @return if descriptor is a class descriptor.
   */
  public static boolean isClassDescriptor(String descriptor) {
    return descriptor.length() >= 3
        && descriptor.charAt(0) == 'L'
        && descriptor.charAt(descriptor.length() - 1) == ';'
        && !descriptor.contains(JAVA_PACKAGE_SEPARATOR);
  }

  /**
   * Basic check of if a string is a binary name.
   *
   * @param name a possible class binary name i.e. "java/lang/Object"
   * @return if name could be a binary name.
   */
  public static boolean isBinaryName(String name) {
    return name.length() > 0 && !name.contains(JAVA_PACKAGE_SEPARATOR) && !isClassDescriptor(name);
  }

  /**
   * Basic check if a string is a binary name for a companion class.
   *
   * @param name a possible binary name of a companion class i.e. "java/lang/Object"
   * @return if name is a companion class binary name.
   */
  public static boolean isCompanionClassBinaryName(String name) {
    checkArgument(isBinaryName(name), "'%s' is not a binary name", name);
    return name.endsWith(R8Utils.INTERFACE_COMPANION_SUFFIX);
  }

  /**
   * Convert class descriptor to its binary name.
   *
   * @param descriptor a class descriptor i.e. "Ljava/lang/Object;"
   * @return class binary name i.e. "java/lang/Object"
   */
  public static String descriptorToBinaryName(String descriptor) {
    checkArgument(isClassDescriptor(descriptor), "'%s' is not a class descriptor", descriptor);
    return Type.getType(descriptor).getInternalName();
  }

  /**
   * Convert class to its binary name.
   *
   * @param clazz a class
   * @return class binary name i.e. "java/lang/Object"
   */
  public static String classToBinaryName(Class<?> clazz) {
    return Type.getInternalName(clazz);
  }

  /**
   * Convert class descriptor to its class file file name.
   *
   * @param descriptor a class descriptor i.e. "Ljava/lang/Object;"
   * @return class file file name i.e. "java/lang/Object.class"
   */
  public static String descriptorToClassFileName(String descriptor) {
    return descriptorToBinaryName(descriptor) + CLASS_EXTENSION;
  }

  private DescriptorUtils() {}
}
