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

package com.google.devtools.build.lib.skyframe.serialization;

import static com.google.common.collect.ImmutableList.toImmutableList;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.flogger.GoogleLogger;
import com.google.common.reflect.ClassPath;
import com.google.common.reflect.ClassPath.ClassInfo;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.RegisteredSingletonDoNotUse;
import java.io.IOException;
import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Modifier;
import java.util.Comparator;
import java.util.function.Predicate;

/**
 * Scans the classpath to find {@link ObjectCodec} and {@link CodecRegisterer} instances.
 *
 * <p>To avoid loading classes unnecessarily, the scanner filters by class name before loading.
 * {@link ObjectCodec} implementation class names should end in "Codec" while {@link
 * CodecRegisterer} implementation class names should end in "CodecRegisterer".
 *
 * <p>See {@link CodecRegisterer} for more details.
 */
final class CodecScanner {

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  /**
   * Initializes an {@link ObjectCodecRegistry} builder by scanning classes matching the given
   * package filter.
   *
   * @param packageFilter a filter applied to the package name of each class
   * @see CodecRegisterer
   */
  static ObjectCodecRegistry.Builder initializeCodecRegistry(Predicate<String> packageFilter)
      throws IOException, ReflectiveOperationException {
    logger.atInfo().log("Building ObjectCodecRegistry");
    ObjectCodecRegistry.Builder builder = ObjectCodecRegistry.newBuilder();
    for (ClassInfo classInfo : getClassInfos(packageFilter)) {
      if (classInfo.getName().endsWith("Codec")) {
        processLikelyCodec(classInfo.load(), builder);
      } else if (classInfo.getName().endsWith("CodecRegisterer")) {
        processLikelyRegisterer(classInfo.load(), builder);
      } else if (classInfo.getName().endsWith(CodecScanningConstants.REGISTERED_SINGLETON_SUFFIX)) {
        processLikelyConstant(classInfo.load(), builder);
      } else {
        builder.addClassName(classInfo.getName().intern());
      }
    }
    return builder;
  }

  private static void processLikelyCodec(Class<?> type, ObjectCodecRegistry.Builder builder)
      throws ReflectiveOperationException {
    if (ObjectCodec.class.equals(type)
        || !ObjectCodec.class.isAssignableFrom(type)
        || Modifier.isAbstract(type.getModifiers())) {
      return;
    }

    try {
      Constructor<?> constructor = type.getDeclaredConstructor();
      constructor.setAccessible(true);
      ObjectCodec<?> codec = (ObjectCodec<?>) constructor.newInstance();
      if (codec.autoRegister()) {
        builder.add(codec);
      }
    } catch (NoSuchMethodException e) {
      logger.atFine().withCause(e).log(
          "Skipping registration of %s because it had no default constructor", type);
    }
  }

  private static void processLikelyRegisterer(Class<?> type, ObjectCodecRegistry.Builder builder)
      throws NoSuchMethodException, InvocationTargetException, InstantiationException,
          IllegalAccessException {
    if (CodecRegisterer.class.equals(type) || !CodecRegisterer.class.isAssignableFrom(type)) {
      return;
    }

    Constructor<? extends CodecRegisterer> constructor =
        type.asSubclass(CodecRegisterer.class).getDeclaredConstructor();
    constructor.setAccessible(true);
    CodecRegisterer registerer = constructor.newInstance();
    for (ObjectCodec<?> codec : registerer.getCodecsToRegister()) {
      builder.add(codec);
    }
  }

  private static void processLikelyConstant(Class<?> type, ObjectCodecRegistry.Builder builder) {
    if (!RegisteredSingletonDoNotUse.class.isAssignableFrom(type)) {
      return;
    }
    Field field;
    try {
      field = type.getDeclaredField(CodecScanningConstants.REGISTERED_SINGLETON_INSTANCE_VAR_NAME);
    } catch (NoSuchFieldException e) {
      throw new IllegalStateException(
          type
              + " inherits from "
              + RegisteredSingletonDoNotUse.class
              + " but does not have a field "
              + CodecScanningConstants.REGISTERED_SINGLETON_INSTANCE_VAR_NAME,
          e);
    }
    try {
      builder.addReferenceConstant(
          Preconditions.checkNotNull(field.get(null), "%s %s", field, type));
    } catch (IllegalAccessException e) {
      throw new IllegalStateException("Could not access field " + field + " for " + type, e);
    }
  }

  /** Return the {@link ClassInfo}s matching {@code packageFilter}, sorted by name. */
  private static ImmutableList<ClassInfo> getClassInfos(Predicate<String> packageFilter)
      throws IOException {
    return ClassPath.from(ClassLoader.getSystemClassLoader()).getResources().stream()
        .filter(ClassInfo.class::isInstance)
        .map(ClassInfo.class::cast)
        .filter(c -> packageFilter.test(c.getPackageName()))
        .sorted(Comparator.comparing(ClassInfo::getName))
        .collect(toImmutableList());
  }

  private CodecScanner() {}
}
