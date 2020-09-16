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

import com.google.common.base.Preconditions;
import com.google.common.flogger.GoogleLogger;
import com.google.common.reflect.ClassPath;
import com.google.common.reflect.ClassPath.ClassInfo;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.RegisteredSingletonDoNotUse;
import java.io.IOException;
import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.lang.reflect.ParameterizedType;
import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.function.Predicate;
import java.util.stream.Stream;

/**
 * Scans the classpath to find {@link ObjectCodec} and {@link CodecRegisterer} instances.
 *
 * <p>To avoid loading classes unnecessarily, the scanner filters by class name before loading.
 * {@link ObjectCodec} implementation class names should end in "Codec" while {@link
 * CodecRegisterer} implementation class names should end in "CodecRegisterer".
 *
 * <p>See {@link CodecRegisterer} for more details.
 */
class CodecScanner {

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
    ArrayList<Class<? extends ObjectCodec<?>>> codecs = new ArrayList<>();
    ArrayList<Class<? extends CodecRegisterer<?>>> registerers = new ArrayList<>();
    ObjectCodecRegistry.Builder builder = ObjectCodecRegistry.newBuilder();
    getClassInfos(packageFilter)
        .forEach(
            classInfo -> {
              if (classInfo.getName().endsWith("Codec")) {
                processLikelyCodec(classInfo.load(), codecs);
              } else if (classInfo.getName().endsWith("CodecRegisterer")) {
                processLikelyRegisterer(classInfo.load(), registerers);
              } else if (classInfo
                  .getName()
                  .endsWith(CodecScanningConstants.REGISTERED_SINGLETON_SUFFIX)) {
                processLikelyConstant(classInfo.load(), builder);
              } else {
                builder.addClassName(classInfo.getName().intern());
              }
            });

    HashSet<Class<? extends ObjectCodec<?>>> alreadyRegistered =
        runRegisterers(builder, registerers);
    applyDefaultRegistration(builder, alreadyRegistered, codecs);
    return builder;
  }

  @SuppressWarnings("unchecked")
  private static void processLikelyCodec(
      Class<?> type, ArrayList<Class<? extends ObjectCodec<?>>> codecs) {
    if (!ObjectCodec.class.equals(type)
        && ObjectCodec.class.isAssignableFrom(type)
        && !Modifier.isAbstract(type.getModifiers())) {
      codecs.add((Class<? extends ObjectCodec<?>>) type);
    }
  }

  @SuppressWarnings("unchecked")
  private static void processLikelyRegisterer(
      Class<?> type, ArrayList<Class<? extends CodecRegisterer<?>>> registerers) {
    if (!CodecRegisterer.class.equals(type) && CodecRegisterer.class.isAssignableFrom(type)) {
      registerers.add((Class<? extends CodecRegisterer<?>>) type);
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

  @SuppressWarnings("unchecked")
  private static HashSet<Class<? extends ObjectCodec<?>>> runRegisterers(
      ObjectCodecRegistry.Builder builder,
      ArrayList<Class<? extends CodecRegisterer<?>>> registerers)
      throws ReflectiveOperationException {
    HashSet<Class<? extends ObjectCodec<?>>> registered = new HashSet<>();
    for (Class<? extends CodecRegisterer<?>> registererType : registerers) {
      Class<? extends ObjectCodec<?>> objectCodecType = getObjectCodecType(registererType);
      registered.add(objectCodecType);
      Constructor<CodecRegisterer<?>> constructor =
          (Constructor<CodecRegisterer<?>>) registererType.getDeclaredConstructor();
      constructor.setAccessible(true);
      CodecRegisterer<?> registerer = constructor.newInstance();
      for (ObjectCodec<?> codec : registerer.getCodecsToRegister()) {
        builder.add(codec);
      }
    }
    return registered;
  }

  @SuppressWarnings("rawtypes")
  private static void applyDefaultRegistration(
      ObjectCodecRegistry.Builder builder,
      HashSet<Class<? extends ObjectCodec<?>>> alreadyRegistered,
      ArrayList<Class<? extends ObjectCodec<?>>> codecs)
      throws ReflectiveOperationException {
    for (Class<? extends ObjectCodec<?>> codecType : codecs) {
      if (alreadyRegistered.contains(codecType)) {
        continue;
      }
      try {
        Constructor constructor = codecType.getDeclaredConstructor();
        constructor.setAccessible(true);
        builder.add((ObjectCodec<?>) constructor.newInstance());
      } catch (NoSuchMethodException e) {
        logger.atFine().withCause(e).log(
            "Skipping registration of %s because it had no default constructor.", codecType);
      }
    }
  }

  @SuppressWarnings("unchecked")
  private static Class<? extends ObjectCodec<?>> getObjectCodecType(
      Class<? extends CodecRegisterer<?>> registererType) {
    Type typeArg =
        ((ParameterizedType)
                registererType.getGenericInterfaces()[getCodecRegistererIndex(registererType)])
            .getActualTypeArguments()[0];
    // This occurs when the generic parameter of CodecRegisterer is not reified, for example:
    //   class MyCodecRegisterer<T> implements CodecRegisterer<T>
    Preconditions.checkArgument(
        typeArg instanceof Class,
        "Illegal CodecRegisterer definition: %s"
            + "\nCodecRegisterer generic parameter must be reified.",
        registererType);
    return (Class<? extends ObjectCodec<?>>) typeArg;
  }

  private static int getCodecRegistererIndex(Class<? extends CodecRegisterer<?>> registererType) {
    Class<?>[] interfaces = registererType.getInterfaces();
    for (int i = 0; i < interfaces.length; ++i) {
      if (CodecRegisterer.class.equals(interfaces[i])) {
        return i;
      }
    }
    // The following line is reached when there are multiple layers of inheritance involving
    // CodecRegisterer, which is prohibited.
    throw new IllegalStateException(registererType + " doesn't directly implement CodecRegisterer");
  }

  /** Return the {@link ClassInfo}s matching {@code packageFilter}, sorted by name. */
  private static Stream<ClassInfo> getClassInfos(Predicate<String> packageFilter)
      throws IOException {
    return ClassPath.from(ClassLoader.getSystemClassLoader()).getResources().stream()
        .filter(r -> r instanceof ClassInfo)
        .map(r -> (ClassInfo) r)
        .filter(c -> packageFilter.test(c.getPackageName()))
        .sorted(Comparator.comparing(ClassInfo::getName));
  }

  private CodecScanner() {}
}
