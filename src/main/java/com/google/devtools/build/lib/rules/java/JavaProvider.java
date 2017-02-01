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
package com.google.devtools.build.lib.rules.java;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.SkylarkProviders;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.analysis.TransitiveInfoProviderMap;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.SkylarkClassObject;
import com.google.devtools.build.lib.packages.SkylarkClassObjectConstructor;
import java.util.Arrays;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;
import javax.annotation.Nullable;

/** A Skylark declared provider that encapsulates all providers that are needed by Java rules. */
@Immutable
public final class JavaProvider extends SkylarkClassObject implements TransitiveInfoProvider {

  public static final SkylarkClassObjectConstructor JAVA_PROVIDER =
      SkylarkClassObjectConstructor.createNative("java_common.provider");

  private static final Set<Class<? extends TransitiveInfoProvider>> allowedProviders =
      new HashSet<>(Arrays.asList(
        JavaCompilationArgsProvider.class,
        JavaSourceJarsProvider.class)
      );

  private final TransitiveInfoProviderMap providers;

  /** Returns the instance for the provided providerClass, or <tt>null</tt> if not present. */
  @Nullable
  public <P extends TransitiveInfoProvider> P getProvider(Class<P> providerClass) {
    return providers.getProvider(providerClass);
  }

  /**
   * Merges the given providers into one {@link JavaProvider}. All the providers with the same type
   * in the given list are merged into one provider that is added to the resulting
   * {@link JavaProvider}.
   */
  public static JavaProvider merge(List<JavaProvider> providers) {
    List<JavaCompilationArgsProvider> javaCompilationArgsProviders =
        JavaProvider.fetchProvidersFromList(providers, JavaCompilationArgsProvider.class);
    List<JavaSourceJarsProvider> javaSourceJarsProviders =
        JavaProvider.fetchProvidersFromList(providers, JavaSourceJarsProvider.class);

    return JavaProvider.Builder.create()
        .addProvider(
            JavaCompilationArgsProvider.class,
            JavaCompilationArgsProvider.merge(javaCompilationArgsProviders))
        .addProvider(
          JavaSourceJarsProvider.class, JavaSourceJarsProvider.merge(javaSourceJarsProviders))
        .build();
  }

  /**
   * Returns a list of providers of the specified class, fetched from the given list of
   * {@link JavaProvider}s.
   * Returns an empty list if no providers can be fetched.
   * Returns a list of the same size as the given list if the requested providers are of type
   * JavaCompilationArgsProvider.
   */
  public static <C extends TransitiveInfoProvider> List<C> fetchProvidersFromList(
      List<JavaProvider> javaProviders, Class<C> providersClass) {
    List<C> fetchedProviders = new LinkedList<>();
    for (JavaProvider javaProvider : javaProviders) {
      C provider = javaProvider.getProvider(providersClass);
      if (provider != null) {
        fetchedProviders.add(provider);
      }
    }
    return fetchedProviders;
  }

  /**
   * Returns a provider of the specified class, fetched from the specified target or, if not found,
   * from the JavaProvider of the given target. JavaProvider can be found as a declared provider
   * in SkylarkProviders.
   * Returns null if no such provider exists.
   *
   * <p>A target can either have both the specified provider and JavaProvider that encapsulates the
   * same information, or just one of them.</p>
   */
  @Nullable
  public static <T extends TransitiveInfoProvider> T getProvider(
      Class<T> providerClass, TransitiveInfoCollection target) {
    T provider = target.getProvider(providerClass);
    if (provider != null) {
      return provider;
    }
    SkylarkProviders skylarkProviders = target.getProvider(SkylarkProviders.class);
    if (skylarkProviders == null) {
      return null;
    }
    JavaProvider javaProvider =
        (JavaProvider) skylarkProviders.getDeclaredProvider(JavaProvider.JAVA_PROVIDER.getKey());
    if (javaProvider == null) {
      return null;
    }
    return javaProvider.getProvider(providerClass);
  }

  private JavaProvider(TransitiveInfoProviderMap providers) {
    super(JAVA_PROVIDER, ImmutableMap.<String, Object>of());
    this.providers = providers;
  }

  /**
   * A Builder for {@link JavaProvider}.
   */
  public static class Builder {
    TransitiveInfoProviderMap.Builder providerMap = new TransitiveInfoProviderMap.Builder();

    private Builder() {}

    public static Builder create() {
      return new Builder();
    }

    public <P extends TransitiveInfoProvider> Builder addProvider(
        Class<P> providerClass, TransitiveInfoProvider provider) {
      Preconditions.checkArgument(allowedProviders.contains(providerClass));
      providerMap.put(providerClass, provider);
      return this;
    }

    public JavaProvider build() {
      Preconditions.checkArgument(providerMap.contains(JavaCompilationArgsProvider.class));
      return new JavaProvider(providerMap.build());
    }
  }
}
