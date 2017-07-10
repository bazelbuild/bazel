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
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.analysis.TransitiveInfoProviderMap;
import com.google.devtools.build.lib.analysis.TransitiveInfoProviderMapBuilder;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.NativeClassObjectConstructor;
import com.google.devtools.build.lib.packages.SkylarkClassObject;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import javax.annotation.Nullable;

/** A Skylark declared provider that encapsulates all providers that are needed by Java rules. */
@Immutable
public final class JavaProvider extends SkylarkClassObject implements TransitiveInfoProvider {

  public static final NativeClassObjectConstructor<JavaProvider> JAVA_PROVIDER =
      new NativeClassObjectConstructor<JavaProvider>(JavaProvider.class, "java_common.provider") {};

  private static final ImmutableSet<Class<? extends TransitiveInfoProvider>> ALLOWED_PROVIDERS =
      ImmutableSet.of(
        JavaCompilationArgsProvider.class,
        JavaSourceJarsProvider.class,
        ProtoJavaApiInfoAspectProvider.class,
        JavaRuleOutputJarsProvider.class,
        JavaRunfilesProvider.class,
        JavaPluginInfoProvider.class
      );

  private final TransitiveInfoProviderMap providers;

  /** Returns the instance for the provided providerClass, or <tt>null</tt> if not present. */
  @Nullable
  public <P extends TransitiveInfoProvider> P getProvider(Class<P> providerClass) {
    return providers.getProvider(providerClass);
  }

  public TransitiveInfoProviderMap getProviders() {
    return providers;
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
    List<ProtoJavaApiInfoAspectProvider> protoJavaApiInfoAspectProviders =
        JavaProvider.fetchProvidersFromList(providers, ProtoJavaApiInfoAspectProvider.class);
    List<JavaRunfilesProvider> javaRunfilesProviders =
        JavaProvider.fetchProvidersFromList(providers, JavaRunfilesProvider.class);
    List<JavaPluginInfoProvider> javaPluginInfoProviders =
        JavaProvider.fetchProvidersFromList(providers, JavaPluginInfoProvider.class);

    Runfiles mergedRunfiles = Runfiles.EMPTY;
    for (JavaRunfilesProvider javaRunfilesProvider : javaRunfilesProviders) {
      Runfiles runfiles = javaRunfilesProvider.getRunfiles();
      mergedRunfiles = mergedRunfiles == Runfiles.EMPTY ? runfiles : mergedRunfiles.merge(runfiles);
    }

    return JavaProvider.Builder.create()
        .addProvider(
            JavaCompilationArgsProvider.class,
            JavaCompilationArgsProvider.merge(javaCompilationArgsProviders))
        .addProvider(
          JavaSourceJarsProvider.class, JavaSourceJarsProvider.merge(javaSourceJarsProviders))
        .addProvider(
            ProtoJavaApiInfoAspectProvider.class,
            ProtoJavaApiInfoAspectProvider.merge(protoJavaApiInfoAspectProviders))
        // When a rule merges multiple JavaProviders, its purpose is to pass on information, so
        // it doesn't have any output jars.
        .addProvider(JavaRuleOutputJarsProvider.class, JavaRuleOutputJarsProvider.builder().build())
        .addProvider(JavaRunfilesProvider.class, new JavaRunfilesProvider(mergedRunfiles))
        .addProvider(
            JavaPluginInfoProvider.class, JavaPluginInfoProvider.merge(javaPluginInfoProviders))
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
      Iterable<JavaProvider> javaProviders, Class<C> providersClass) {
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
    JavaProvider javaProvider =
        (JavaProvider) target.get(JavaProvider.JAVA_PROVIDER.getKey());
    if (javaProvider == null) {
      return null;
    }
    return javaProvider.getProvider(providerClass);
  }

  public static <T extends TransitiveInfoProvider> List<T> getProvidersFromListOfTargets(
      Class<T> providerClass, Iterable<? extends TransitiveInfoCollection> targets) {
    List<T> providersList = new ArrayList<>();
    for (TransitiveInfoCollection target : targets) {
      T provider = getProvider(providerClass, target);
      if (provider != null) {
        providersList.add(provider);
      }
    }
    return providersList;
  }

  /**
   * Returns a list of the given provider class with all the said providers retrieved from the
   * given {@link JavaProvider}s.
   */
  public static <T extends TransitiveInfoProvider> ImmutableList<T>
      getProvidersFromListOfJavaProviders(
          Class<T> providerClass, Iterable<JavaProvider> javaProviders) {
    ImmutableList.Builder<T> providersList = new ImmutableList.Builder<>();
    for (JavaProvider javaProvider : javaProviders) {
      T provider = javaProvider.getProvider(providerClass);
      if (provider != null) {
        providersList.add(provider);
      }
    }
    return providersList.build();
  }

  private JavaProvider(TransitiveInfoProviderMap providers) {
    super(JAVA_PROVIDER, ImmutableMap.<String, Object>of(
        "transitive_runtime_jars", SkylarkNestedSet.of(
            Artifact.class,
            providers.getProvider(JavaCompilationArgsProvider.class)
                .getRecursiveJavaCompilationArgs().getRuntimeJars()),
        "compile_jars", SkylarkNestedSet.of(
            Artifact.class,
            providers.getProvider(JavaCompilationArgsProvider.class)
                .getJavaCompilationArgs().getCompileTimeJars())
    ));
    this.providers = providers;
  }

  /**
   * A Builder for {@link JavaProvider}.
   */
  public static class Builder {
    TransitiveInfoProviderMapBuilder providerMap;

    private Builder(TransitiveInfoProviderMapBuilder providerMap) {
      this.providerMap = providerMap;
    }

    public static Builder create() {
      return new Builder(new TransitiveInfoProviderMapBuilder());
    }

    public static Builder copyOf(JavaProvider javaProvider) {
      return new Builder(
          new TransitiveInfoProviderMapBuilder().addAll(javaProvider.getProviders()));
    }

    public <P extends TransitiveInfoProvider> Builder addProvider(
        Class<P> providerClass, TransitiveInfoProvider provider) {
      Preconditions.checkArgument(ALLOWED_PROVIDERS.contains(providerClass));
      providerMap.put(providerClass, provider);
      return this;
    }

    public JavaProvider build() {
      Preconditions.checkArgument(providerMap.contains(JavaCompilationArgsProvider.class));
      return new JavaProvider(providerMap.build());
    }
  }
}
