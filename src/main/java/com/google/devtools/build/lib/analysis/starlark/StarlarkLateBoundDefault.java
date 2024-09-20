// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis.starlark;

import com.github.benmanes.caffeine.cache.Caffeine;
import com.github.benmanes.caffeine.cache.LoadingCache;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.starlark.annotations.StarlarkConfigurationField;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.Attribute.AbstractLabelLateBoundDefault;
import com.google.devtools.build.lib.packages.Attribute.LateBoundDefault;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.starlarkbuildapi.LateBoundDefaultApi;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.Map;
import javax.annotation.Nullable;
import javax.annotation.concurrent.Immutable;
import net.starlark.java.annot.StarlarkAnnotations;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.eval.Printer;

/**
 * An implementation of {@link LateBoundDefault} which obtains a late-bound attribute value (of type
 * 'label') specifically by Starlark configuration fragment name and field name, as registered by
 * {@link StarlarkConfigurationField}.
 *
 * <p>For example, a StarlarkLateBoundDefault on "java" and "toolchain" would require a valid
 * configuration fragment named "java" with a method annotated with {@link
 * StarlarkConfigurationField} of name "toolchain". This {@link LateBoundDefault} would provide a
 * late-bound dependency (defined by the label returned by that configuration field) in the current
 * target configuration.
 */
@Immutable
public class StarlarkLateBoundDefault<FragmentT> extends AbstractLabelLateBoundDefault<FragmentT>
    implements LateBoundDefaultApi {

  private final Method method;
  private final String fragmentName;
  private final String fragmentFieldName;

  @Override
  public Label resolve(
      Rule rule,
      AttributeMap attributes,
      FragmentT config,
      Object analysisContext,
      EventHandler eventHandler) {
    try {
      Object result = method.invoke(config);
      return (Label) result;
    } catch (IllegalAccessException | InvocationTargetException e) {
      // Configuration field methods should not throw either of these exceptions.
      throw new AssertionError("Method invocation failed: " + e);
    }
  }

  /** Returns the {@link StarlarkConfigurationField} annotation corresponding to this method. */
  @Nullable
  private static Label getDefaultLabel(
      StarlarkConfigurationField annotation, RepositoryName toolsRepository) {
    if (annotation.defaultLabel().isEmpty()) {
      return null;
    }
    Label defaultLabel =
        annotation.defaultInToolRepository()
            ? Label.parseCanonicalUnchecked(toolsRepository + annotation.defaultLabel())
            : Label.parseCanonicalUnchecked(annotation.defaultLabel());
    return defaultLabel;
  }

  private StarlarkLateBoundDefault(
      StarlarkConfigurationField annotation,
      Class<FragmentT> fragmentClass,
      String fragmentName,
      Method method,
      RepositoryName toolsRepository) {
    this(
        getDefaultLabel(annotation, toolsRepository),
        fragmentClass,
        method,
        fragmentName,
        annotation.name());
  }

  private StarlarkLateBoundDefault(
      Label defaultVal,
      Class<FragmentT> fragmentClass,
      Method method,
      String fragmentName,
      String fragmentFieldName) {
    super(fragmentClass, defaultVal);
    this.method = method;
    this.fragmentName = fragmentName;
    this.fragmentFieldName = fragmentFieldName;
  }

  /**
   * Returns the Starlark name of the configuration fragment that this late bound default requires.
   */
  public String getFragmentName() {
    return fragmentName;
  }

  /**
   * Returns the Starlark name of the configuration field name, as registered by {@link
   * StarlarkConfigurationField} annotation on the configuration fragment.
   */
  public String getFragmentFieldName() {
    return fragmentFieldName;
  }

  @Override
  public void repr(Printer printer) {
    printer.append("<late-bound default>");
  }

  /**
   * An exception thrown if a user specifies an invalid configuration field identifier.
   *
   * @see StarlarkConfigurationField
   */
  public static class InvalidConfigurationFieldException extends Exception {
    public InvalidConfigurationFieldException(String message) {
      super(message);
    }
  }


  private static class CacheKey {
    private final Class<?> fragmentClass;
    private final RepositoryName toolsRepository;

    private CacheKey(Class<?> fragmentClass, RepositoryName toolsRepository) {
      this.fragmentClass = fragmentClass;
      this.toolsRepository = toolsRepository;
    }

    @Override
    public boolean equals(Object object) {
      if (object == this) {
        return true;
      } else if (!(object instanceof CacheKey cacheKey)) {
        return false;
      } else {
        return fragmentClass.equals(cacheKey.fragmentClass)
            && toolsRepository.equals(cacheKey.toolsRepository);
      }
    }

    @Override
    public int hashCode() {
      int result = fragmentClass.hashCode();
      result = 31 * result + toolsRepository.hashCode();
      return result;
    }
  }

  /**
   * A cache for efficient {@link StarlarkLateBoundDefault} loading by configuration fragment. Each
   * configuration fragment class key is mapped to a {@link Map} where keys are configuration field
   * Starlark names, and values are the {@link StarlarkLateBoundDefault}s. Methods must be annotated
   * with {@link StarlarkConfigurationField} to be considered.
   */
  private static final LoadingCache<CacheKey, Map<String, StarlarkLateBoundDefault<?>>> fieldCache =
      Caffeine.newBuilder()
          .initialCapacity(10)
          .maximumSize(100)
          .build(
              key -> {
                ImmutableMap.Builder<String, StarlarkLateBoundDefault<?>> lateBoundDefaultMap =
                    new ImmutableMap.Builder<>();
                Class<?> fragmentClass = key.fragmentClass;
                StarlarkBuiltin fragmentModule =
                    StarlarkAnnotations.getStarlarkBuiltin(fragmentClass);

                if (fragmentModule != null) {
                  for (Method method : fragmentClass.getMethods()) {
                    if (method.isAnnotationPresent(StarlarkConfigurationField.class)) {
                      // TODO(b/68817606): Use annotation processors to verify these constraints.
                      Preconditions.checkArgument(
                          method.getReturnType() == Label.class,
                          "Method %s must have return type 'Label'",
                          method);
                      Preconditions.checkArgument(
                          method.getParameterTypes().length == 0,
                          "Method %s must not accept arguments",
                          method);

                      StarlarkConfigurationField configField =
                          method.getAnnotation(StarlarkConfigurationField.class);
                      lateBoundDefaultMap.put(
                          configField.name(),
                          new StarlarkLateBoundDefault<>(
                              configField,
                              fragmentClass,
                              fragmentModule.name(),
                              method,
                              key.toolsRepository));
                    }
                  }
                }
                return lateBoundDefaultMap.buildOrThrow();
              });

  /**
   * Returns a {@link LateBoundDefault} which obtains a late-bound attribute value (of type 'label')
   * specifically by Starlark configuration fragment name and field name, as registered by {@link
   * StarlarkConfigurationField}.
   *
   * @param fragmentClass the configuration fragment class, which must have a valid Starlark name
   * @param fragmentFieldName the configuration field name, as registered by {@link
   *     StarlarkConfigurationField} annotation
   * @param toolsRepository the Bazel tools repository path fragment
   * @throws InvalidConfigurationFieldException if there is no valid configuration field with the
   *     given fragment class and field name
   */
  @SuppressWarnings("unchecked")
  public static <FragmentT> StarlarkLateBoundDefault<FragmentT> forConfigurationField(
      Class<FragmentT> fragmentClass, String fragmentFieldName, RepositoryName toolsRepository)
      throws InvalidConfigurationFieldException {
      CacheKey cacheKey = new CacheKey(fragmentClass, toolsRepository);
      StarlarkLateBoundDefault<?> resolver = fieldCache.get(cacheKey).get(fragmentFieldName);
      if (resolver == null) {
        StarlarkBuiltin moduleAnnotation = StarlarkAnnotations.getStarlarkBuiltin(fragmentClass);
        if (moduleAnnotation == null) {
          throw new AssertionError("fragment class must have a valid Starlark name");
        }
        throw new InvalidConfigurationFieldException(
            String.format("invalid configuration field name '%s' on fragment '%s'",
                fragmentFieldName, moduleAnnotation.name()));
      }
      return (StarlarkLateBoundDefault<FragmentT>) resolver; // unchecked cast
  }

}
