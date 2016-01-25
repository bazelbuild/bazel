// Copyright 2015 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.packages;

/**
 * A class of aspects that are implemented natively in Bazel.
 *
 * <p>This class just wraps a {@link java.lang.Class} implementing the
 * aspect factory. All wrappers of the same class are equal.
 */
public final class NativeAspectClass<T extends NativeAspectClass.NativeAspectFactory>
    implements AspectClass {
  private final Class<? extends T> nativeClass;

  public NativeAspectClass(Class<? extends T> nativeClass) {
    this.nativeClass = nativeClass;
  }

  @Override
  public String getName() {
    return nativeClass.getSimpleName();
  }

  @Override
  public AspectDefinition getDefinition(AspectParameters aspectParameters) {
    return newInstance().getDefinition(aspectParameters);
  }

  public T newInstance() {
    try {
      return nativeClass.newInstance();
    } catch (Exception e) {
      throw new IllegalStateException(e);
    }
  }

  @Override
  public int hashCode() {
    return nativeClass.hashCode();
  }

  @Override
  public boolean equals(Object obj) {
    if (!(obj instanceof NativeAspectClass)) {
      return false;
    }
    return nativeClass.equals(((NativeAspectClass<?>) obj).nativeClass);
  }

  /**
   * Every native aspect should implement this interface.
   */
  public interface NativeAspectFactory {
    /**
     * Returns the definition of the aspect.
     */
    AspectDefinition getDefinition(AspectParameters aspectParameters);
  }
}
