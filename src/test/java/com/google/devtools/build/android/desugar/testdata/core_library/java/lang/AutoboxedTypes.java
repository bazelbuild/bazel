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

// This test class is in the java.lang namespace to trigger the hardcoded JVM restrictions that
// desugar --core_library works around
package java.lang;

/**
 * This class will be desugared with --core_library and then functionally tested by {@code
 * DesugarCoreLibraryFunctionalTest}
 */
public class AutoboxedTypes {
  /**
   * Dummy functional interface for autoboxedTypeLambda to return without introducing a dependency
   * on any other java.* classes.
   */
  @FunctionalInterface
  public interface Lambda {
    String charAt(String s);
  }

  public static Lambda autoboxedTypeLambda(Integer i) {
    return n -> n.substring(i, i + 1);
  }
}
