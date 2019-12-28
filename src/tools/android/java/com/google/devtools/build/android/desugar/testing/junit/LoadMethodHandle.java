/*
 * Copyright 2019 The Bazel Authors. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.devtools.build.android.desugar.testing.junit;

import static com.google.devtools.build.android.desugar.testing.junit.LoadMethodHandle.MemberUseContext.METHOD_INVOCATION;

import com.google.common.annotations.UsedReflectively;
import java.lang.annotation.Documented;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;
import org.objectweb.asm.tree.ClassNode;

/**
 * Identifies injectable ASM node fields (e.g. {@link org.objectweb.asm.tree.ClassNode}, {@link
 * org.objectweb.asm.tree.MethodNode}, {@link org.objectweb.asm.tree.FieldNode}) with a qualified
 * class name. The desugar rule resolves the requested class at runtime, parses it into a {@link
 * ClassNode}, and assigns the parsed class node to the annotated field. An injectable ASM node
 * field may have any access modifier (private, package-private, protected, public). Sample usage:
 *
 * <pre><code>
 * &#064;RunWith(JUnit4.class)
 * public class DesugarRuleTest {
 *
 *   &#064;Rule
 *   public final DesugarRule desugarRule =
 *       DesugarRule.builder(this, MethodHandles.lookup())
 *           .addRuntimeInputs("path/to/my_jar.jar")
 *           .build();
 *
 *   &#064;LoadMethodHandle(
 *       className = "my.package.ClassToDesugar",
 *       memberName = "add",
 *       memberDescriptor = "(II)I",
 *       round = 2,
 *   )
 *   private MethodHandle adder;
 *
 *   // ... Test methods ...
 * }
 * </code></pre>
 */
@UsedReflectively
@Documented
@Target(ElementType.FIELD)
@Retention(RetentionPolicy.RUNTIME)
public @interface LoadMethodHandle {

  /**
   * The fully-qualified class name of the class to load. The format agrees with {@link
   * Class#getName}.
   */
  String className();

  /** If non-empty, load the specified class member (field or method) from the enclosing class. */
  String memberName() default "";

  /** If non-empty, use the specified member descriptor to disambiguate overloaded methods. */
  String memberDescriptor() default "";

  MemberUseContext usage() default METHOD_INVOCATION;

  /** The round during which its associated jar is being used. */
  int round() default 1;

  /** All class member use context types. */
  enum MemberUseContext {
    METHOD_INVOCATION,
    FIELD_GETTER,
    FIELD_SETTER,
  }
}
