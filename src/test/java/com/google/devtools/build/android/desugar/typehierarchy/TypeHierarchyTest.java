/*
 * Copyright 2020 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.android.desugar.typehierarchy;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth8.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.android.desugar.langmodel.ClassName;
import com.google.devtools.build.android.desugar.langmodel.MethodKey;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Functional tests for {@link TypeHierarchy}.
 *
 * <ol>
 *   <li>
 *       <p>Mixed classes and interfaces.
 *       <table border="0" width="96px"  style="text-align:center">
 *      <tr style="text-align:center">
 *        <td width="25%">CZ</td>
 *        <td width="25%"></td>
 *        <td width="25%"></td>
 *        <td width="25%"></td>
 *      </tr>
 *      <tr style="text-align:center">
 *        <td>&#11107;</td>
 *        <td></td>
 *        <td></td>
 *        <td></td>
 *      </tr>
 *      <tr style="text-align:center">
 *        <td>CA</td>
 *        <td>&#11106;</td>
 *        <td>IA</td>
 *        <td></td>
 *      </tr>
 *      <tr style="text-align:center">
 *        <td>&#11107;</td>
 *        <td>&#8600;</td>
 *        <td>&#11107;</td>
 *        <td>&#8600;</td>
 *      </tr>
 *      <tr style="text-align:center">
 *        <td>CB</td>
 *        <td>&#11106;</td>
 *        <td>IB</td>
 *        <td>&#11107;</td>
 *      </tr>
 *      <tr style="text-align:center">
 *        <td>&#11107;</td>
 *        <td>&#8600;</td>
 *        <td>&#11107;</td>
 *        <td>&#8601;</td>
 *      </tr>
 *      <tr style="text-align:center">
 *        <td>CC</td>
 *        <td>&#11106;</td>
 *        <td>IC</td>
 *        <td></td>
 *      </tr>
 *      <tr style="text-align:center">
 *        <td>&#11107;</td>
 *        <td></td>
 *        <td>&#11107;</td>
 *        <td></td>
 *      </tr>
 *      <tr style="text-align:center">
 *        <td>CPC</td>
 *        <td></td>
 *        <td>IZ</td>
 *        <td></td>
 *      </tr>
 *  </table>
 *       <ul>
 *         <li>C: Class
 *         <li>I: Interface
 *         <li>P: package
 *         <li>The arrow indicates class inheritance.
 *       </ul>
 *   <li>
 *       <p>Overridable methods with package-private visibility.
 *       <table border="0" width="72px" style="text-align:center">
 *      <tr style="text-align:center">
 *        <td width="33%">CX</td>
 *        <td width="33%">&#11106;</td>
 *        <td width="33%">CY</td>
 *      </tr>
 *      <tr style="text-align:center">
 *        <td>&#11105;</td>
 *        <td></td>
 *        <td></td>
 *      </tr>
 *      <tr style="text-align:center">
 *        <td>BX</td>
 *        <td>&#11104;</td>
 *        <td>BY</td>
 *      </tr>
 *      <tr style="text-align:center">
 *        <td></td>
 *        <td></td>
 *        <td>&#11105;</td>
 *      </tr>
 *      <tr style="text-align:center">
 *        <td>AX</td>
 *        <td>&#11106;</td>
 *        <td>AY</td>
 *      </tr>
 *  </table>
 *       <ul>
 *         <li>X: In package X (pkgx)
 *         <li>Y: In package Y (pkgy)
 *       </ul>
 */
@RunWith(JUnit4.class)
public class TypeHierarchyTest {

  private static final Path INPUT_JAR_PATH = Paths.get(System.getProperty("input_lib"));
  private static final String TEST_LIB_ROOT =
      "com/google/devtools/build/android/desugar/typehierarchy/testlib/";
  private TypeHierarchy typeHierarchy;
  private static final HierarchicalTypeKey OBJECT_CLASS =
      HierarchicalTypeKey.create(ClassName.create("java/lang/Object"));
  private static final HierarchicalTypeKey CLASS_ALPHA =
      HierarchicalTypeKey.create(ClassName.create(TEST_LIB_ROOT + "pkga/ClassAlpha"));
  private static final HierarchicalTypeKey INTERFACE_ALPHA =
      HierarchicalTypeKey.create(ClassName.create(TEST_LIB_ROOT + "pkga/InterfaceAlpha"));
  private static final HierarchicalTypeKey CLASS_BRAVO =
      HierarchicalTypeKey.create(ClassName.create(TEST_LIB_ROOT + "pkgb/ClassBravo"));
  private static final HierarchicalTypeKey INTERFACE_BRAVO =
      HierarchicalTypeKey.create(ClassName.create(TEST_LIB_ROOT + "pkgb/InterfaceBravo"));
  private static final HierarchicalTypeKey CLASS_CHARLIE =
      HierarchicalTypeKey.create(ClassName.create(TEST_LIB_ROOT + "pkgc/ClassCharlie"));
  private static final HierarchicalTypeKey CLASS_PACKAGE_CHARLIE =
      HierarchicalTypeKey.create(ClassName.create(TEST_LIB_ROOT + "pkgc/ClassPackageCharlie"));
  private static final HierarchicalTypeKey INTERFACE_CHARLIE =
      HierarchicalTypeKey.create(ClassName.create(TEST_LIB_ROOT + "pkgc/InterfaceCharlie"));
  private static final HierarchicalTypeKey CLASS_ZULU =
      HierarchicalTypeKey.create(ClassName.create(TEST_LIB_ROOT + "ClassZulu"));
  private static final HierarchicalTypeKey INTERFACE_ZULU =
      HierarchicalTypeKey.create(ClassName.create(TEST_LIB_ROOT + "InterfaceZulu"));

  private static final HierarchicalTypeKey PIE_AX =
      HierarchicalTypeKey.create(ClassName.create(TEST_LIB_ROOT + "pkgx/PieAX"));
  private static final HierarchicalTypeKey PIE_BX =
      HierarchicalTypeKey.create(ClassName.create(TEST_LIB_ROOT + "pkgx/PieBX"));
  private static final HierarchicalTypeKey PIE_CX =
      HierarchicalTypeKey.create(ClassName.create(TEST_LIB_ROOT + "pkgx/PieCX"));
  private static final HierarchicalTypeKey PIE_AY =
      HierarchicalTypeKey.create(ClassName.create(TEST_LIB_ROOT + "pkgy/PieAY"));
  private static final HierarchicalTypeKey PIE_BY =
      HierarchicalTypeKey.create(ClassName.create(TEST_LIB_ROOT + "pkgy/PieBY"));
  private static final HierarchicalTypeKey PIE_CY =
      HierarchicalTypeKey.create(ClassName.create(TEST_LIB_ROOT + "pkgy/PieCY"));

  @Before
  public void setUp() {
    typeHierarchy = TypeHierarchyScavenger.analyze(ImmutableList.of(INPUT_JAR_PATH), true);
  }

  @Test
  public void classAlpha() {
    HierarchicalTypeQuery classAlpha = CLASS_ALPHA.inTypeHierarchy(typeHierarchy);

    assertThat(classAlpha.findDirectSuperClass()).isEqualTo(CLASS_BRAVO);
    assertThat(classAlpha.findDirectSuperInterfaces())
        .containsExactly(INTERFACE_ALPHA, INTERFACE_BRAVO);
    assertThat(classAlpha.findDirectSuperTypes())
        .containsExactly(CLASS_BRAVO, INTERFACE_ALPHA, INTERFACE_BRAVO);

    assertThat(classAlpha.findTransitiveSuperClasses())
        .containsExactly(CLASS_BRAVO, CLASS_CHARLIE, CLASS_PACKAGE_CHARLIE, OBJECT_CLASS);
    assertThat(classAlpha.findTransitiveSuperInterfaces())
        .containsExactly(INTERFACE_ALPHA, INTERFACE_BRAVO, INTERFACE_CHARLIE, INTERFACE_ZULU);
    assertThat(classAlpha.findTransitiveSuperTypes())
        .containsExactly(
            CLASS_BRAVO,
            CLASS_CHARLIE,
            CLASS_PACKAGE_CHARLIE,
            OBJECT_CLASS,
            INTERFACE_ALPHA,
            INTERFACE_BRAVO,
            INTERFACE_CHARLIE,
            INTERFACE_ZULU);
  }

  @Test
  public void interfaceAlpha() {
    HierarchicalTypeQuery interfaceAlpha = INTERFACE_ALPHA.inTypeHierarchy(typeHierarchy);

    assertThat(interfaceAlpha.findDirectSuperClass()).isEqualTo(OBJECT_CLASS);
    assertThat(interfaceAlpha.findDirectSuperInterfaces())
        .containsExactly(INTERFACE_BRAVO, INTERFACE_CHARLIE);
    assertThat(interfaceAlpha.findDirectSuperTypes())
        .containsExactly(OBJECT_CLASS, INTERFACE_BRAVO, INTERFACE_CHARLIE);

    assertThat(interfaceAlpha.findTransitiveSuperClasses()).containsExactly(OBJECT_CLASS);
    assertThat(interfaceAlpha.findTransitiveSuperInterfaces())
        .containsExactly(INTERFACE_BRAVO, INTERFACE_CHARLIE, INTERFACE_ZULU);
    assertThat(interfaceAlpha.findTransitiveSuperTypes())
        .containsExactly(OBJECT_CLASS, INTERFACE_BRAVO, INTERFACE_CHARLIE, INTERFACE_ZULU);
  }

  @Test
  public void classBravo() {
    HierarchicalTypeQuery classBravo = CLASS_BRAVO.inTypeHierarchy(typeHierarchy);

    assertThat(classBravo.findDirectSuperClass()).isEqualTo(CLASS_CHARLIE);
    assertThat(classBravo.findDirectSuperInterfaces())
        .containsExactly(INTERFACE_BRAVO, INTERFACE_CHARLIE);
    assertThat(classBravo.findDirectSuperTypes())
        .containsExactly(CLASS_CHARLIE, INTERFACE_BRAVO, INTERFACE_CHARLIE);

    assertThat(classBravo.findTransitiveSuperClasses())
        .containsExactly(CLASS_CHARLIE, CLASS_PACKAGE_CHARLIE, OBJECT_CLASS);
    assertThat(classBravo.findTransitiveSuperInterfaces())
        .containsExactly(INTERFACE_BRAVO, INTERFACE_CHARLIE, INTERFACE_ZULU);
    assertThat(classBravo.findTransitiveSuperTypes())
        .containsExactly(
            CLASS_CHARLIE,
            CLASS_PACKAGE_CHARLIE,
            OBJECT_CLASS,
            INTERFACE_BRAVO,
            INTERFACE_CHARLIE,
            INTERFACE_ZULU);
  }

  @Test
  public void interfaceBravo() {
    HierarchicalTypeQuery interfaceBravo = INTERFACE_BRAVO.inTypeHierarchy(typeHierarchy);

    assertThat(interfaceBravo.findDirectSuperClass()).isEqualTo(OBJECT_CLASS);
    assertThat(interfaceBravo.findDirectSuperInterfaces()).containsExactly(INTERFACE_CHARLIE);
    assertThat(interfaceBravo.findDirectSuperTypes())
        .containsExactly(OBJECT_CLASS, INTERFACE_CHARLIE);

    assertThat(interfaceBravo.findTransitiveSuperClasses()).containsExactly(OBJECT_CLASS);
    assertThat(interfaceBravo.findTransitiveSuperInterfaces())
        .containsExactly(INTERFACE_CHARLIE, INTERFACE_ZULU);
    assertThat(interfaceBravo.findTransitiveSuperTypes())
        .containsExactly(OBJECT_CLASS, INTERFACE_CHARLIE, INTERFACE_ZULU);
  }

  @Test
  public void classCharlie() {
    HierarchicalTypeQuery classCharlie = CLASS_CHARLIE.inTypeHierarchy(typeHierarchy);

    assertThat(classCharlie.findDirectSuperClass()).isEqualTo(CLASS_PACKAGE_CHARLIE);
    assertThat(classCharlie.findDirectSuperInterfaces()).containsExactly(INTERFACE_CHARLIE);
    assertThat(classCharlie.findDirectSuperTypes())
        .containsExactly(CLASS_PACKAGE_CHARLIE, INTERFACE_CHARLIE);

    assertThat(classCharlie.findTransitiveSuperClasses())
        .containsExactly(CLASS_PACKAGE_CHARLIE, OBJECT_CLASS);
    assertThat(classCharlie.findTransitiveSuperInterfaces())
        .containsExactly(INTERFACE_CHARLIE, INTERFACE_ZULU);
    assertThat(classCharlie.findTransitiveSuperTypes())
        .containsExactly(CLASS_PACKAGE_CHARLIE, OBJECT_CLASS, INTERFACE_CHARLIE, INTERFACE_ZULU);
  }

  @Test
  public void interfaceCharlie() {
    HierarchicalTypeQuery interfaceCharlie = INTERFACE_CHARLIE.inTypeHierarchy(typeHierarchy);

    assertThat(interfaceCharlie.findDirectSuperClass()).isEqualTo(OBJECT_CLASS);
    assertThat(interfaceCharlie.findDirectSuperInterfaces()).containsExactly(INTERFACE_ZULU);
    assertThat(interfaceCharlie.findDirectSuperTypes())
        .containsExactly(OBJECT_CLASS, INTERFACE_ZULU);

    assertThat(interfaceCharlie.findTransitiveSuperClasses()).containsExactly(OBJECT_CLASS);
    assertThat(interfaceCharlie.findTransitiveSuperInterfaces()).containsExactly(INTERFACE_ZULU);
    assertThat(interfaceCharlie.findTransitiveSuperTypes())
        .containsExactly(OBJECT_CLASS, INTERFACE_ZULU);
  }

  @Test
  public void classZulu() {
    HierarchicalTypeQuery classZulu = CLASS_ZULU.inTypeHierarchy(typeHierarchy);

    assertThat(classZulu.findDirectSuperClass()).isEqualTo(CLASS_ALPHA);
    assertThat(classZulu.findDirectSuperInterfaces()).isEmpty();
    assertThat(classZulu.findDirectSuperTypes()).containsExactly(CLASS_ALPHA);

    assertThat(classZulu.findTransitiveSuperClasses())
        .containsExactly(
            CLASS_ALPHA, CLASS_BRAVO, CLASS_CHARLIE, CLASS_PACKAGE_CHARLIE, OBJECT_CLASS);
    assertThat(classZulu.findTransitiveSuperInterfaces())
        .containsExactly(INTERFACE_ALPHA, INTERFACE_BRAVO, INTERFACE_CHARLIE, INTERFACE_ZULU);
    assertThat(classZulu.findTransitiveSuperTypes())
        .containsExactly(
            CLASS_ALPHA,
            CLASS_BRAVO,
            CLASS_CHARLIE,
            CLASS_PACKAGE_CHARLIE,
            OBJECT_CLASS,
            INTERFACE_ALPHA,
            INTERFACE_BRAVO,
            INTERFACE_CHARLIE,
            INTERFACE_ZULU);
  }

  @Test
  public void interfaceZulu() {
    HierarchicalTypeQuery classZulu = INTERFACE_ZULU.inTypeHierarchy(typeHierarchy);

    assertThat(classZulu.findDirectSuperClass()).isEqualTo(OBJECT_CLASS);
    assertThat(classZulu.findDirectSuperInterfaces()).isEmpty();
    assertThat(classZulu.findDirectSuperTypes()).containsExactly(OBJECT_CLASS);

    assertThat(classZulu.findTransitiveSuperClasses()).containsExactly(OBJECT_CLASS);
    assertThat(classZulu.findTransitiveSuperInterfaces()).isEmpty();
    assertThat(classZulu.findTransitiveSuperTypes()).containsExactly(OBJECT_CLASS);
  }

  @Test
  public void classZulu_getTag_publicVisibility() {
    HierarchicalMethodQuery getTag =
        HierarchicalMethodKey.from(
                MethodKey.create(
                    /* ownerClass= */ CLASS_ZULU.type(),
                    /* name= */ "getTag",
                    /* descriptor= */ "()Ljava/lang/String;"))
            .inTypeHierarchy(typeHierarchy);

    assertThat(getTag.getBaseInterfaceMethods().stream().map(HierarchicalMethodKey::owner))
        .containsExactly(INTERFACE_ZULU);
    assertThat(getTag.getBaseClassMethods().stream().map(HierarchicalMethodKey::owner))
        .containsExactly(CLASS_ALPHA, CLASS_BRAVO, CLASS_CHARLIE);
    assertThat(getTag.getBaseMethods().stream().map(HierarchicalMethodKey::owner))
        .containsExactly(CLASS_ALPHA, CLASS_BRAVO, CLASS_CHARLIE, INTERFACE_ZULU);
  }

  @Test
  public void classPieAX_twoOperands_packageVisibilityFiltering() {
    HierarchicalMethodQuery withTwoOperands =
        HierarchicalMethodKey.from(
                MethodKey.create(
                    /* ownerClass= */ PIE_AX.type(),
                    /* name= */ "withTwoOperands",
                    /* descriptor= */ "(JJ)J"))
            .inTypeHierarchy(typeHierarchy);

    assertThat(withTwoOperands.getBaseClassMethods().stream().map(HierarchicalMethodKey::owner))
        .containsExactly(PIE_BX, PIE_CX);
    assertThat(withTwoOperands.getBaseMethods().stream().map(HierarchicalMethodKey::owner))
        .containsExactly(PIE_BX, PIE_CX);
  }

  @Test
  public void classPieAY_twoOperands_packageVisibilityFiltering() {
    HierarchicalMethodQuery withTwoOperands =
        HierarchicalMethodKey.from(
                MethodKey.create(
                    /* ownerClass= */ PIE_AY.type(),
                    /* name= */ "withTwoOperands",
                    /* descriptor= */ "(JJ)J"))
            .inTypeHierarchy(typeHierarchy);

    assertThat(withTwoOperands.getBaseClassMethods().stream().map(HierarchicalMethodKey::owner))
        .containsExactly(PIE_BY, PIE_CY);
    assertThat(withTwoOperands.getBaseMethods().stream().map(HierarchicalMethodKey::owner))
        .containsExactly(PIE_BY, PIE_CY);
  }
}
