// Copyright 2010 The Bazel Authors. All Rights Reserved.
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

package com.google.testing.junit.runner.junit4;

import com.google.testing.junit.junit4.runner.CategoryFilter;
import com.google.testing.junit.runner.util.GoogleTestSecurityManager;
import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.Properties;

import static com.google.common.truth.Truth.assertThat;
import static com.google.testing.junit.runner.junit4.JUnit4Config.JUNIT_API_VERSION_PROPERTY;
import static com.google.testing.junit.runner.junit4.JUnit4Config.SHOULD_INSTALL_SECURITY_MANAGER_PROPERTY;
import static org.junit.Assert.assertThrows;

/**
 * Tests for {@link JUnit4Config}.
 */
@RunWith(JUnit4.class)
public class JUnit4ConfigTest {

  private JUnit4Config createConfigWithApiVersion(String apiVersion) {
    Properties properties = new Properties();
    properties.put(JUNIT_API_VERSION_PROPERTY, apiVersion);
    return createConfigWithProperties(properties);
  }

  private JUnit4Config createConfigWithProperties(Properties properties) {
    return new JUnit4Config("", null, null, null, null, properties);
  }

  @Test
  public void testGetJUnitRunnerApiVersion_defaultValue() {
    JUnit4Config config = createConfigWithApiVersion("1");
    assertThat(config.getJUnitRunnerApiVersion()).isEqualTo(1);
  }

  @Test
  public void testGetJUnitRunnerApiVersion_failsIfNotNumeric() {
    JUnit4Config config = createConfigWithApiVersion("I love pesto");

    IllegalStateException expected =
        assertThrows(IllegalStateException.class, () -> config.getJUnitRunnerApiVersion());
    assertThat(expected).hasMessageThat().contains("I love pesto");
  }

  @Test
  public void testGetJUnitRunnerApiVersion_failsIfNotAnInteger() {
    JUnit4Config config = createConfigWithApiVersion("3.14");

    IllegalStateException expected =
        assertThrows(IllegalStateException.class, () -> config.getJUnitRunnerApiVersion());
    assertThat(expected).hasMessageThat().contains("3.14");
  }

  @Test
  public void testGetJUnitRunnerApiVersion_failsIfNotOne() {
    JUnit4Config config = createConfigWithApiVersion("13");

    IllegalStateException expected =
        assertThrows(IllegalStateException.class, () -> config.getJUnitRunnerApiVersion());
    assertThat(expected).hasMessageThat().contains("13");
  }

  @Test
  public void testGetJUnitRunnerApiVersion_oneIsValid() {
    JUnit4Config config = createConfigWithApiVersion("1");
    assertThat(config.getJUnitRunnerApiVersion()).isEqualTo(1);
  }

  @Test
  public void testShouldInstallSecurityManager_defaultValue() {
    GoogleTestSecurityManager.uninstallIfInstalled();

    JUnit4Config config = createConfigWithProperties(new Properties());
    assertThat(config.shouldInstallSecurityManager()).isTrue();
  }

  @Test
  public void testShouldInstallSecurityManager_securityManagerPropertySet() {
    Properties properties = new Properties();
    properties.put("java.security.manager", "MySecurityManager");
    properties.put(SHOULD_INSTALL_SECURITY_MANAGER_PROPERTY, "true");
    JUnit4Config config = createConfigWithProperties(properties);
    assertThat(config.shouldInstallSecurityManager()).isFalse();
  }

  @Test
  public void testShouldInstallSecurityManager_shouldInstallSecurityManagerPropertySetToFalse() {
    Properties properties = new Properties();
    properties.put(SHOULD_INSTALL_SECURITY_MANAGER_PROPERTY, "false");
    JUnit4Config config = createConfigWithProperties(properties);
    assertThat(config.shouldInstallSecurityManager()).isFalse();
  }

  @Test
  public void testThrowsOnBadCategoryClasses() {
    Assert.assertThrows(
            RuntimeException.class,
            () -> new JUnit4Config(null, null, "not.a.real.Class", null, null));
    Assert.assertThrows(
            RuntimeException.class,
            () -> new JUnit4Config(null, null, null, "not.a.real.Class", null));
  }

  @Test
  public void testCreatesCategoryFilter() {
    JUnit4Config config = new JUnit4Config(null, null, Object.class.getName(), Boolean.class.getName(), null);
    assertThat(config.getCategoriesFilter()).isInstanceOf(CategoryFilter.class);
  }

}
