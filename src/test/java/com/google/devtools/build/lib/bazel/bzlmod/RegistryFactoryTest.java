// Copyright 2022 The Bazel Authors. All rights reserved.
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
//

package com.google.devtools.build.lib.bazel.bzlmod;

import static org.junit.Assert.assertThrows;

import com.google.common.base.Suppliers;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.bazel.repository.downloader.HttpDownloader;
import java.net.URISyntaxException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link RegistryFactory}. */
@RunWith(JUnit4.class)
public class RegistryFactoryTest {

  @Test
  public void badSchemes() throws Exception {
    RegistryFactory registryFactory =
        new RegistryFactoryImpl(new HttpDownloader(), Suppliers.ofInstance(ImmutableMap.of()));
    assertThrows(URISyntaxException.class, () -> registryFactory.getRegistryWithUrl("/home/www"));
    assertThrows(URISyntaxException.class, () -> registryFactory.getRegistryWithUrl("foo://bar"));
  }
}
