// Copyright 2020 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android.r8.desugar;

import com.android.tools.r8.ClassFileResourceProvider;
import com.android.tools.r8.ProgramResource;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Sets;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

/**
 * Classpath provider which will de-dupe duplicate classes from several providers. For any defined
 * class the definition from the first provider defining the class is used.
 */
public class OrderedClassFileResourceProvider implements ClassFileResourceProvider {
  private final Set<String> descriptors = Sets.newHashSet();
  private final Map<String, ClassFileResourceProvider> descriptorToProvider = new HashMap<>();

  public OrderedClassFileResourceProvider(
      ImmutableList<ClassFileResourceProvider> bootclasspathProviders,
      ImmutableList<ClassFileResourceProvider> classfileProviders) {
    final Set<String> bootclasspathDescriptors = Sets.newHashSet();
    bootclasspathProviders.forEach(p -> bootclasspathDescriptors.addAll(p.getClassDescriptors()));
    for (ClassFileResourceProvider provider : classfileProviders) {
      // Collect all descriptors provided and the first provider providing each.
      for (String descriptor : provider.getClassDescriptors()) {
        // Pick first definition of classpath class and filter out platform classes
        // from classpath if present.
        if (!bootclasspathDescriptors.contains(descriptor)
            && descriptors.add(descriptor)) {
          descriptorToProvider.put(descriptor, provider);
        }
      }
    }
  }

  @Override
  public Set<String> getClassDescriptors() {
    return descriptors;
  }

  @Override
  public ProgramResource getProgramResource(String descriptor) {
    ClassFileResourceProvider provider = descriptorToProvider.get(descriptor);
    return provider != null ? provider.getProgramResource(descriptor) : null;
  }
}
