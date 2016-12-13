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
package com.google.devtools.build.lib.analysis;

import com.google.common.base.Function;
import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.rules.SkylarkApiProvider;
import com.google.devtools.build.lib.syntax.ClassObject;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.EvalUtils;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;

import java.util.Map;

/**
 * Utility class to validate results of executing Skylark rules and aspects.
 */
public class SkylarkProviderValidationUtil {
  /**
   * Check if the value provided by a Skylark provider is safe (i.e. can be a
   * TransitiveInfoProvider value).
   */
  public static void checkSkylarkObjectSafe(Object value) {
    if (!isSimpleSkylarkObjectSafe(value.getClass())
        // Java transitive Info Providers are accessible from Skylark.
        && !(value instanceof TransitiveInfoProvider)) {
      checkCompositeSkylarkObjectSafe(value);
    }
  }

  /**
   * Check if the value provided by a Skylark provider is safe (i.e. can be a
   * TransitiveInfoProvider value).
   * Throws {@link EvalException} if not.
   */
  public static void validateAndThrowEvalException(String providerName, Object value, Location loc)
      throws EvalException {
    try {
      checkSkylarkObjectSafe(value);
    } catch (IllegalArgumentException e) {
      throw new EvalException(
          loc,
          String.format(
              "Value of provider '%s' is of an illegal type: %s", providerName, e.getMessage()));
    }
  }


  private static void checkCompositeSkylarkObjectSafe(Object object) {
    if (object instanceof SkylarkApiProvider) {
      return;
    } else if (object instanceof SkylarkList) {
      SkylarkList list = (SkylarkList) object;
      if (list.isEmpty()) {
        // Try not to iterate over the list if avoidable.
        return;
      }
      // The list can be a tuple or a list of composite items.
      for (Object listItem : list) {
        checkSkylarkObjectSafe(listItem);
      }
      return;
    } else if (object instanceof SkylarkNestedSet) {
      // SkylarkNestedSets cannot have composite items.
      Class<?> contentType = ((SkylarkNestedSet) object).getContentType().getType();
      if (!contentType.equals(Object.class) && !isSimpleSkylarkObjectSafe(contentType)) {
        throw new IllegalArgumentException(EvalUtils.getDataTypeNameFromClass(contentType));
      }
      return;
    } else if (object instanceof Map<?, ?>) {
      for (Map.Entry<?, ?> entry : ((Map<?, ?>) object).entrySet()) {
        checkSkylarkObjectSafe(entry.getKey());
        checkSkylarkObjectSafe(entry.getValue());
      }
      return;
    } else if (object instanceof ClassObject) {
      ClassObject struct = (ClassObject) object;
      for (String key : struct.getKeys()) {
        checkSkylarkObjectSafe(struct.getValue(key));
      }
      return;
    }
    throw new IllegalArgumentException(EvalUtils.getDataTypeName(object));
  }

  private static boolean isSimpleSkylarkObjectSafe(Class<?> type) {
    return type.equals(String.class)
        || type.equals(Integer.class)
        || type.equals(Boolean.class)
        || Artifact.class.isAssignableFrom(type)
        || ActionAnalysisMetadata.class.isAssignableFrom(type)
        || type.equals(Label.class)
        || type.equals(com.google.devtools.build.lib.syntax.Runtime.NoneType.class);
  }

  public static void checkOrphanArtifacts(RuleContext ruleContext) throws EvalException {
    ImmutableSet<Artifact> orphanArtifacts =
        ruleContext.getAnalysisEnvironment().getOrphanArtifacts();
    if (!orphanArtifacts.isEmpty()) {
      throw new EvalException(null, "The following files have no generating action:\n"
          + Joiner.on("\n").join(Iterables.transform(orphanArtifacts,
          new Function<Artifact, String>() {
            @Override
            public String apply(Artifact artifact) {
              return artifact.getRootRelativePathString();
            }
          })));
    }
  }
}
