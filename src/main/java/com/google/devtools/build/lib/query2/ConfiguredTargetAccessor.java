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
package com.google.devtools.build.lib.query2;

import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.TargetAccessor;
import com.google.devtools.build.lib.query2.engine.QueryException;
import com.google.devtools.build.lib.query2.engine.QueryExpression;
import com.google.devtools.build.lib.query2.engine.QueryVisibility;
import java.util.List;
import java.util.Set;

/** A {@link TargetAccessor} for {@link ConfiguredTarget} objects. Incomplete. */
class ConfiguredTargetAccessor implements TargetAccessor<ConfiguredTarget> {
  @Override
  public String getTargetKind(ConfiguredTarget target) {
    return target.getTarget().getTargetKind();
  }

  @Override
  public String getLabel(ConfiguredTarget target) {
    return target.getLabel().toString();
  }

  @Override
  public String getPackage(ConfiguredTarget target) {
    return target.getTarget().getPackage().getNameFragment().toString();
  }

  @Override
  public boolean isRule(ConfiguredTarget target) {
    return target.getTarget() instanceof Rule;
  }

  @Override
  public boolean isTestRule(ConfiguredTarget target) {
    return TargetUtils.isTestRule(target.getTarget());
  }

  @Override
  public boolean isTestSuite(ConfiguredTarget target) {
    return TargetUtils.isTestSuiteRule(target.getTarget());
  }

  @Override
  public List<ConfiguredTarget> getLabelListAttr(
      QueryExpression caller,
      ConfiguredTarget configuredTarget,
      String attrName,
      String errorMsgPrefix)
      throws QueryException, InterruptedException {
    // TODO(bazel-team): implement this if needed.
    throw new UnsupportedOperationException();
  }

  @Override
  public List<String> getStringListAttr(ConfiguredTarget target, String attrName) {
    return TargetUtils.getStringListAttr(target.getTarget(), attrName);
  }

  @Override
  public String getStringAttr(ConfiguredTarget target, String attrName) {
    return TargetUtils.getStringAttr(target.getTarget(), attrName);
  }

  @Override
  public Iterable<String> getAttrAsString(ConfiguredTarget target, String attrName) {
    return TargetUtils.getAttrAsString(target.getTarget(), attrName);
  }

  @Override
  public Set<QueryVisibility<ConfiguredTarget>> getVisibility(ConfiguredTarget from)
      throws QueryException, InterruptedException {
    // TODO(bazel-team): implement this if needed.
    throw new UnsupportedOperationException();
  }
}
