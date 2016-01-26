// Copyright 2014 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.util.Preconditions;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;

import javax.annotation.Nullable;

/**
 * Checked exception for analysis-time errors, which can store the errors for later reporting.
 *
 * <p>It's more robust for a method to throw this exception than expecting a
 * {@link RuleErrorConsumer} object (which may be null).
 */
public final class AnalysisIssues extends Exception {

  /**
   * An error entry.
   *
   * <p>{@link AnalysisIssues} can accumulate multiple of these, and report all of them at once.
   */
  public static final class Entry {
    private final String attribute;
    private final String messageTemplate;
    private final Object[] arguments;

    private Entry(@Nullable String attribute, String messageTemplate, Object... arguments) {
      this.attribute = attribute;
      this.messageTemplate = messageTemplate;
      this.arguments = arguments;
    }

    private void reportTo(RuleErrorConsumer errors) {
      String msg = String.format(messageTemplate, arguments);
      if (attribute == null) {
        errors.ruleError(msg);
      } else {
        errors.attributeError(attribute, msg);
      }
    }

    private void reportTo(StringBuilder sb) {
      String msg = String.format(messageTemplate, arguments);
      if (attribute == null) {
        sb.append("ERROR: ").append(msg);
      } else {
        sb.append("ERROR: in attribute \"").append(attribute).append("\": ").append(msg);
      }
    }

    @Override
    public String toString() {
      if (attribute == null) {
        return String.format("ERROR: " + messageTemplate, arguments);
      } else {
        List<Object> args = new ArrayList<>();
        args.add(attribute);
        Collections.addAll(args, arguments);
        return String.format("ERROR in '%s': " + messageTemplate, args.toArray());
      }
    }
  }

  private final ImmutableList<Entry> entries;

  public AnalysisIssues(Entry entry) {
    this.entries = ImmutableList.of(Preconditions.checkNotNull(entry));
  }

  public AnalysisIssues(Collection<Entry> entries) {
    this.entries = ImmutableList.copyOf(Preconditions.checkNotNull(entries));
  }

  /**
   * Creates a attribute error entry that will be added to a {@link AnalysisIssues} later.
   */
  public static Entry attributeError(String attribute, String messageTemplate,
      Object... arguments) {
    return new Entry(attribute, messageTemplate, arguments);
  }

  public static Entry ruleError(String messageTemplate, Object... arguments) {
    return new Entry(null, messageTemplate, arguments);
  }

  /**
   * Report all accumulated errors and warnings to the given consumer object.
   */
  public void reportTo(RuleErrorConsumer errors) {
    Preconditions.checkNotNull(errors);
    for (Entry e : entries) {
      e.reportTo(errors);
    }
  }

  @Nullable
  private String asString() {
    if (entries == null) {
      return null;
    }

    StringBuilder sb = new StringBuilder();
    for (Entry e : entries) {
      e.reportTo(sb);
    }
    return sb.toString();
  }

  @Override
  public String getMessage() {
    return asString();
  }

  @Override
  public String toString() {
    String s = asString();
    return s == null ? "" : s;
  }
}
