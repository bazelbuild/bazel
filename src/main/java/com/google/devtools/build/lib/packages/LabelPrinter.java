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

package com.google.devtools.build.lib.packages;

import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import net.starlark.java.eval.Printer;
import net.starlark.java.eval.StarlarkSemantics;

/** Knows how to print labels consistently in various formats. */
public interface LabelPrinter {
  /**
   * Creates a {@link LabelPrinter} that prints labels in the same way as the Starlark `str` method.
   * This behavior is useful when matching labels against Starlark values, in particular in tools.
   *
   * <p>Do not use this method directly, call {@link
   * com.google.devtools.build.lib.query2.common.CommonQueryOptions#getLabelPrinter(StarlarkSemantics,
   * RepositoryMapping)} instead.
   */
  static LabelPrinter starlark(StarlarkSemantics starlarkSemantics) {
    return new LabelPrinter() {
      @Override
      public String toString(Label label) {
        Printer printer = new Printer();
        label.str(printer, starlarkSemantics);
        return printer.toString();
      }

      @Override
      public String toString(PackageIdentifier packageIdentifier) {
        // PackageIdentifier is not a StarlarkValue and thus doesn't have a str method. Since it is
        // only used in the context of --output=package, we reuse Label#str by stripping a
        // placeholder name.
        String label = toString(Label.createUnvalidated(packageIdentifier, "unused"));
        return label.substring(0, label.length() - ":unused".length());
      }
    };
  }

  /**
   * Creates a {@link LabelPrinter} that prints labels in a form meant for consumption by humans. It
   * the main repository has visibility into the label's repository, the apparent repository name is
   * used instead of the canonical repository name.
   *
   * <p>Do not use this method directly, call {@link
   * com.google.devtools.build.lib.query2.common.CommonQueryOptions#getLabelPrinter(StarlarkSemantics,
   * RepositoryMapping)} instead.
   */
  static LabelPrinter displayForm(RepositoryMapping mainRepoMapping) {
    return new LabelPrinter() {
      @Override
      public String toString(Label label) {
        return label.getDisplayForm(mainRepoMapping);
      }

      @Override
      public String toString(PackageIdentifier packageIdentifier) {
        return packageIdentifier.getDisplayForm(mainRepoMapping);
      }
    };
  }

  LabelPrinter LEGACY =
      new LabelPrinter() {
        @Override
        public String toString(Label label) {
          return label.toString();
        }

        @Override
        public String toString(PackageIdentifier packageIdentifier) {
          return packageIdentifier.toString();
        }
      };

  /**
   * Creates a {@link LabelPrinter} that prints labels via {@link Label#toString()}. This should
   * only be used for backwards compatibility in cases where exact label forms matter, such as for
   * genquery or in digests, or call sites outside of the query commands.
   */
  static LabelPrinter legacy() {
    return LEGACY;
  }

  /** Returns a string representation of the given label. */
  String toString(Label label);

  String toString(PackageIdentifier packageIdentifier);
}
