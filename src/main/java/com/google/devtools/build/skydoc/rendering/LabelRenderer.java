// Copyright 2023 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.skydoc.rendering;

import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import java.util.Optional;
import net.starlark.java.eval.Printer;

/**
 * A wrapper around a Starlark value printer which prints the Starlark representation of a value
 * with any embedded {@link Label} values rendered in a form suitable for API documentation.
 *
 * <p>Labels are rendered via {@link label#getShorthandDisplayForm} with a provided repository
 * mapping, further adding an optional explicit repo name to labels in the main repo, and allowing
 * the {@code Label} constructor to be either included or omitted (rendering label objects as string
 * values).
 */
public final class LabelRenderer {
  private final RepositoryMapping repositoryMapping;
  private final Optional<String> mainRepoName;

  /** A LabelRenderer which always uses {@link Label#getShorthandDisplayForm} for rendering. */
  public static final LabelRenderer DEFAULT =
      new LabelRenderer(RepositoryMapping.ALWAYS_FALLBACK, Optional.empty());

  @SuppressWarnings("NonApiType") // for convenience of use by StarlarkDocExtract
  public LabelRenderer(RepositoryMapping repositoryMapping, Optional<String> mainRepoName) {
    this.repositoryMapping = repositoryMapping;
    this.mainRepoName = mainRepoName;
  }

  /**
   * Renders a label as an unquoted string via {@link Label#getShorthandDisplayForm}, further adding
   * an explicit repo component for labels in the main repo if {@code mainRepoName} was provided.
   */
  public String render(Label label) {
    return render(label, /* shorthand= */ true);
  }

  // This method could be public, but there are currently no users outside this class who would want
  // to call it with shorthand = false.
  private String render(Label label, boolean shorthand) {
    String labelString =
        shorthand
            ? label.getShorthandDisplayForm(repositoryMapping)
            : label.getDisplayForm(repositoryMapping);
    if (mainRepoName.isEmpty() || labelString.startsWith("@")) {
      return labelString;
    } else {
      // label.getShorthandDisplayForm omits the repo name part for labels in the main repo
      // regardless of what repositoryMapping says. Therefore, if we want to rename the main repo
      // in labels in emitted docs, we have to do so manually.
      if (shorthand && labelString.equals("//:" + mainRepoName.get())) {
        // Special case: the shorthand form of "@foo//:foo" is "@foo".
        return "@" + mainRepoName.get();
      }
      return String.format("@%s%s", mainRepoName.get(), labelString);
    }
  }

  /**
   * Renders the {@code repr()} of a Starlark value as a string, with any embedded label values
   * first converted to string values via {@link #render}.
   */
  public String reprWithoutLabelConstructor(Object o) {
    return new Printer() {
      @Override
      public Printer repr(Object o) {
        if (o instanceof Label) {
          return repr(render((Label) o));
        } else {
          return super.repr(o);
        }
      }
    }.repr(o).toString();
  }

  /**
   * Renders the {@code repr()} of a Starlark value as a string, with the argument to the {@code
   * Label} constructor for any embedded label values produced via {@link Label#getDisplayForm},
   * further adding an explicit repo component for labels in the main repo if {@code mainRepoName}
   * was provided.
   *
   * <p>Invariant: if the label's repo is mapped by {@code repositoryMapping}, or if {@code
   * repositoryMapping} allows fallback, then {@code this.repr(label)} equals {@code
   * Starlark.repr(Label.parseCanonicalUnchecked(this.render(label)))}.
   */
  public String repr(Object o) {
    return new Printer() {
      @Override
      public Printer repr(Object o) {
        if (o instanceof Label) {
          return append("Label(")
              // For consistency with Starlark.repr(label), we use label.getDisplayForm() instead of
              // the shorthand form.
              .repr(render((Label) o, /* shorthand= */ false))
              .append(")");
        } else {
          return super.repr(o);
        }
      }
    }.repr(o).toString();
  }
}
