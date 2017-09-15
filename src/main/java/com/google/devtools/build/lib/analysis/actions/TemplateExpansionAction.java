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

package com.google.devtools.build.lib.analysis.actions;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Joiner;
import com.google.common.base.Objects;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.syntax.SkylarkDict;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.util.ResourceFileLoader;
import com.google.devtools.build.lib.util.StringUtilities;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.protobuf.ByteString;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.util.Collection;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.StreamSupport;

/**
 * Action to expand a template and write the expanded content to a file.
 */
@Immutable // if all substitutions are immutable
public final class TemplateExpansionAction extends AbstractFileWriteAction {

  private static final String GUID = "786c1fe0-dca8-407a-b108-e1ecd6d1bc7f";

  /**
   * A pair of a string to be substituted and a string to substitute it with.
   * For simplicity, these are called key and value. All implementations must
   * be immutable, and always return the identical key. The returned values
   * must be the same, though they need not be the same object.
   *
   * <p>It should be assumed that the {@link #getKey} invocation is cheap, and
   * that the {@link #getValue} invocation is expensive.
   */
  @Immutable // if the keys and values in the passed in lists and maps are all immutable
  public abstract static class Substitution {
    private Substitution() {
    }

    public abstract String getKey();
    public abstract String getValue();

    /**
     * Returns an immutable Substitution instance for the given key and value.
     */
    public static Substitution of(final String key, final String value) {
      return new Substitution() {
        @Override
        public String getKey() {
          return key;
        }

        @Override
        public String getValue() {
          return value;
        }
      };
    }

    /**
     * Returns an immutable Substitution instance for the key and list of values. The
     * values will be joined by spaces before substitution.
     */
    public static Substitution ofSpaceSeparatedList(
        final String key, final ImmutableList<?> value) {
      return new Substitution() {
        @Override
        public String getKey() {
          return key;
        }

        @Override
        public String getValue() {
          return Joiner.on(" ").join(value);
        }
      };
    }

    /**
     * Returns an immutable Substitution instance for the key and map of values.  Corresponding
     * values in the map will be joined with "=", and pairs will be joined by spaces before
     * substitution.
     *
     * <p>For example, the map <(a,1), (b,2), (c,3)> will become "a=1 b=2 c=3".
     */
    public static Substitution ofSpaceSeparatedMap(
        final String key, final ImmutableMap<?, ?> value) {
      return new Substitution() {
        @Override
        public String getKey() {
          return key;
        }

        @Override
        public String getValue() {
          return Joiner.on(" ").withKeyValueSeparator("=").join(value);
        }
      };
    }

    @Override
    public boolean equals(Object object) {
      if (this == object) {
        return true;
      }
      if (object instanceof Substitution) {
        Substitution substitution = (Substitution) object;
        return substitution.getKey().equals(this.getKey())
            && substitution.getValue().equals(this.getValue());
      }
      return false;
    }

    @Override
    public int hashCode() {
      return Objects.hashCode(getKey(), getValue());
    }

    @Override
    public String toString() {
      return "Substitution(" + getKey() + " -> " + getValue() + ")";
    }
  }

  /**
   * A substitution with a fixed key, and a computed value. The computed value
   * must not change over the lifetime of an instance, though the {@link
   * #getValue} method may return different String objects.
   *
   * <p>It should be assumed that the {@link #getKey} invocation is cheap, and
   * that the {@link #getValue} invocation is expensive.
   */
  public abstract static class ComputedSubstitution extends Substitution {
    private final String key;

    public ComputedSubstitution(String key) {
      this.key = key;
    }

    @Override
    public String getKey() {
      return key;
    }
  }

  /**
   * Expands a fragment value.
   *
   * <p>This is slighly more memory efficient since it defers the expansion of the path fragment's
   * string until requested. Often a template action is never executed, meaning the string is never
   * needed.
   */
  public static final class PathFragmentSubstitution extends ComputedSubstitution {
    private final PathFragment pathFragment;

    public PathFragmentSubstitution(String key, PathFragment pathFragment) {
      super(key);
      this.pathFragment = pathFragment;
    }

    @Override
    public String getValue() {
      return pathFragment.getPathString();
    }
  }

  /**
   * Expands a label value to its canonical string value.
   *
   * <p>This is more memory efficient than directly using the {@Label#toString}, since that method
   * constructs a new string every time it's called.
   */
  public static final class LabelSubstitution extends ComputedSubstitution {
    private final Label label;

    public LabelSubstitution(String key, Label label) {
      super(key);
      this.label = label;
    }

    @Override
    public String getValue() {
      return label.getCanonicalForm();
    }
  }

  /**
   * Expands a collection of artifacts to their short (root relative paths).
   *
   * <p>This is much more memory efficient than eagerly joining them into a string.
   */
  public static final class JoinedArtifactShortPathSubstitution extends ComputedSubstitution {
    private final Iterable<Artifact> artifacts;
    private final String joinStr;

    public JoinedArtifactShortPathSubstitution(
        String key, ImmutableList<Artifact> artifacts, String joinStr) {
      this(key, (Iterable<Artifact>) artifacts, joinStr);
    }

    public JoinedArtifactShortPathSubstitution(
        String key, NestedSet<Artifact> artifacts, String joinStr) {
      this(key, (Iterable<Artifact>) artifacts, joinStr);
    }

    private JoinedArtifactShortPathSubstitution(
        String key, Iterable<Artifact> artifacts, String joinStr) {
      super(key);
      this.artifacts = artifacts;
      this.joinStr = joinStr;
    }

    @Override
    public String getValue() {
      return StreamSupport.stream(artifacts.spliterator(), false)
          .map(artifact -> artifact.getRootRelativePath().getPathString())
          .collect(Collectors.joining(joinStr));
    }
  }

  /**
   * A template that contains text content, or alternatively throws an {@link
   * IOException}.
   */
  @Immutable // all subclasses are immutable
  public abstract static class Template {

    private static final Charset DEFAULT_CHARSET = StandardCharsets.UTF_8;

    /**
     * We only allow subclasses in this file.
     */
    private Template() {
    }

    /**
     * Returns the text content of the template.
     */
    protected abstract String getContent() throws IOException;

    /**
     * Returns a string that is used for the action key. This must change if
     * the getContent method returns something different, but is not allowed to
     * throw an exception.
     */
    protected abstract String getKey();

    /**
     * Loads a template from the given resource. The resource is looked up
     * relative to the given class. If the resource cannot be loaded, the returned
     * template throws an {@link IOException} when {@link #getContent} is
     * called. This makes it safe to use this method in a constant initializer.
     */
    public static Template forResource(final Class<?> relativeToClass, final String templateName) {
      try {
        String content = ResourceFileLoader.loadResource(relativeToClass, templateName);
        return forString(content);
      } catch (final IOException e) {
        return new Template() {
          @Override
          protected String getContent() throws IOException {
            throw new IOException("failed to load resource file '" + templateName
                + "' due to I/O error: " + e.getMessage(), e);
          }

          @Override
          protected String getKey() {
            return "ERROR: " + e.getMessage();
          }
        };
      }
    }

    /**
     * Returns a template for the given text string.
     */
    public static Template forString(final String templateText) {
      return new Template() {
        @Override
        protected String getContent() {
          return templateText;
        }

        @Override
        protected String getKey() {
          return templateText;
        }
      };
    }

    /**
     * Returns a template that loads the given artifact. It is important that
     * the artifact is also an input for the action, or this won't work.
     * Therefore this method is private, and you should use the corresponding
     * {@link TemplateExpansionAction} constructor.
     */
    private static Template forArtifact(final Artifact templateArtifact) {
      return new Template() {
        @Override
        protected String getContent() throws IOException {
          Path templatePath = templateArtifact.getPath();
          try {
            return FileSystemUtils.readContent(templatePath, DEFAULT_CHARSET);
          } catch (IOException e) {
            throw new IOException("failed to load template file '" + templatePath.getPathString()
                + "' due to I/O error: " + e.getMessage(), e);
          }
        }

        @Override
        protected String getKey() {
          // This isn't strictly necessary, because the action inputs are automatically considered.
          return "ARTIFACT: " + templateArtifact.getExecPathString();
        }
      };
    }
  }

  private final Template template;
  private final ImmutableList<Substitution> substitutions;

  /**
   * Creates a new TemplateExpansionAction instance.
   *
   * @param owner the action owner.
   * @param inputs the Artifacts that this Action depends on
   * @param output the Artifact that will be created by executing this Action.
   * @param template the template that will be expanded by this Action.
   * @param substitutions the substitutions that will be applied to the
   *   template. All substitutions will be applied in order.
   * @param makeExecutable iff true will change the output file to be
   *   executable.
   */
  private TemplateExpansionAction(ActionOwner owner,
                                  Collection<Artifact> inputs,
                                  Artifact output,
                                  Template template,
                                  List<Substitution> substitutions,
                                  boolean makeExecutable) {
    super(owner, inputs, output, makeExecutable);
    this.template = template;
    this.substitutions = ImmutableList.copyOf(substitutions);
  }

  /**
   * Creates a new TemplateExpansionAction instance for an artifact template.
   *
   * @param owner the action owner.
   * @param templateArtifact the Artifact that will be read as the text template
   *   file
   * @param output the Artifact that will be created by executing this Action.
   * @param substitutions the substitutions that will be applied to the
   *   template. All substitutions will be applied in order.
   * @param makeExecutable iff true will change the output file to be
   *   executable.
   */
  public TemplateExpansionAction(ActionOwner owner,
                                 Artifact templateArtifact,
                                 Artifact output,
                                 List<Substitution> substitutions,
                                 boolean makeExecutable) {
    this(owner, ImmutableList.of(templateArtifact), output, Template.forArtifact(templateArtifact),
        substitutions, makeExecutable);
  }

  /**
   * Creates a new TemplateExpansionAction instance without inputs.
   *
   * @param owner the action owner.
   * @param output the Artifact that will be created by executing this Action.
   * @param template the template
   * @param substitutions the substitutions that will be applied to the
   *   template. All substitutions will be applied in order.
   * @param makeExecutable iff true will change the output file to be
   *   executable.
   */
  public TemplateExpansionAction(ActionOwner owner,
                                 Artifact output,
                                 Template template,
                                 List<Substitution> substitutions,
                                 boolean makeExecutable) {
    this(owner, Artifact.NO_ARTIFACTS, output, template, substitutions, makeExecutable);
  }

  /**
   * Expands the template by applying all substitutions.
   * @param template
   * @return the expanded text.
   */
  private String expandTemplate(String template) {
    for (Substitution entry : substitutions) {
      template = StringUtilities.replaceAllLiteral(template, entry.getKey(), entry.getValue());
    }
    return template;
  }

  @VisibleForTesting
  public String getFileContents() throws IOException {
    return expandTemplate(template.getContent());
  }

  @Override
  public String getSkylarkContent() throws IOException {
    return getFileContents();
  }

  @Override
  public DeterministicWriter newDeterministicWriter(ActionExecutionContext ctx) throws IOException {
    return new ByteStringDeterministicWriter(
        ByteString.copyFrom(getFileContents().getBytes(Template.DEFAULT_CHARSET)));
  }

  @Override
  protected String computeKey() {
    Fingerprint f = new Fingerprint();
    f.addString(GUID);
    f.addString(String.valueOf(makeExecutable));
    f.addString(template.getKey());
    f.addInt(substitutions.size());
    for (Substitution entry : substitutions) {
      f.addString(entry.getKey());
      f.addString(entry.getValue());
    }
    return f.hexDigestAndReset();
  }

  @Override
  public String getMnemonic() {
    return "TemplateExpand";
  }

  @Override
  protected String getRawProgressMessage() {
    return "Expanding template " + Iterables.getOnlyElement(getOutputs()).prettyPrint();
  }

  public List<Substitution> getSubstitutions() {
    return substitutions;
  }

  @Override
  public SkylarkDict<String, String> getSkylarkSubstitutions() {
    ImmutableMap.Builder<String, String> builder = ImmutableMap.builder();
    for (Substitution entry : substitutions) {
      builder.put(entry.getKey(), entry.getValue());
    }
    return SkylarkDict.copyOf(null, builder.build());
  }
}
