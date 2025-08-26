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
package com.google.devtools.build.lib.skyframe;

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.devtools.build.lib.cmdline.Label.labelCodec;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.MoreObjects;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableTable;
import com.google.devtools.build.lib.cmdline.BazelModuleKey;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.BzlVisibility;
import com.google.devtools.build.lib.skyframe.serialization.LeafDeserializationContext;
import com.google.devtools.build.lib.skyframe.serialization.LeafObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.LeafSerializationContext;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.devtools.build.lib.skyframe.serialization.VisibleForSerialization;
import com.google.devtools.build.lib.util.HashCodes;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyKey.SkyKeyInterner;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.errorprone.annotations.Keep;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import net.starlark.java.eval.Module;

/**
 * A value that represents the .bzl (or .scl) module loaded by a Starlark {@code load()} statement.
 *
 * <p>Note: Historically, all modules had the .bzl suffix, but this is no longer true now that Bazel
 * supports the .scl dialect. In identifiers, code comments, and documentation, you should generally
 * assume any "bzl" term could mean a .scl file as well.
 *
 * <p>The key consists of an absolute {@link Label} and the context in which the load occurs. The
 * Label should not reference the special {@code external} package.
 *
 * <p>This value is also used to represent the special prelude file that may be implicitly loaded
 * and sourced by BUILD files. The prelude file need not end in ".bzl".
 */
public class BzlLoadValue implements SkyValue {

  private final Module module; // .bzl module (and indirectly, the entire load DAG)
  // TODO(brandjon): Is this field redundant with BazelModuleContext#bzlTransitiveDigest, accessible
  // from the Module as client data?
  private final byte[] transitiveDigest; // of .bzl file and load dependencies
  private final BzlVisibility bzlVisibility;
  private final ImmutableTable<RepositoryName, String, RepositoryName> recordedRepoMappings;

  @VisibleForTesting
  public BzlLoadValue(
      Module module,
      byte[] transitiveDigest,
      BzlVisibility bzlVisibility,
      ImmutableTable<RepositoryName, String, RepositoryName> recordedRepoMappings) {
    this.module = checkNotNull(module);
    this.transitiveDigest = checkNotNull(transitiveDigest);
    this.bzlVisibility = checkNotNull(bzlVisibility);
    this.recordedRepoMappings = checkNotNull(recordedRepoMappings);
  }

  /** Returns the .bzl module. */
  public Module getModule() {
    return module;
  }

  /** Returns the digest of the .bzl module and its transitive load dependencies. */
  public byte[] getTransitiveDigest() {
    return transitiveDigest;
  }

  /** Returns the visibility of this module for the purpose of {@code load()} statements. */
  public BzlVisibility getBzlVisibility() {
    return bzlVisibility;
  }

  /**
   * Returns the repo mapping entries used to laod this bzl file. Stored for correctness across
   * Bazel server restarts.
   */
  public ImmutableTable<RepositoryName, String, RepositoryName> getRecordedRepoMappings() {
    return recordedRepoMappings;
  }

  private static final SkyKeyInterner<Key> keyInterner = SkyKey.newInterner();

  private abstract static sealed class KeyForLocalEval extends Key
      permits KeyForBuild, KeyForBuiltins {}

  /** SkyKey for a Starlark load. */
  public abstract static sealed class Key implements BazelModuleKey
      permits KeyForLocalEval, KeyForBzlmod {
    // Closed, for class-based equals()/hashCode().
    private Key() {}

    /**
     * Returns the absolute label of the .bzl file to be loaded.
     *
     * <p>For {@link KeyForBuiltins}, it must begin with {@code @_builtins//:}. (It is legal for
     * other keys to use {@code @_builtins}, but since no real repo by that name may be defined,
     * they won't evaluate to a successful result.)
     */
    @Override
    public abstract Label getLabel();

    /** Returns true if this is a request for the special BUILD prelude file. */
    boolean isBuildPrelude() {
      return false;
    }

    /** Returns true if this is a request for a builtins bzl file. */
    boolean isBuiltins() {
      return false;
    }

    /** Returns true if the requested file follows the .scl dialect. */
    // Note: Just as with .bzl, the same .scl file can be referred to from multiple key types, for
    // instance if a BUILD file and a module rule both load foo.scl. Conceptually, .scl files
    // shouldn't depend on what kind of top-level file caused them to load, but in practice, this
    // implementation quirk means that the .scl file will be loaded twice as separate copies.
    //
    // This shouldn't matter except in rare edge cases, such as if a Starlark function is loaded
    // from both copies and compared for equality. Performance wise, it also means that all
    // transitive .scl files will be double-loaded, but we don't expect that to be significant.
    //
    // The alternative is to use a separate key type just for .scl, but that complicates repo logic;
    // see BzlLoadFunction#getRepositoryMapping.
    final boolean isSclDialect() {
      return getLabel().getName().endsWith(".scl");
    }

    /**
     * Constructs a new key suitable for evaluating a {@code load()} dependency of this key's .bzl
     * file.
     *
     * <p>The new key uses the given label but the same contextual information -- whether the
     * top-level requesting value is a BUILD or WORKSPACE file, and if it's a WORKSPACE, its
     * chunking info.
     */
    abstract Key getKeyForLoad(Label loadLabel);

    /**
     * Constructs a BzlCompileValue key suitable for retrieving the Starlark code for this .bzl,
     * given the Root in which to find its file.
     */
    abstract BzlCompileValue.Key getCompileKey(Root root);

    @Override
    public final boolean valueIsShareable() {
      // We don't guarantee that all constructs implement equality, meaning we can't correctly
      // compare deserialized instances. This is currently the case for attribute descriptors.
      return false;
    }

    @SuppressWarnings("EqualsGetClass") // All subclasses are known.
    @Override
    public boolean equals(Object obj) {
      if (this == obj) {
        return true;
      }
      if (obj == null) {
        return false;
      }
      if (!this.getClass().equals(obj.getClass())) {
        return false;
      }
      Key that = (Key) obj;
      return this.getLabel().equals(that.getLabel())
          && (this.isBuildPrelude() == that.isBuildPrelude())
          && (this.isBuiltins() == that.isBuiltins());
    }

    @Override
    public int hashCode() {
      int result = HashCodes.hashObjects(getClass(), getLabel());
      result = 31 * result + Boolean.hashCode(isBuildPrelude());
      result = 31 * result + Boolean.hashCode(isBuiltins());
      return result;
    }

    protected final MoreObjects.ToStringHelper toStringHelper() {
      return MoreObjects.toStringHelper(this)
          .add("label", getLabel())
          .add("isBuildPrelude", isBuildPrelude());
    }

    @Override
    public String toString() {
      return toStringHelper().toString();
    }

    @Override
    public SkyKeyInterner<Key> getSkyKeyInterner() {
      return keyInterner;
    }
  }

  /** A key for loading a .bzl during package loading (BUILD evaluation). */
  @Immutable
  @VisibleForSerialization
  static final class KeyForBuild extends KeyForLocalEval {
    private final Label label;

    /**
     * True if this is the special prelude file, whose declarations are implicitly loaded by all
     * BUILD files.
     */
    private final boolean isBuildPrelude;

    private KeyForBuild(Label label, boolean isBuildPrelude) {
      this.label = checkNotNull(label);
      this.isBuildPrelude = isBuildPrelude;
    }

    @Override
    public Label getLabel() {
      return label;
    }

    @Override
    boolean isBuildPrelude() {
      return isBuildPrelude;
    }

    @Override
    Key getKeyForLoad(Label loadLabel) {
      // Note that the returned key always has !isBuildPrelude. I.e., if the prelude file loads
      // another .bzl, the loaded .bzl is processed as normal with no special prelude magic. This is
      // because 1) only the prelude file, not its dependencies, should automatically re-export its
      // loaded symbols; and 2) we don't want prelude-loaded modules to end up cloned if they're
      // also loaded through normal means.
      return keyForBuild(loadLabel);
    }

    @Override
    BzlCompileValue.Key getCompileKey(Root root) {
      if (isBuildPrelude) {
        return BzlCompileValue.keyForBuildPrelude(root, label);
      } else {
        return BzlCompileValue.key(root, label);
      }
    }
  }

  /**
   * A key for loading a .bzl during {@code @_builtins} evaluation.
   *
   * <p>This kind of key is only requested by {@link StarlarkBuiltinsFunction} and its transitively
   * loaded {@link BzlLoadFunction} calls.
   *
   * <p>The label must have {@link RepositoryName#BUILTINS} as its repository component. (It is
   * valid for other key types to use that repo name, but since it is not a real repository and
   * cannot be fetched, any attempt to resolve such a key would fail.)
   */
  @Immutable
  @VisibleForSerialization
  static final class KeyForBuiltins extends KeyForLocalEval {
    private final Label label;

    private KeyForBuiltins(Label label) {
      this.label = checkNotNull(label);
      if (!StarlarkBuiltinsValue.isBuiltinsRepo(label.getRepository())) {
        throw new IllegalArgumentException("repository name for builtins key must be '@_builtins'");
      }
    }

    @Override
    public Label getLabel() {
      return label;
    }

    @Override
    boolean isBuiltins() {
      return true;
    }

    @Override
    Key getKeyForLoad(Label label) {
      return keyForBuiltins(label);
    }

    @Override
    BzlCompileValue.Key getCompileKey(Root root) {
      return BzlCompileValue.keyForBuiltins(root, label);
    }
  }

  /** A key for loading a .bzl to get the repo rule required by Bzlmod generated repositories. */
  @Immutable
  @VisibleForSerialization
  static sealed class KeyForBzlmod extends Key permits KeyForBzlmodBootstrap {
    private final Label label;

    private KeyForBzlmod(Label label) {
      this.label = checkNotNull(label);
    }

    @Override
    public Label getLabel() {
      return label;
    }

    @Override
    Key getKeyForLoad(Label loadLabel) {
      return keyForBzlmod(loadLabel);
    }

    @Override
    BzlCompileValue.Key getCompileKey(Root root) {
      return BzlCompileValue.key(root, label);
    }
  }

  @Immutable
  @VisibleForSerialization
  static final class KeyForBzlmodBootstrap extends KeyForBzlmod {
    private KeyForBzlmodBootstrap(Label label) {
      super(label);
    }

    @Override
    Key getKeyForLoad(Label loadLabel) {
      return keyForBzlmodBootstrap(loadLabel);
    }
  }

  /** Constructs a key for loading a regular .bzl file from BUILD files. */
  public static Key keyForBuild(Label label) {
    return keyInterner.intern(new KeyForBuild(label, /* isBuildPrelude= */ false));
  }

  /** Constructs a key for loading a .bzl file within the {@code @_builtins} pseudo-repository. */
  public static Key keyForBuiltins(Label label) {
    return keyInterner.intern(new KeyForBuiltins(label));
  }

  /** Constructs a key for loading the special prelude .bzl. */
  static Key keyForBuildPrelude(Label label) {
    return keyInterner.intern(new KeyForBuild(label, /* isBuildPrelude= */ true));
  }

  /** Constructs a key for loading a .bzl for Bzlmod repos */
  public static Key keyForBzlmod(Label label) {
    return keyInterner.intern(new KeyForBzlmod(label));
  }

  public static Key keyForBzlmodBootstrap(Label label) {
    Preconditions.checkArgument(
        label.getRepository().equals(RepositoryName.BAZEL_TOOLS),
        "keyForBzlmodBootstrap must be called with a label in the bazel_tools repository");
    return keyInterner.intern(new KeyForBzlmodBootstrap(label));
  }

  public static KeyCodec bzlLoadKeyCodec() {
    return KeyCodec.INSTANCE;
  }

  @Keep
  private static final class KeyCodec extends LeafObjectCodec<Key> {
    private static final KeyCodec INSTANCE = new KeyCodec();

    @Override
    public Class<KeyForLocalEval> getEncodedClass() {
      return KeyForLocalEval.class;
    }

    @Override
    public void serialize(LeafSerializationContext context, Key obj, CodedOutputStream codedOut)
        throws SerializationException, IOException {
      context.serializeLeaf(obj.getLabel(), labelCodec(), codedOut);

      switch (obj) {
        case KeyForBuild forBuild -> {
          codedOut.writeInt32NoTag(0);
          codedOut.writeBoolNoTag(forBuild.isBuildPrelude());
        }
        case KeyForBuiltins forBuiltins -> {
          codedOut.writeInt32NoTag(1);
        }
        case KeyForBzlmodBootstrap keyForBzlmodBootstrap -> {
          codedOut.writeInt32NoTag(2);
        }
        case KeyForBzlmod keyForBzlmod -> {
          codedOut.writeInt32NoTag(3);
        }
      }
    }

    @Override
    public Key deserialize(LeafDeserializationContext context, CodedInputStream codedIn)
        throws SerializationException, IOException {
      Label label = context.deserializeLeaf(codedIn, labelCodec());
      int discriminant = codedIn.readInt32();
      return switch (discriminant) {
        case 0 -> codedIn.readBool() ? keyForBuildPrelude(label) : keyForBuild(label);
        case 1 -> keyForBuiltins(label);
        case 2 -> keyForBzlmodBootstrap(label);
        case 3 -> keyForBzlmod(label);
        default -> {
          throw new SerializationException("unexpected discriminant: " + discriminant);
        }
      };
    }
  }
}
