package build.stack.devtools.build.constellate.fakebuildapi;

import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.starlarkbuildapi.StarlarkNativeModuleApi;
import com.google.devtools.build.lib.starlarkdocextract.StardocOutputProtos.RuleInfo;
import java.util.Collections;
import java.util.Map;
import javax.annotation.Nullable;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.NoneType;
import net.starlark.java.eval.Printer;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkCallable;
import net.starlark.java.eval.StarlarkInt;
import net.starlark.java.eval.StarlarkList;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.eval.Structure;
import net.starlark.java.syntax.Location;

/** Fake implementation of {@link StarlarkNativeModuleApi}. */
public class FakeStarlarkNativeModuleApi implements StarlarkNativeModuleApi, Structure {

  private final Map<String, RuleInfo> nativeRules;

  public FakeStarlarkNativeModuleApi(Map<String, RuleInfo> nativeRules) {
    this.nativeRules = nativeRules != null ? nativeRules : Collections.emptyMap();
  }

  // Default constructor for backward compatibility
  public FakeStarlarkNativeModuleApi() {
    this(Collections.emptyMap());
  }

  @Override
  public Sequence<?> glob(
      Sequence<?> include,
      Sequence<?> exclude,
      StarlarkInt excludeDirectories,
      Object allowEmpty,
      StarlarkThread thread) {
    return StarlarkList.of(thread.mutability());
  }

  @Override
  public Object existingRule(String name, StarlarkThread thread) {
    return null;
  }

  @Override
  public Sequence<?> subpackages(
      Sequence<?> include,
      Sequence<?> exclude,
      boolean allowEmpty,
      StarlarkThread thread) throws EvalException {
    return StarlarkList.empty();
  }

  @Override
  public Dict<String, Dict<String, Object>> existingRules(StarlarkThread thread)
      throws EvalException {
    return Dict.of(thread.mutability());
  }

  @Override
  public NoneType packageGroup(
      String name, Sequence<?> packages, Sequence<?> includes, StarlarkThread thread) {
    return null;
  }

  @Override
  public NoneType exportsFiles(
      Sequence<?> srcs, Object visibility, Object licenses, StarlarkThread thread) {
    return null;
  }

  @Override
  public String moduleVersion(StarlarkThread thread) throws EvalException {
    // Stub implementation - return empty version
    return "";
  }

  @Override
  public String moduleName(StarlarkThread thread) throws EvalException {
    // Stub implementation - return empty module name
    return "";
  }

  @Override
  public com.google.devtools.build.lib.cmdline.Label packageRelativeLabel(Object input, StarlarkThread thread) throws EvalException {
    // Stub implementation - try to parse as label or return a dummy label
    if (input instanceof com.google.devtools.build.lib.cmdline.Label) {
      return (com.google.devtools.build.lib.cmdline.Label) input;
    }
    try {
      return com.google.devtools.build.lib.cmdline.Label.parseCanonical(input.toString());
    } catch (Exception e) {
      throw new EvalException("Invalid label: " + input);
    }
  }

  @Override
  public String packageName(StarlarkThread thread) {
    return "";
  }

  @Override
  public String repositoryName(StarlarkThread thread) {
    return "";
  }

  @Override
  public String repoName(StarlarkThread thread) throws EvalException {
    // Stub implementation - return empty repo name
    return "";
  }

  @Override
  public java.util.List<com.google.devtools.build.lib.cmdline.Label> packageDefaultVisibility(StarlarkThread thread) throws EvalException {
    // Stub implementation - return empty visibility list
    return java.util.Collections.emptyList();
  }

  @Nullable
  @Override
  public Object getValue(String name) throws EvalException {
    // Check if this is a known native rule
    final RuleInfo ruleInfo = nativeRules.get(name);

    if (ruleInfo != null) {
      // Return a callable that represents the native rule
      return new StarlarkCallable() {
        @Override
        public Object fastcall(StarlarkThread thread, Object[] positional, Object[] named) {
          return Starlark.NONE;
        }

        @Override
        public String getName() {
          return name;
        }

        @Override
        public Location getLocation() {
          return Location.BUILTIN;
        }

        @Override
        public void repr(Printer printer) {
          printer.append("<native." + name + ">");
        }
      };
    }

    // Bazel's notion of the global "native" isn't fully exposed via public interfaces, for example,
    // as far as native rules are concerned. Returning None on all unsupported invocations of
    // native.[func_name]() is the safest "best effort" approach to implementing a fake for
    // "native".
    return new StarlarkCallable() {
      @Override
      public Object fastcall(StarlarkThread thread, Object[] positional, Object[] named) {
        return Starlark.NONE;
      }

      @Override
      public String getName() {
        return name;
      }

      @Override
      public Location getLocation() {
        return Location.BUILTIN;
      }

      @Override
      public void repr(Printer printer) {
        printer.append("<faked no-op function " + name + ">");
      }
    };
  }

  @Override
  public ImmutableCollection<String> getFieldNames() {
    return ImmutableList.of();
  }

  @Nullable
  @Override
  public String getErrorMessageForUnknownField(String field) {
    return "";
  }
}
