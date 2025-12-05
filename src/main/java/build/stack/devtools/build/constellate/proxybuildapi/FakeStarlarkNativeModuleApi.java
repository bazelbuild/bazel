package build.stack.devtools.build.constellate.fakebuildapi;

import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.starlarkbuildapi.StarlarkNativeModuleApi;
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
  public String packageName(StarlarkThread thread) {
    return "";
  }

  @Override
  public String repositoryName(StarlarkThread thread) {
    return "";
  }

  @Nullable
  @Override
  public Object getValue(String name) throws EvalException {
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
