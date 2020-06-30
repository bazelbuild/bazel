package com.google.devtools.build.lib.syntax;

import java.util.Objects;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.common.collect.Interner;

/** Ultra-fast call signature (which is a call without star or star-star args.
 * Starlark can optimize these calls. */
public class UltraFastCallSig {
    /** Number of positional arguments. */
    public final int numPositional;
    /** Named arguments. */
    public final ImmutableList<String> named;
    public final boolean hasNamed;

    private final int hashCode;

    private UltraFastCallSig(int numPositional, ImmutableList<String> named) {
        this.numPositional = numPositional;
        this.named = named;
        this.hasNamed = !named.isEmpty();
        this.hashCode = Objects.hash(numPositional, named);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        UltraFastCallSig that = (UltraFastCallSig) o;
        return numPositional == that.numPositional
                && named.equals(that.named);
    }

    @Override
    public int hashCode() {
        // Cache hash-code, because map lookup by signature must be lightning fast
        return hashCode;
    }

    private static final Interner<UltraFastCallSig> interner = BlazeInterners.newWeakInterner();

    public static UltraFastCallSig create(int numPositional, ImmutableList<String> named) {
        // We need to intern call signatures because cache lookup should be very fast,
        // and deep `UltraFastCallSig` comparison is not.
        return interner.intern(new UltraFastCallSig(numPositional, named));
    }
}
