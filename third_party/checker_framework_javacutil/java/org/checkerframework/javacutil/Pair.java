package org.checkerframework.javacutil;

/*>>>
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
*/

/**
 * Simple pair class for multiple returns.
 *
 * TODO: as class is immutable, use @Covariant annotation.
 */
public class Pair<V1, V2> {
    public final V1 first;
    public final V2 second;

    private Pair(V1 v1, V2 v2) {
        this.first = v1;
        this.second = v2;
    }

    public static <V1, V2> Pair<V1, V2> of(V1 v1, V2 v2) {
        return new Pair<V1, V2>(v1, v2);
    }

    /*@SideEffectFree*/
    @Override
    public String toString() {
        return "Pair(" + first + ", " + second + ")";
    }

    private int hashCode = -1;

    /*@Pure*/
    @Override
    public int hashCode() {
        if (hashCode == -1) {
            hashCode = 31;
            if (first != null) {
                hashCode += 17 * first.hashCode();
            }
            if (second != null) {
                hashCode += 17 * second.hashCode();
            }
        }
        return hashCode;
    }

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof Pair)) {
            return false;
        }
        @SuppressWarnings("unchecked")
        Pair<V1, V2> other = (Pair<V1, V2>) o;
        if (this.first == null) {
            if (other.first != null) return false;
        } else {
            if (!this.first.equals(other.first)) return false;
        }
        if (this.second == null) {
            if (other.second != null) return false;
        } else {
            if (!this.second.equals(other.second)) return false;
        }

        return true;
    }
}
