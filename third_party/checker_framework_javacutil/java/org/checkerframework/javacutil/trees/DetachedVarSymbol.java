package org.checkerframework.javacutil.trees;

import com.sun.source.tree.VariableTree;
import com.sun.tools.javac.code.Symbol;
import com.sun.tools.javac.code.Type;
import com.sun.tools.javac.util.Name;

/*>>>
import org.checkerframework.checker.nullness.qual.Nullable;
*/

/**
 * A DetachedVarSymbol represents a variable that is not part of any
 * AST Tree.  DetachedVarSymbols are created when desugaring source
 * code constructs and they carry important type information, but some
 * methods such as TreeInfo.declarationFor do not work on them.
 */

public class DetachedVarSymbol extends Symbol.VarSymbol {

    protected /*@Nullable*/ VariableTree decl;

    /**
     * Construct a detached variable symbol, given its flags, name,
     * type and owner.
     */
    public DetachedVarSymbol(long flags, Name name, Type type, Symbol owner) {
        super(flags, name, type, owner);
        this.decl = null;
    }

    /**
     * Set the declaration tree for the variable.
     */
    public void setDeclaration(VariableTree decl) {
        this.decl = decl;
    }

    /**
     * Get the declaration tree for the variable.
     */
    public /*@Nullable*/ VariableTree getDeclaration() {
        return decl;
    }
}
