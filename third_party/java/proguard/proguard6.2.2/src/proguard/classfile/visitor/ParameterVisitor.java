/*
 * ProGuard -- shrinking, optimization, obfuscation, and preverification
 *             of Java bytecode.
 *
 * Copyright (c) 2002-2019 Guardsquare NV
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 2 of the License, or (at your option)
 * any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 */
package proguard.classfile.visitor;

import proguard.classfile.*;

/**
 * This interface specifies the methods for a visitor of method parameters or
 * field types (which can be considered parameters when storing values). The
 * parameters do not include or count the 'this' parameter or the method return
 * value.
 *
 * @author Eric Lafortune
 */
public interface ParameterVisitor
{
    /**
     * Visits the given parameter.
     * @param clazz           the class of the method.
     * @param member          the field or method of the parameter.
     * @param parameterIndex  the index of the parameter.
     * @param parameterCount  the total number of parameters.
     * @param parameterOffset the offset of the parameter, accounting for
     *                        longs and doubles taking up two entries.
     * @param parameterSize   the total size of the parameters, accounting for
     *                        longs and doubles taking up two entries.
     * @param parameterType   the parameter type.
     * @param referencedClass the class contained in the parameter type, if any.
     */
    public void visitParameter(Clazz  clazz,
                               Member member,
                               int    parameterIndex,
                               int    parameterCount,
                               int    parameterOffset,
                               int    parameterSize,
                               String parameterType,
                               Clazz  referencedClass);
}
