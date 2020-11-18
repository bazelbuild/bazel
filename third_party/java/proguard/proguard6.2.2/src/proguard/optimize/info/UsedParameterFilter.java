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
package proguard.optimize.info;

import proguard.classfile.*;
import proguard.classfile.visitor.*;

/**
 * This ParameterVisitor delegates all its visits to one of two other
 * ParameterVisitor instances, depending on whether the parameter is
 * used or not.
 *
 * @see ParameterUsageMarker
 * @author Eric Lafortune
 */
public class UsedParameterFilter
implements   ParameterVisitor
{
    private final ParameterVisitor usedParameterVisitor;
    private final ParameterVisitor unusedParameterVisitor;


    /**
     * Creates a new UsedParameterFilter that delegates visits to used
     * parameters to the given parameter visitor.
     */
    public UsedParameterFilter(ParameterVisitor usedParameterVisitor)
    {
        this(usedParameterVisitor, null);
    }


    /**
     * Creates a new UsedParameterFilter that delegates to one of the two
     * given parameter visitors.
     */
    public UsedParameterFilter(ParameterVisitor usedParameterVisitor,
                               ParameterVisitor unusedParameterVisitor)
    {
        this.usedParameterVisitor   = usedParameterVisitor;
        this.unusedParameterVisitor = unusedParameterVisitor;
    }


    // Implementations for ParameterVisitor.

    public void visitParameter(Clazz clazz, Member member, int parameterIndex, int parameterCount, int parameterOffset, int parameterSize, String parameterType, Clazz referencedClass)
    {
        ParameterVisitor parameterVisitor =
            ParameterUsageMarker.isParameterUsed((Method)member,
                                                 parameterOffset) ?
                usedParameterVisitor :
                unusedParameterVisitor;

        if (parameterVisitor != null)
        {
            parameterVisitor.visitParameter(clazz,
                                            member,
                                            parameterIndex,
                                            parameterCount,
                                            parameterOffset,
                                            parameterSize,
                                            parameterType,
                                            referencedClass);
        }
    }
}