/*
 * ProGuard -- shrinking, optimization, obfuscation, and preverification
 *             of Java bytecode.
 *
 * Copyright (c) 2002-2017 Eric Lafortune @ GuardSquare
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
package proguard.optimize.evaluation;

import proguard.classfile.*;
import proguard.classfile.util.SimplifiedVisitor;
import proguard.classfile.visitor.*;
import proguard.evaluation.value.*;
import proguard.optimize.info.*;

/**
 * This ClassVisitor propagates the value of the $VALUES field to the values()
 * method in the simple enum classes that it visits.
 *
 * @see SimpleEnumMarker
 * @author Eric Lafortune
 */
public class SimpleEnumArrayPropagator
extends      SimplifiedVisitor
implements   ClassVisitor,
             MemberVisitor
{
    private final ValueFactory valueFactory = new ParticularValueFactory();

    private Value array;


    // Implementations for ClassVisitor.

    public void visitProgramClass(ProgramClass programClass)
    {
        // Update the return value of the "int[] values()" method.
        programClass.methodsAccept(new MemberDescriptorFilter("()[I", this));
    }


    // Implementations for MemberVisitor.

    public void visitProgramMethod(ProgramClass programClass, ProgramMethod programMethod)
    {
        // Find the array length of the "int[] $VALUES" field.
        programClass.fieldsAccept(new MemberDescriptorFilter("[I", this));

        if (array != null)
        {
            // Set the array value with the found array length. We can't use
            // the original array, because its elements might get overwritten.
            Value propagatedArray =
                valueFactory.createArrayReferenceValue("I",
                                                       null,
                                                       array.referenceValue().arrayLength(
                                                           valueFactory));

            setMethodReturnValue(programMethod, propagatedArray);

            array = null;
        }
    }

    public void visitProgramField(ProgramClass programClass, ProgramField programField)
    {
        array = StoringInvocationUnit.getFieldValue(programField);
    }


    // Small utility methods.

    private static void setMethodReturnValue(Method method, Value value)
    {
        MethodOptimizationInfo info = MethodOptimizationInfo.getMethodOptimizationInfo(method);
        if (info != null)
        {
            info.setReturnValue(value);
        }
    }
}
