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
package proguard.evaluation;

import proguard.*;
import proguard.classfile.*;
import proguard.classfile.attribute.visitor.*;
import proguard.classfile.util.SimplifiedVisitor;
import proguard.classfile.visitor.*;
import proguard.evaluation.value.*;
import proguard.optimize.OptimizationInfoMemberFilter;
import proguard.optimize.info.*;

import java.util.*;

/**
 * This factory creates visitors to efficiently travel to specified classes and
 * class members and set specified values on them.
 *
 * @author Eric Lafortune
 */
public class AssumeClassSpecificationVisitorFactory
extends      ClassSpecificationVisitorFactory
{
    private final ValueFactory valueFactory;


    public AssumeClassSpecificationVisitorFactory(ValueFactory valueFactory)
    {
        this.valueFactory = valueFactory;
    }


    // Overriding implementations for ClassSpecificationVisitorFactory.

    protected ClassVisitor createNonTestingClassVisitor(MemberSpecification memberSpecification,
                                                        boolean             isField,
                                                        MemberVisitor       memberVisitor,
                                                        AttributeVisitor    attributeVisitor,
                                                        List                variableStringMatchers)
    {
        if (memberSpecification instanceof MemberValueSpecification)
        {
            // We can only know the value of this member specification at this
            // point.
            MemberValueSpecification memberValueSpecification =
                (MemberValueSpecification)memberSpecification;

            Number[] values = memberValueSpecification.values;
            if (values != null)
            {
                // Convert the Number array to a Value.
                Value value = value(values);

                // We're adding a member visitor to set the value.
                memberVisitor =
                    new MultiMemberVisitor(
                        memberVisitor,
                        new OptimizationInfoMemberFilter(
                        new MyMemberValueSetter(value)));
            }
        }

        return super.createNonTestingClassVisitor(memberSpecification,
                                                  isField,
                                                  memberVisitor,
                                                  attributeVisitor,
                                                  variableStringMatchers);
    }


    // Small utility methods.

    private Value value(Number[] values)
    {
        return values.length == 1 ?
            valueFactory.createIntegerValue(values[0].intValue()) :
            valueFactory.createIntegerValue(values[0].intValue(),
                                            values[1].intValue());
    }


    /**
     * This MemberVisitor sets a given value on the optimization info of the
     * members that it visits.
     */
    private static class MyMemberValueSetter
    implements           MemberVisitor
    {
        private final Value value;


        public MyMemberValueSetter(Value value)
        {
            this.value = value;
        }


        // Implementations for MemberVisitor.

        public void visitProgramField(ProgramClass programClass, ProgramField programField)
        {
            FieldOptimizationInfo
                .getFieldOptimizationInfo(programField)
                .setValue(value);
        }


        public void visitProgramMethod(ProgramClass programClass, ProgramMethod programMethod)
        {
            MethodOptimizationInfo
                .getMethodOptimizationInfo(programMethod)
                .setReturnValue(value);
        }


        public void visitLibraryField(LibraryClass libraryClass, LibraryField libraryField)
        {
            FieldOptimizationInfo
                .getFieldOptimizationInfo(libraryField)
                .setValue(value);
        }


        public void visitLibraryMethod(LibraryClass libraryClass, LibraryMethod libraryMethod)
        {
            MethodOptimizationInfo
                .getMethodOptimizationInfo(libraryMethod)
                .setReturnValue(value);
        }
    }
}
