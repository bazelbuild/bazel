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
import proguard.classfile.attribute.*;
import proguard.classfile.attribute.visitor.AttributeVisitor;
import proguard.classfile.instruction.Instruction;
import proguard.classfile.instruction.visitor.InstructionVisitor;
import proguard.classfile.util.*;
import proguard.classfile.visitor.ClassVisitor;
import proguard.evaluation.*;
import proguard.evaluation.value.*;
import proguard.optimize.evaluation.*;

/**
 * This AttributeVisitor marks the classes that are escaping from the visited
 * code attributes.
 *
 * @see ReferenceEscapeChecker
 * @author Eric Lafortune
 */
public class EscapingClassMarker
extends      SimplifiedVisitor
implements   AttributeVisitor,
             InstructionVisitor,
             ClassVisitor
{
    //*
    private static final boolean DEBUG = false;
    /*/
    private static       boolean DEBUG = System.getProperty("ecm") != null;
    //*/


    private final PartialEvaluator       partialEvaluator;
    private final boolean                runPartialEvaluator;
    private final ReferenceEscapeChecker referenceEscapeChecker;
    private final boolean                runReferenceEscapeChecker;


    /**
     * Creates a new EscapingClassMarker.
     */
    public EscapingClassMarker()
    {
        // We need typed references.
        this(new TypedReferenceValueFactory());
    }


    /**
     * Creates a new EscapingClassMarker.
     */
    public EscapingClassMarker(ValueFactory valueFactory)
    {
        this(valueFactory,
             new ReferenceTracingValueFactory(valueFactory));
    }


    /**
     * Creates a new EscapingClassMarker.
     */
    public EscapingClassMarker(ValueFactory                 valueFactory,
                               ReferenceTracingValueFactory tracingValueFactory)
    {
        this(new PartialEvaluator(tracingValueFactory,
                                  new ReferenceTracingInvocationUnit(new BasicInvocationUnit(tracingValueFactory)),
                                  true,
                                  tracingValueFactory),
             true);
    }


    /**
     * Creates a new EscapingClassMarker.
     */
    public EscapingClassMarker(PartialEvaluator partialEvaluator,
                               boolean          runPartialEvaluator)
    {
        this(partialEvaluator,
             runPartialEvaluator,
             new ReferenceEscapeChecker(partialEvaluator, false),
             true);
    }


    /**
     * Creates a new EscapingClassMarker.
     */
    public EscapingClassMarker(PartialEvaluator       partialEvaluator,
                               boolean                runPartialEvaluator,
                               ReferenceEscapeChecker referenceEscapeChecker,
                               boolean                runReferenceEscapeChecker)
    {
        this.partialEvaluator          = partialEvaluator;
        this.runPartialEvaluator       = runPartialEvaluator;
        this.referenceEscapeChecker    = referenceEscapeChecker;
        this.runReferenceEscapeChecker = runReferenceEscapeChecker;
    }


    // Implementations for AttributeVisitor.

    public void visitAnyAttribute(Clazz clazz, Attribute attribute) {}


    public void visitCodeAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute)
    {
        // Evaluate the code.
        if (runPartialEvaluator)
        {
            partialEvaluator.visitCodeAttribute(clazz, method, codeAttribute);
        }

        if (runReferenceEscapeChecker)
        {
            referenceEscapeChecker.visitCodeAttribute(clazz, method, codeAttribute);
        }

        // Mark all escaping classes.
        codeAttribute.instructionsAccept(clazz,
                                         method,
                                         partialEvaluator.tracedInstructionFilter(this));
    }


    // Implementations for InstructionVisitor.

    public void visitAnyInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, Instruction instruction)
    {
        // Does the instruction push a value that escapes?
        // We'll also count values that are returned, since they may be
        // downcast and the downcast type may escape in some calling
        // method.
        // TODO: Refine check: is a value is downcast to an escaping class, while it is being returned?
        if (instruction.stackPushCount(clazz) == 1 &&
            (referenceEscapeChecker.isInstanceEscaping(offset) ||
             referenceEscapeChecker.isInstanceReturned(offset)))
        {
            TracedStack stackAfter = partialEvaluator.getStackAfter(offset);
            Value       stackEntry = stackAfter.getTop(0);

            // Is it really a reference type?
            if (stackEntry.computationalType() == Value.TYPE_REFERENCE)
            {
                // Is it a plain class reference type?
                ReferenceValue referenceValue = stackEntry.referenceValue();
                if (referenceValue.isNull() != Value.ALWAYS &&
                    !ClassUtil.isInternalArrayType(referenceValue.getType()))
                {
                    // Do we know the class?
                    Clazz referencedClass = referenceValue.getReferencedClass();
                    if (referencedClass != null)
                    {
                        if (DEBUG)
                        {
                            System.out.println("EscapingClassMarker: ["+clazz.getName()+"."+method.getName(clazz)+method.getDescriptor(clazz)+"]: "+instruction.toString(offset)+" pushes escaping ["+referencedClass.getName()+"]");
                        }

                        // Mark it, along with its superclasses.
                        referencedClass.hierarchyAccept(true, true, true, false, this);
                    }
                }
            }
        }
    }



    // Implementations for ClassVisitor.

    public void visitLibraryClass(LibraryClass libraryClass) {}

    public void visitProgramClass(ProgramClass programClass)
    {
        markClassEscaping(programClass);
    }


    // Small utility methods.

    /**
     * Marks the given class as escaping.
     */
    private void markClassEscaping(Clazz clazz)
    {
        ClassOptimizationInfo info = ProgramClassOptimizationInfo.getClassOptimizationInfo(clazz);
        if (info instanceof ProgramClassOptimizationInfo)
        {
            ((ProgramClassOptimizationInfo)info).setEscaping();
        }
    }


    /**
     * Returns whether the given class is escaping.
     */
    public static boolean isClassEscaping(Clazz clazz)
    {
        return ClassOptimizationInfo.getClassOptimizationInfo(clazz).isEscaping();
    }
}