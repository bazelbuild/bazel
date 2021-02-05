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

import proguard.classfile.ClassPool;
import proguard.classfile.visitor.ClassPoolVisitor;

/**
 * This ClassPoolVisitor repeatedly delegates to a given class pool visitor, as
 * long as it keeps setting a given flag.
 *
 * @author Eric Lafortune
 */
public class RepeatedClassPoolVisitor
implements   ClassPoolVisitor
{
    //*
    private static final boolean DEBUG = false;
    /*/
    private static       boolean DEBUG = System.getProperty("rcpv") != null;
    //*/


    private final MutableBoolean   repeatTrigger;
    private final ClassPoolVisitor classPoolVisitor;


    /**
     * Creates a new RepeatedClassPoolVisitor.
     * @param repeatTrigger    the mutable boolean flag that the class pool
     *                         visitor can set to indicate that the class pool
     *                         should be visited again.
     * @param classPoolVisitor the class pool visitor to apply.
     */
    public RepeatedClassPoolVisitor(MutableBoolean   repeatTrigger,
                                    ClassPoolVisitor classPoolVisitor)
    {
        this.repeatTrigger    = repeatTrigger;
        this.classPoolVisitor = classPoolVisitor;
    }


    // Implementations for ClassPoolVisitor.

    public void visitClassPool(ClassPool classPool)
    {
        // Visit all classes at least once, until the class visitors stop
        // setting the repeat trigger.
        do
        {
            if (DEBUG)
            {
                System.out.println("RepeatedClassPoolVisitor: new iteration");
            }

            repeatTrigger.reset();

            // Visit over all classes once.
            classPoolVisitor.visitClassPool(classPool);
        }
        while (repeatTrigger.isSet());

        if (DEBUG)
        {
            System.out.println("RepeatedClassPoolVisitor: done iterating");
        }
    }
}
